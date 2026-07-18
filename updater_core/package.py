import hashlib
import os
import shutil
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import requests

from .paths import (
    APP_ID,
    CURRENT_VERSION_FILE,
    INSTALLED_MANIFEST_FILE,
    INSTALL_MARKER_FILE,
    TARGET_PROGRAM,
    UPDATER_PROGRAM,
    atomic_write_json,
    is_preserved_path,
    normalize_relative_path,
    read_json,
    relative_path_key,
    safe_join,
)


DEFAULT_MAX_ARCHIVE_FILES = 100_000
DEFAULT_MAX_UNCOMPRESSED_SIZE = 4 * 1024 * 1024 * 1024
DEFAULT_PRESERVE_PATHS = ["bot_config.json"]
ProgressCallback = Callable[[int, int, str], None]


def sha256_file(
    path: Path,
    progress: Optional[ProgressCallback] = None,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    current = 0
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            current += len(chunk)
            if progress:
                progress(current, total, f"검증 중: {path.name}")
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    expected_size: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
    timeout: int = 60,
    request_get: Callable[..., Any] = requests.get,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    try:
        partial.unlink(missing_ok=True)
        with request_get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            header_size = int(response.headers.get("content-length", 0) or 0)
            total = expected_size or header_size
            downloaded = 0
            with partial.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    file.write(chunk)
                    downloaded += len(chunk)
                    if progress:
                        progress(downloaded, total, "업데이트 파일 다운로드 중")
                file.flush()
                os.fsync(file.fileno())

        actual_size = partial.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            raise RuntimeError(
                f"다운로드 크기가 다릅니다: expected={expected_size}, actual={actual_size}"
            )
        os.replace(partial, destination)
        return destination
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def verify_download(
    path: Path,
    expected_sha256: Optional[str],
    expected_size: Optional[int],
    progress: Optional[ProgressCallback] = None,
) -> str:
    actual_size = path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise RuntimeError(
            f"패키지 크기가 다릅니다: expected={expected_size}, actual={actual_size}"
        )
    actual_sha256 = sha256_file(path, progress=progress)
    if expected_sha256 and actual_sha256.casefold() != expected_sha256.casefold():
        raise RuntimeError(
            "업데이트 패키지 SHA-256이 일치하지 않습니다. "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    return actual_sha256


def _zip_entry_path(name: str) -> str:
    text = str(name).replace("\\", "/")
    while text.endswith("/"):
        text = text[:-1]
    return normalize_relative_path(text)


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_IFMT(unix_mode) == stat.S_IFLNK


def safe_extract_zip(
    archive_path: Path,
    destination: Path,
    progress: Optional[ProgressCallback] = None,
    max_files: int = DEFAULT_MAX_ARCHIVE_FILES,
    max_uncompressed_size: int = DEFAULT_MAX_UNCOMPRESSED_SIZE,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = archive.infolist()
        if len(members) > max_files:
            raise RuntimeError(f"ZIP 파일 항목이 너무 많습니다: {len(members)}")

        total_uncompressed = sum(member.file_size for member in members)
        if total_uncompressed > max_uncompressed_size:
            raise RuntimeError(
                "압축 해제 크기가 허용 범위를 초과합니다: "
                f"{total_uncompressed} bytes"
            )

        total = max(len(members), 1)
        seen_paths: set[str] = set()
        for index, member in enumerate(members, start=1):
            if _is_zip_symlink(member):
                raise RuntimeError(f"ZIP의 심볼릭 링크는 허용하지 않습니다: {member.filename}")
            relative_path = _zip_entry_path(member.filename)
            path_key = relative_path_key(relative_path)
            if path_key in seen_paths:
                raise RuntimeError(f"ZIP에 중복된 경로가 있습니다: {relative_path}")
            seen_paths.add(path_key)
            target = safe_join(destination, relative_path)

            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            if progress:
                progress(index, total, f"압축 해제: {relative_path}")


def find_update_root(extract_dir: Path) -> Path:
    if (extract_dir / TARGET_PROGRAM).is_file():
        return extract_dir

    entries = [
        entry
        for entry in extract_dir.iterdir()
        if entry.name != "__MACOSX" and not entry.name.startswith("._")
    ]
    if len(entries) == 1 and entries[0].is_dir():
        nested = entries[0]
        if (nested / TARGET_PROGRAM).is_file():
            return nested
    raise RuntimeError(f"업데이트 ZIP 안에서 {TARGET_PROGRAM}을 찾을 수 없습니다.")


def _manifest_excluded(relative_path: str, preserve_paths: Iterable[str]) -> bool:
    key = relative_path_key(relative_path)
    metadata = {
        CURRENT_VERSION_FILE.casefold(),
        INSTALLED_MANIFEST_FILE.casefold(),
        INSTALL_MARKER_FILE.casefold(),
    }
    root_name = relative_path.rsplit("/", 1)[-1].casefold()
    root_stem, separator, root_extension = root_name.rpartition(".")
    inno_uninstaller = (
        "/" not in relative_path
        and bool(separator)
        and root_stem.startswith("unins")
        and root_stem[5:].isdigit()
        and root_extension in {"exe", "dat", "msg"}
    )
    return (
        key in metadata
        or inno_uninstaller
        or is_preserved_path(relative_path, preserve_paths)
    )


def build_file_manifest(
    root: Path,
    app_version: str,
    preserve_paths: Optional[Iterable[str]] = None,
    progress: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    preserve = list(preserve_paths or DEFAULT_PRESERVE_PATHS)
    candidates: list[tuple[str, Path]] = []
    for current_root, directories, files in os.walk(root):
        directories.sort(key=str.casefold)
        files.sort(key=str.casefold)
        current = Path(current_root)
        for file_name in files:
            path = current / file_name
            relative = path.relative_to(root).as_posix()
            normalized = normalize_relative_path(relative)
            if _manifest_excluded(normalized, preserve):
                continue
            candidates.append((normalized, path))

    manifest_files: list[dict[str, Any]] = []
    total = max(len(candidates), 1)
    for index, (relative, path) in enumerate(candidates, start=1):
        if progress:
            progress(index - 1, total, f"파일 해시 계산: {relative}")
        manifest_files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if progress:
        progress(total, total, "파일 매니페스트 생성 완료")

    return {
        "schema_version": 1,
        "app_id": APP_ID,
        "app_version": str(app_version),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preserve": preserve,
        "files": manifest_files,
    }


def validate_file_manifest(data: Any, require_updater: bool = True) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("파일 매니페스트의 최상위 값은 객체여야 합니다.")
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError("지원하지 않는 파일 매니페스트 버전입니다.")
    app_id = str(data.get("app_id") or APP_ID)
    if app_id != APP_ID:
        raise ValueError(f"다른 프로그램의 패키지입니다: {app_id}")

    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("파일 매니페스트에 files 목록이 없습니다.")

    normalized_files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("파일 매니페스트 항목은 객체여야 합니다.")
        path = normalize_relative_path(str(entry.get("path") or ""))
        key = relative_path_key(path)
        if key in seen:
            raise ValueError(f"중복된 파일 경로입니다: {path}")
        seen.add(key)
        size = int(entry.get("size", -1))
        sha256 = str(entry.get("sha256") or "").lower()
        if size < 0:
            raise ValueError(f"파일 크기가 올바르지 않습니다: {path}")
        if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise ValueError(f"파일 SHA-256이 올바르지 않습니다: {path}")
        normalized_files.append({"path": path, "size": size, "sha256": sha256})

    required = {TARGET_PROGRAM.casefold()}
    if require_updater:
        required.add(UPDATER_PROGRAM.casefold())
    missing = required - seen
    if missing:
        raise ValueError(f"필수 파일이 매니페스트에 없습니다: {', '.join(sorted(missing))}")

    preserve = data.get("preserve") or DEFAULT_PRESERVE_PATHS
    if not isinstance(preserve, list):
        raise ValueError("preserve 값은 배열이어야 합니다.")
    normalized_preserve = [normalize_relative_path(str(item).rstrip("/")) for item in preserve]

    result = dict(data)
    result["app_id"] = APP_ID
    result["app_version"] = str(data.get("app_version") or "0.0.0")
    result["files"] = normalized_files
    result["preserve"] = normalized_preserve
    return result


def load_file_manifest(path: Path, require_updater: bool = True) -> dict[str, Any]:
    data = read_json(path)
    if data is None:
        raise FileNotFoundError(path)
    return validate_file_manifest(data, require_updater=require_updater)


def load_or_build_package_manifest(
    update_root: Path,
    app_version: str,
    manifest_name: str = INSTALLED_MANIFEST_FILE,
    transactional: bool = False,
    progress: Optional[ProgressCallback] = None,
) -> tuple[dict[str, Any], Path, bool]:
    manifest_path = safe_join(update_root, manifest_name)
    if manifest_path.is_file():
        manifest = load_file_manifest(manifest_path, require_updater=transactional)
        valid_versions = {str(app_version)} if transactional else {"0.0.0", str(app_version)}
        if manifest["app_version"] not in valid_versions:
            raise ValueError(
                "패키지 버전과 파일 매니페스트 버전이 다릅니다: "
                f"{app_version} != {manifest['app_version']}"
            )
        return manifest, manifest_path, False

    if transactional:
        raise FileNotFoundError(
            f"Transactional package에 {manifest_name}이(가) 없습니다."
        )
    manifest = build_file_manifest(update_root, app_version, progress=progress)
    manifest = validate_file_manifest(manifest, require_updater=False)
    atomic_write_json(manifest_path, manifest)
    return manifest, manifest_path, True


def verify_payload(
    update_root: Path,
    manifest: dict[str, Any],
    progress: Optional[ProgressCallback] = None,
) -> None:
    files = manifest["files"]
    total = max(len(files), 1)
    for index, entry in enumerate(files, start=1):
        path = safe_join(update_root, entry["path"])
        if not path.is_file():
            raise RuntimeError(f"패키지 파일이 없습니다: {entry['path']}")
        actual_size = path.stat().st_size
        if actual_size != entry["size"]:
            raise RuntimeError(
                f"패키지 파일 크기가 다릅니다: {entry['path']} "
                f"expected={entry['size']}, actual={actual_size}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != entry["sha256"]:
            raise RuntimeError(f"패키지 파일 SHA-256이 다릅니다: {entry['path']}")
        if progress:
            progress(index, total, f"파일 검증: {entry['path']}")
