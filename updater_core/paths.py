import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


APP_NAME = "GTMate"
APP_ID = "{CCD0FA1D-EDC7-4D1B-99C0-737D84807422}"
TARGET_PROGRAM = "GTMate.exe"
UPDATER_PROGRAM = "Updater.exe"
CURRENT_VERSION_FILE = "curr_ver.json"
INSTALLED_MANIFEST_FILE = "installed_manifest.json"
INSTALL_MARKER_FILE = ".gtmate-install.json"


def runtime_base_dir() -> Path:
    import sys

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def updater_state_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / APP_NAME / "Updater"
    return Path(tempfile.gettempdir()) / APP_NAME / "Updater"


def normalize_relative_path(value: str) -> str:
    text = str(value).replace("\\", "/").strip()
    if not text or "\x00" in text:
        raise ValueError("빈 경로나 NUL 문자가 포함된 경로는 사용할 수 없습니다.")
    if text.startswith("/") or text.startswith("//"):
        raise ValueError(f"절대 경로는 사용할 수 없습니다: {value}")

    parts = text.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"안전하지 않은 상대 경로입니다: {value}")
    if ":" in parts[0]:
        raise ValueError(f"드라이브 경로는 사용할 수 없습니다: {value}")

    return "/".join(parts)


def relative_path_key(value: str) -> str:
    return normalize_relative_path(value).casefold()


def safe_join(root: Path, relative_path: str) -> Path:
    normalized = normalize_relative_path(relative_path)
    root_resolved = root.resolve()
    candidate = root_resolved.joinpath(*normalized.split("/")).resolve(strict=False)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"루트 밖으로 벗어나는 경로입니다: {relative_path}") from exc
    return candidate


def is_preserved_path(relative_path: str, preserve_paths: Iterable[str]) -> bool:
    key = relative_path_key(relative_path)
    for preserve in preserve_paths:
        normalized = normalize_relative_path(str(preserve).rstrip("/"))
        preserve_key = normalized.casefold()
        if key == preserve_key or key.startswith(preserve_key + "/"):
            return True
    return False


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return default


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for root, _, files in os.walk(path):
        for file_name in files:
            try:
                total += (Path(root) / file_name).stat().st_size
            except OSError:
                continue
    return total


def validate_install_directory(path: Path, require_identity: bool = True) -> Path:
    resolved = path.resolve()
    anchor = Path(resolved.anchor).resolve()
    home = Path.home().resolve()
    windows_dir_text = os.environ.get("WINDIR")
    protected = {anchor, home}
    if windows_dir_text:
        protected.add(Path(windows_dir_text).resolve())

    if resolved in protected:
        raise ValueError(f"안전하지 않은 설치 경로입니다: {resolved}")
    if len(resolved.parts) < 2:
        raise ValueError(f"설치 경로가 너무 짧습니다: {resolved}")

    if require_identity:
        identity_files = (
            resolved / TARGET_PROGRAM,
            resolved / INSTALLED_MANIFEST_FILE,
            resolved / INSTALL_MARKER_FILE,
        )
        if not any(candidate.exists() for candidate in identity_files):
            raise ValueError(f"GTMate 설치 폴더로 확인할 수 없습니다: {resolved}")

    return resolved
