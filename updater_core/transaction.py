import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .package import DEFAULT_PRESERVE_PATHS, load_file_manifest, sha256_file
from .paths import (
    APP_ID,
    CURRENT_VERSION_FILE,
    INSTALLED_MANIFEST_FILE,
    INSTALL_MARKER_FILE,
    atomic_write_json,
    is_preserved_path,
    normalize_relative_path,
    read_json,
    relative_path_key,
    safe_join,
    updater_state_root,
    validate_install_directory,
)


ProgressCallback = Callable[[int, int, str], None]
FINAL_STATES = {"committed", "rolled_back", "cancelled"}
METADATA_FILES = [CURRENT_VERSION_FILE, INSTALLED_MANIFEST_FILE, INSTALL_MARKER_FILE]
LARGE_TRANSACTION_DIRS = ("download", "staging", "backup")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_transaction_workspace(state_root: Optional[Path] = None) -> dict[str, Path]:
    root = state_root or updater_state_root()
    transaction_id = uuid.uuid4().hex
    transaction_dir = root / "transactions" / transaction_id
    paths = {
        "transaction_dir": transaction_dir,
        "download_dir": transaction_dir / "download",
        "extract_dir": transaction_dir / "staging",
        "backup_dir": transaction_dir / "backup",
        "journal_path": transaction_dir / "transaction.json",
    }
    for key in ("download_dir", "extract_dir", "backup_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def load_transaction(journal_path: Path) -> dict[str, Any]:
    data = read_json(journal_path)
    if not isinstance(data, dict):
        raise ValueError(f"Transaction Journal을 읽을 수 없습니다: {journal_path}")
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError("지원하지 않는 Transaction Journal 버전입니다.")
    data["journal_path"] = str(journal_path.resolve())
    return data


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def cleanup_transaction_payload(journal_path: Path) -> bool:
    """Remove large files only after a transaction no longer needs rollback."""
    journal = load_transaction(journal_path)
    if journal.get("state") not in FINAL_STATES:
        return False
    transaction_dir = journal_path.resolve().parent
    for name in LARGE_TRANSACTION_DIRS:
        _remove_path(transaction_dir / name)
    return True


def cleanup_finalized_transactions(state_root: Optional[Path] = None) -> list[Path]:
    root = (state_root or updater_state_root()) / "transactions"
    if not root.is_dir():
        return []
    removed: list[Path] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        journal_path = directory / "transaction.json"
        if not journal_path.is_file():
            continue
        try:
            journal = load_transaction(journal_path)
            if journal.get("state") not in FINAL_STATES:
                continue
            _remove_path(directory)
            removed.append(directory)
        except (OSError, ValueError):
            # A just-finished worker executable can still be locked. The large
            # payload is still safe to remove and the rest is retried later.
            try:
                cleanup_transaction_payload(journal_path)
            except (OSError, ValueError):
                continue
    return removed


def save_transaction(journal: dict[str, Any]) -> None:
    journal["updated_at"] = utc_now()
    path = Path(journal["journal_path"])
    serialized = dict(journal)
    serialized.pop("journal_path", None)
    atomic_write_json(path, serialized)


def set_transaction_state(
    journal: dict[str, Any], state: str, error: Optional[str] = None
) -> None:
    journal["state"] = state
    if error is not None:
        journal["error"] = str(error)
    save_transaction(journal)


def _manifest_file_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {relative_path_key(entry["path"]): entry for entry in manifest["files"]}


def _load_current_manifest(install_dir: Path) -> Optional[dict[str, Any]]:
    path = install_dir / INSTALLED_MANIFEST_FILE
    if not path.is_file():
        return None
    try:
        return load_file_manifest(path, require_updater=False)
    except Exception:
        return None


def create_transaction_journal(
    workspace: dict[str, Path],
    install_dir: Path,
    payload_dir: Path,
    incoming_manifest: dict[str, Any],
    from_version: str,
    to_version: str,
) -> Path:
    install_dir = validate_install_directory(install_dir)
    preserve = list(
        dict.fromkeys(DEFAULT_PRESERVE_PATHS + list(incoming_manifest.get("preserve") or []))
    )
    for entry in incoming_manifest["files"]:
        if is_preserved_path(entry["path"], preserve):
            raise ValueError(
                f"보존 경로를 관리 파일로 덮어쓸 수 없습니다: {entry['path']}"
            )

    old_manifest = _load_current_manifest(install_dir)
    incoming_map = _manifest_file_map(incoming_manifest)
    old_map = _manifest_file_map(old_manifest) if old_manifest else {}
    obsolete = [
        entry["path"]
        for key, entry in old_map.items()
        if key not in incoming_map and not is_preserved_path(entry["path"], preserve)
    ]
    obsolete.sort(key=str.casefold)

    transaction_id = workspace["transaction_dir"].name
    journal: dict[str, Any] = {
        "schema_version": 1,
        "transaction_id": transaction_id,
        "state": "prepared",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "from_version": str(from_version),
        "to_version": str(to_version),
        "install_dir": str(install_dir),
        "payload_dir": str(payload_dir.resolve()),
        "backup_dir": str(workspace["backup_dir"].resolve()),
        "incoming_manifest": incoming_manifest,
        "old_manifest": old_manifest,
        "obsolete_files": obsolete,
        "preserve": preserve,
        "original_files": [],
        "original_metadata": [],
        "backup_completed": False,
        "applied_files": [],
        "deleted_files": [],
        "error": None,
        "journal_path": str(workspace["journal_path"].resolve()),
    }
    save_transaction(journal)
    return workspace["journal_path"]


def _copy_backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def backup_transaction(
    journal: dict[str, Any], progress: Optional[ProgressCallback] = None
) -> None:
    install_dir = validate_install_directory(Path(journal["install_dir"]))
    backup_dir = Path(journal["backup_dir"])
    incoming = [entry["path"] for entry in journal["incoming_manifest"]["files"]]
    affected = sorted(set(incoming + list(journal["obsolete_files"])), key=str.casefold)

    set_transaction_state(journal, "backing_up")
    original_files: list[str] = []
    total = max(len(affected) + len(METADATA_FILES), 1)
    current = 0
    for relative in affected:
        source = safe_join(install_dir, relative)
        if source.is_file():
            destination = safe_join(backup_dir / "files", relative)
            _copy_backup(source, destination)
            original_files.append(relative)
        current += 1
        if progress:
            progress(current, total, f"기존 파일 백업: {relative}")

    original_metadata: list[str] = []
    for name in METADATA_FILES:
        source = install_dir / name
        if source.is_file():
            destination = backup_dir / "metadata" / name
            _copy_backup(source, destination)
            original_metadata.append(name)
        current += 1
        if progress:
            progress(current, total, f"설치 정보 백업: {name}")

    journal["original_files"] = original_files
    journal["original_metadata"] = original_metadata
    journal["backup_completed"] = True
    set_transaction_state(journal, "backup_complete")


def _atomic_replace_file(source: Path, target: Path, transaction_id: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".gtmate-new-{transaction_id}")
    temporary.unlink(missing_ok=True)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _remove_empty_parents(path: Path, stop: Path) -> None:
    current = path
    stop = stop.resolve()
    while current != stop:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def verify_installed_payload(
    install_dir: Path,
    manifest: dict[str, Any],
    progress: Optional[ProgressCallback] = None,
) -> None:
    files = manifest["files"]
    total = max(len(files), 1)
    for index, entry in enumerate(files, start=1):
        target = safe_join(install_dir, entry["path"])
        if not target.is_file():
            raise RuntimeError(f"설치된 파일이 없습니다: {entry['path']}")
        if target.stat().st_size != entry["size"]:
            raise RuntimeError(f"설치된 파일 크기가 다릅니다: {entry['path']}")
        if sha256_file(target) != entry["sha256"]:
            raise RuntimeError(f"설치된 파일 SHA-256이 다릅니다: {entry['path']}")
        if progress:
            progress(index, total, f"설치 결과 검증: {entry['path']}")


def _write_install_metadata(journal: dict[str, Any]) -> None:
    install_dir = Path(journal["install_dir"])
    atomic_write_json(
        install_dir / INSTALLED_MANIFEST_FILE,
        journal["incoming_manifest"],
    )
    atomic_write_json(
        install_dir / CURRENT_VERSION_FILE,
        {"version": journal["to_version"]},
    )
    existing_marker = read_json(install_dir / INSTALL_MARKER_FILE, default={})
    created_at = (
        existing_marker.get("created_at")
        if isinstance(existing_marker, dict)
        else None
    ) or utc_now()
    atomic_write_json(
        install_dir / INSTALL_MARKER_FILE,
        {
            "schema_version": 1,
            "app_id": APP_ID,
            "install_dir": str(install_dir.resolve()),
            "version": journal["to_version"],
            "created_at": created_at,
            "updated_at": utc_now(),
        },
    )


def apply_transaction(
    journal_path: Path,
    progress: Optional[ProgressCallback] = None,
    registration_callback: Optional[Callable[[Path, str], None]] = None,
) -> dict[str, Any]:
    journal = load_transaction(journal_path)
    install_dir = validate_install_directory(Path(journal["install_dir"]))
    payload_dir = Path(journal["payload_dir"])
    incoming_files = journal["incoming_manifest"]["files"]
    transaction_id = journal["transaction_id"]

    try:
        if journal["state"] not in {"backup_complete", "applying", "verifying"}:
            backup_transaction(journal, progress=progress)

        set_transaction_state(journal, "applying")
        total = max(len(incoming_files) + len(journal["obsolete_files"]), 1)
        current = 0
        applied_files = list(journal.get("applied_files") or [])
        for entry in incoming_files:
            relative = normalize_relative_path(entry["path"])
            source = safe_join(payload_dir, relative)
            target = safe_join(install_dir, relative)
            _atomic_replace_file(source, target, transaction_id)
            if relative not in applied_files:
                applied_files.append(relative)
            journal["applied_files"] = applied_files
            current += 1
            if current % 25 == 0:
                save_transaction(journal)
            if progress:
                progress(current, total, f"파일 적용: {relative}")

        deleted_files = list(journal.get("deleted_files") or [])
        for relative in journal["obsolete_files"]:
            target = safe_join(install_dir, relative)
            if target.is_file():
                target.unlink()
                _remove_empty_parents(target.parent, install_dir)
            if relative not in deleted_files:
                deleted_files.append(relative)
            journal["deleted_files"] = deleted_files
            current += 1
            if progress:
                progress(current, total, f"이전 관리 파일 삭제: {relative}")
        save_transaction(journal)

        set_transaction_state(journal, "verifying")
        verify_installed_payload(
            install_dir,
            journal["incoming_manifest"],
            progress=progress,
        )
        _write_install_metadata(journal)

        if registration_callback:
            try:
                registration_callback(install_dir, journal["to_version"])
                journal["registration_updated"] = True
            except Exception as registration_error:
                journal["registration_updated"] = False
                journal["registration_error"] = str(registration_error)

        set_transaction_state(journal, "committed")
        return journal
    except Exception as error:
        set_transaction_state(journal, "failed", error=str(error))
        try:
            rollback_transaction(journal_path, progress=progress)
        except Exception as rollback_error:
            journal = load_transaction(journal_path)
            journal["rollback_error"] = str(rollback_error)
            set_transaction_state(journal, "rollback_failed", error=str(error))
            raise RuntimeError(
                f"업데이트 실패 후 롤백도 실패했습니다: {error}; rollback={rollback_error}"
            ) from error
        raise


def rollback_transaction(
    journal_path: Path,
    progress: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    journal = load_transaction(journal_path)
    if not journal.get("backup_completed"):
        set_transaction_state(journal, "cancelled")
        return journal
    install_dir = validate_install_directory(Path(journal["install_dir"]), require_identity=False)
    backup_dir = Path(journal["backup_dir"])
    original_keys = {relative_path_key(path) for path in journal.get("original_files") or []}
    incoming = [entry["path"] for entry in journal["incoming_manifest"]["files"]]
    affected = sorted(set(incoming + list(journal["obsolete_files"])), key=str.casefold)

    set_transaction_state(journal, "rolling_back")
    total = max(len(affected) + len(METADATA_FILES), 1)
    current = 0
    for relative in affected:
        target = safe_join(install_dir, relative)
        backup = safe_join(backup_dir / "files", relative)
        if relative_path_key(relative) in original_keys:
            if not backup.is_file():
                raise RuntimeError(f"롤백 백업 파일이 없습니다: {relative}")
            _atomic_replace_file(backup, target, journal["transaction_id"] + "-rollback")
        elif target.is_file():
            target.unlink()
            _remove_empty_parents(target.parent, install_dir)
        current += 1
        if progress:
            progress(current, total, f"파일 복원: {relative}")

    original_metadata = set(journal.get("original_metadata") or [])
    for name in METADATA_FILES:
        target = install_dir / name
        backup = backup_dir / "metadata" / name
        if name in original_metadata:
            if not backup.is_file():
                raise RuntimeError(f"설치 정보 백업이 없습니다: {name}")
            _atomic_replace_file(backup, target, journal["transaction_id"] + "-meta")
        else:
            target.unlink(missing_ok=True)
        current += 1
        if progress:
            progress(current, total, f"설치 정보 복원: {name}")

    set_transaction_state(journal, "rolled_back")
    return journal


def find_incomplete_transactions(state_root: Optional[Path] = None) -> list[Path]:
    root = (state_root or updater_state_root()) / "transactions"
    if not root.is_dir():
        return []
    results: list[tuple[float, Path]] = []
    for directory in root.iterdir():
        journal_path = directory / "transaction.json"
        if not journal_path.is_file():
            continue
        try:
            journal = load_transaction(journal_path)
        except Exception:
            continue
        if journal.get("state") not in FINAL_STATES:
            results.append((journal_path.stat().st_mtime, journal_path))
    results.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in results]


def wait_for_path_unlock(path: Path, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with path.open("a+b"):
                return True
        except OSError:
            time.sleep(0.25)
    return False
