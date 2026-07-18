import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .paths import (
    APP_ID,
    APP_NAME,
    INSTALL_MARKER_FILE,
    INSTALLED_MANIFEST_FILE,
    TARGET_PROGRAM,
    UPDATER_PROGRAM,
    atomic_write_json,
    directory_size,
    read_json,
    updater_state_root,
    validate_install_directory,
)


IS_WINDOWS = os.name == "nt"
UNINSTALL_KEY = (
    "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\"
    + APP_ID
    + "_is1"
)
ProgressCallback = Callable[[int, int, str], None]


def is_admin() -> bool:
    if not IS_WINDOWS:
        return True
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def process_exists(process_id: int) -> bool:
    if process_id <= 0:
        return False
    if IS_WINDOWS:
        synchronize = 0x00100000
        wait_object_0 = 0x00000000
        wait_timeout = 0x00000102
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, process_id)
        if not handle:
            return False
        try:
            wait_result = ctypes.windll.kernel32.WaitForSingleObject(handle, 0)
            if wait_result == wait_timeout:
                return True
            if wait_result == wait_object_0:
                return False
            return False
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(process_id, 0)
        return True
    except OSError:
        return False


def wait_for_process_exit(process_id: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_exists(process_id):
            return True
        time.sleep(0.2)
    return not process_exists(process_id)


def terminate_program(image_name: str = TARGET_PROGRAM, timeout: float = 8.0) -> None:
    if not IS_WINDOWS:
        return
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.run(
        ["taskkill", "/im", image_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        creationflags=creation_flags,
    )
    time.sleep(min(timeout, 2.0))
    subprocess.run(
        ["taskkill", "/f", "/im", image_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        creationflags=creation_flags,
    )


def _worker_executable(host_dir: Path) -> tuple[str, list[str]]:
    host_dir.mkdir(parents=True, exist_ok=True)
    if getattr(sys, "frozen", False):
        host_path = host_dir / f"GTMateUpdateHost-{os.getpid()}.exe"
        shutil.copy2(Path(sys.executable), host_path)
        return str(host_path), []
    script = Path(sys.argv[0]).resolve()
    return sys.executable, [str(script)]


def launch_worker(
    arguments: list[str],
    host_dir: Path,
    elevated: bool = False,
) -> None:
    executable, prefix = _worker_executable(host_dir)
    all_arguments = prefix + arguments
    if IS_WINDOWS and elevated and not is_admin():
        parameters = subprocess.list2cmdline(all_arguments)
        result = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            executable,
            parameters,
            str(host_dir),
            1,
        )
        if result <= 32:
            raise RuntimeError(f"관리자 권한 작업을 시작하지 못했습니다: code={result}")
        return

    creation_flags = 0
    if IS_WINDOWS:
        creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(
        [executable, *all_arguments],
        cwd=str(host_dir),
        creationflags=creation_flags,
        close_fds=True,
    )


def launch_apply_worker(journal_path: Path, elevated: bool = True) -> None:
    host_dir = journal_path.parent / "host"
    launch_worker(
        ["--apply", str(journal_path), "--parent-pid", str(os.getpid())],
        host_dir,
        elevated=elevated,
    )


def launch_uninstall_worker(
    install_dir: Path,
    remove_user_data: bool,
    elevated: bool = True,
) -> None:
    host_dir = Path(tempfile.gettempdir()) / APP_NAME / "Uninstall" / str(int(time.time()))
    arguments = [
        "--uninstall-worker",
        str(install_dir.resolve()),
        "--parent-pid",
        str(os.getpid()),
    ]
    if remove_user_data:
        arguments.append("--remove-user-data")
    launch_worker(arguments, host_dir, elevated=elevated)


def start_gtmate(install_dir: Path) -> None:
    executable = install_dir / TARGET_PROGRAM
    if executable.is_file():
        subprocess.Popen([str(executable)], cwd=str(install_dir))


def _registry_locations(write: bool = False):
    if not IS_WINDOWS:
        return []
    import winreg

    access = winreg.KEY_READ | (winreg.KEY_WRITE if write else 0)
    views = [winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY]
    return [
        (winreg.HKEY_LOCAL_MACHINE, access | view)
        for view in views
    ] + [
        (winreg.HKEY_CURRENT_USER, access | view)
        for view in views
    ]


def _find_registration(write: bool = False):
    if not IS_WINDOWS:
        return None
    import winreg

    for root, access in _registry_locations(write=write):
        try:
            key = winreg.OpenKey(root, UNINSTALL_KEY, 0, access)
            return root, access, key
        except OSError:
            continue
    return None


def registered_install_location() -> Optional[Path]:
    if not IS_WINDOWS:
        return None
    import winreg

    found = _find_registration(write=False)
    if not found:
        return None
    _, _, key = found
    try:
        value, _ = winreg.QueryValueEx(key, "InstallLocation")
        return Path(value).resolve() if value else None
    except OSError:
        return None
    finally:
        winreg.CloseKey(key)


def _write_registration_values(key, install_dir: Path, version: str) -> None:
    import winreg

    updater = install_dir / UPDATER_PROGRAM
    gt_mate = install_dir / TARGET_PROGRAM
    uninstall_command = f'"{updater}" --uninstall'
    winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, APP_NAME)
    winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, str(version))
    winreg.SetValueEx(key, "DisplayIcon", 0, winreg.REG_SZ, str(gt_mate))
    winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, str(install_dir))
    winreg.SetValueEx(key, "Publisher", 0, winreg.REG_SZ, "GreenRC")
    winreg.SetValueEx(
        key,
        "URLInfoAbout",
        0,
        winreg.REG_SZ,
        "https://github.com/GreenRiceCake/GTMate",
    )
    winreg.SetValueEx(key, "UninstallString", 0, winreg.REG_SZ, uninstall_command)
    winreg.SetValueEx(
        key,
        "QuietUninstallString",
        0,
        winreg.REG_SZ,
        uninstall_command + " --quiet",
    )
    winreg.SetValueEx(
        key,
        "EstimatedSize",
        0,
        winreg.REG_DWORD,
        min(directory_size(install_dir) // 1024, 0xFFFFFFFF),
    )
    winreg.SetValueEx(
        key,
        "InstallDate",
        0,
        winreg.REG_SZ,
        datetime.now().strftime("%Y%m%d"),
    )
    winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
    winreg.SetValueEx(key, "NoRepair", 0, winreg.REG_DWORD, 0)


def update_install_registration(
    install_dir: Path,
    version: str,
    create_if_missing: bool = False,
) -> bool:
    if not IS_WINDOWS:
        return False
    import winreg

    install_dir = validate_install_directory(install_dir)
    found = _find_registration(write=True)
    if found:
        _, _, key = found
        try:
            _write_registration_values(key, install_dir, version)
            return True
        finally:
            winreg.CloseKey(key)

    if not create_if_missing:
        return False

    root = winreg.HKEY_LOCAL_MACHINE if is_admin() else winreg.HKEY_CURRENT_USER
    view = winreg.KEY_WOW64_64KEY
    key = winreg.CreateKeyEx(root, UNINSTALL_KEY, 0, winreg.KEY_WRITE | view)
    try:
        _write_registration_values(key, install_dir, version)
        return True
    finally:
        winreg.CloseKey(key)


def write_install_marker(install_dir: Path, version: str) -> None:
    install_dir = validate_install_directory(install_dir)
    marker_path = install_dir / INSTALL_MARKER_FILE
    existing = read_json(marker_path, default={})
    created_at = existing.get("created_at") if isinstance(existing, dict) else None
    atomic_write_json(
        marker_path,
        {
            "schema_version": 1,
            "app_id": APP_ID,
            "install_dir": str(install_dir),
            "version": str(version),
            "created_at": created_at or datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        },
    )


def repair_install_registration(install_dir: Path, version: str) -> bool:
    install_dir = validate_install_directory(install_dir)
    manifest_path = install_dir / INSTALLED_MANIFEST_FILE
    if not manifest_path.is_file():
        from .package import build_file_manifest, validate_file_manifest

        manifest = build_file_manifest(install_dir, version)
        manifest = validate_file_manifest(manifest, require_updater=True)
        atomic_write_json(manifest_path, manifest)

    updated = update_install_registration(install_dir, version, create_if_missing=True)
    if not updated:
        raise RuntimeError("제어판 설치 정보를 만들 수 없습니다.")
    write_install_marker(install_dir, version)
    return True


def _delete_registration() -> None:
    if not IS_WINDOWS:
        return
    import winreg

    for root, access in _registry_locations(write=True):
        try:
            key = winreg.OpenKey(root, UNINSTALL_KEY, 0, access)
            winreg.CloseKey(key)
            view = access & (winreg.KEY_WOW64_64KEY | winreg.KEY_WOW64_32KEY)
            winreg.DeleteKeyEx(root, UNINSTALL_KEY, view, 0)
        except FileNotFoundError:
            continue
        except OSError:
            continue


def _shortcut_candidates() -> list[Path]:
    candidates: list[Path] = []
    app_data = os.environ.get("APPDATA")
    user_profile = os.environ.get("USERPROFILE")
    public = os.environ.get("PUBLIC")
    if app_data:
        candidates.append(
            Path(app_data) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "GTMate.lnk"
        )
    if user_profile:
        candidates.append(Path(user_profile) / "Desktop" / "GTMate.lnk")
    if public:
        candidates.append(Path(public) / "Desktop" / "GTMate.lnk")
    return candidates


def validate_uninstall_target(install_dir: Path) -> Path:
    install_dir = validate_install_directory(install_dir)
    marker = read_json(install_dir / INSTALL_MARKER_FILE, default={})
    marker_valid = (
        isinstance(marker, dict)
        and marker.get("app_id") == APP_ID
        and Path(str(marker.get("install_dir", ""))).resolve() == install_dir
    )
    registered = registered_install_location()
    registration_valid = registered is not None and registered == install_dir
    if not marker_valid and not registration_valid:
        raise RuntimeError("설치 마커 또는 제어판 등록 정보가 일치하지 않습니다.")
    if not (install_dir / TARGET_PROGRAM).exists() and not (
        install_dir / INSTALLED_MANIFEST_FILE
    ).exists():
        raise RuntimeError("GTMate 프로그램 파일을 확인할 수 없습니다.")
    return install_dir


def perform_uninstall(
    install_dir: Path,
    remove_user_data: bool = False,
    progress: Optional[ProgressCallback] = None,
) -> None:
    install_dir = validate_uninstall_target(install_dir)
    if progress:
        progress(1, 4, "GTMate 종료 중")
    terminate_program(TARGET_PROGRAM)

    if progress:
        progress(2, 4, "바로가기 제거 중")
    for shortcut in _shortcut_candidates():
        shortcut.unlink(missing_ok=True)

    if progress:
        progress(3, 4, "프로그램 파일 제거 중")
    shutil.rmtree(install_dir)

    if remove_user_data:
        app_data = os.environ.get("APPDATA")
        if app_data:
            user_data = Path(app_data) / APP_NAME
            if user_data.is_dir():
                shutil.rmtree(user_data)

    shutil.rmtree(updater_state_root(), ignore_errors=True)

    _delete_registration()
    if progress:
        progress(4, 4, "GTMate 제거 완료")
