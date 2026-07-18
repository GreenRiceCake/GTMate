import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import tkinter as tk
from tkinter import messagebox, ttk

from updater_core.manifest import (
    DEFAULT_UPDATE_INFO_URL,
    UPDATER_PROTOCOL_VERSION,
    UpdateInfo,
    fetch_update_info,
    is_newer_version,
    load_current_version,
)
from updater_core.package import (
    download_file,
    find_update_root,
    load_or_build_package_manifest,
    safe_extract_zip,
    verify_download,
    verify_payload,
)
from updater_core.paths import (
    CURRENT_VERSION_FILE,
    INSTALLED_MANIFEST_FILE,
    TARGET_PROGRAM,
    atomic_write_json,
    runtime_base_dir,
    updater_state_root,
    validate_install_directory,
)
from updater_core.transaction import (
    cleanup_finalized_transactions,
    cleanup_transaction_payload,
    create_transaction_journal,
    create_transaction_workspace,
    find_incomplete_transactions,
    load_transaction,
    rollback_transaction,
    apply_transaction,
)
from updater_core.windows import (
    IS_WINDOWS,
    is_admin,
    launch_apply_worker,
    launch_uninstall_worker,
    launch_worker,
    perform_uninstall,
    repair_install_registration,
    start_gtmate,
    terminate_program,
    update_install_registration,
    wait_for_process_exit,
)


BASE_DIR = runtime_base_dir()
UPDATE_INFO_URL = os.environ.get("GTMATE_UPDATE_INFO_URL", DEFAULT_UPDATE_INFO_URL)


class ProgressWindow:
    def __init__(self, title: str, close_locked: bool = True):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("560x410")
        self.root.resizable(False, False)
        self.close_locked = close_locked
        self.finished = False
        self.install_dir: Optional[Path] = None
        self._last_logged_message = ""

        self.status_var = tk.StringVar(value="준비 중...")
        self.download_var = tk.StringVar(value="")
        self.file_var = tk.StringVar(value="")

        tk.Label(self.root, text=title, font=("Arial", 15, "bold")).pack(
            pady=(14, 4)
        )
        tk.Label(self.root, textvariable=self.status_var, font=("Arial", 10)).pack(
            pady=(0, 8)
        )

        self.download_bar = ttk.Progressbar(self.root, length=480, mode="determinate")
        self.download_bar.pack(pady=(2, 2))
        tk.Label(self.root, textvariable=self.download_var, font=("Arial", 9)).pack()

        self.file_bar = ttk.Progressbar(self.root, length=480, mode="determinate")
        self.file_bar.pack(pady=(10, 2))
        tk.Label(self.root, textvariable=self.file_var, font=("Arial", 9)).pack()

        self.log_text = tk.Text(self.root, height=11, width=67)
        self.log_text.config(state=tk.DISABLED)
        self.log_text.pack(pady=(10, 8))

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=(0, 8))
        self.restart_button = tk.Button(
            self.button_frame,
            text="GTMate 다시 실행",
            state=tk.DISABLED,
            command=self.restart_gtmate,
        )
        self.restart_button.grid(row=0, column=0, padx=6)
        self.close_button = tk.Button(
            self.button_frame,
            text="닫기",
            state=tk.DISABLED if close_locked else tk.NORMAL,
            command=self.root.destroy,
        )
        self.close_button.grid(row=0, column=1, padx=6)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        if self.close_locked and not self.finished:
            return
        self.root.destroy()

    def on_ui(self, callback: Callable[[], None]) -> None:
        self.root.after(0, callback)

    def log(self, message: str) -> None:
        message = str(message)

        def update():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)

        self.on_ui(update)

    def set_status(self, message: str, log: bool = True) -> None:
        self.on_ui(lambda: self.status_var.set(message))
        if log and message != self._last_logged_message:
            self._last_logged_message = message
            self.log(message)

    def progress(self, current: int, total: int, message: str) -> None:
        percent = int(current * 100 / total) if total else 0
        percent = max(0, min(percent, 100))
        is_download = message.startswith("업데이트 파일 다운로드")

        def update():
            self.status_var.set(message)
            if is_download:
                self.download_bar["value"] = percent
                if total:
                    self.download_var.set(
                        f"다운로드 {percent}% "
                        f"({current // 1024 // 1024}MB / {total // 1024 // 1024}MB)"
                    )
                else:
                    self.download_var.set(f"다운로드 {current // 1024 // 1024}MB")
            else:
                self.file_bar["value"] = percent
                self.file_var.set(f"{percent}% ({current} / {total})")

        self.on_ui(update)
        if current in {1, total} or (current and current % 50 == 0):
            self.log(message)

    def complete(self, message: str, install_dir: Optional[Path] = None) -> None:
        self.install_dir = install_dir

        def update():
            self.finished = True
            self.status_var.set(message)
            self.download_bar["value"] = 100
            self.file_bar["value"] = 100
            self.close_button.config(state=tk.NORMAL)
            if install_dir and (install_dir / TARGET_PROGRAM).is_file():
                self.restart_button.config(state=tk.NORMAL)

        self.on_ui(update)
        self.log(message)

    def fail(self, error: Exception | str) -> None:
        message = str(error)

        def update():
            self.finished = True
            self.status_var.set("작업 실패")
            self.close_button.config(state=tk.NORMAL)
            messagebox.showerror("GTMate Updater", message, parent=self.root)

        self.on_ui(update)
        self.log(f"오류: {message}")

    def restart_gtmate(self) -> None:
        if self.install_dir:
            start_gtmate(self.install_dir)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def show_update_prompt(info: UpdateInfo, current_version: str, repair: bool = False) -> bool:
    root = tk.Tk()
    root.title("GTMate 복구" if repair else "GTMate 업데이트")
    root.geometry("500x405")
    root.resizable(False, False)
    accepted = {"value": False}

    heading = "현재 버전을 다시 설치합니다" if repair else (
        f"업데이트 확인: {current_version} -> {info.version}"
    )
    tk.Label(root, text=heading, font=("Arial", 14, "bold")).pack(pady=(14, 8))
    tk.Label(root, text=info.title, font=("Arial", 11, "bold")).pack()

    changelog = tk.Text(root, height=15, width=58, wrap=tk.WORD)
    changelog.insert(tk.END, info.changelog or "변경 내역이 없습니다.")
    changelog.config(state=tk.DISABLED)
    changelog.pack(padx=12, pady=10)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=5)

    def accept():
        accepted["value"] = True
        root.destroy()

    tk.Button(
        button_frame,
        text="복구" if repair else "업데이트",
        width=12,
        command=accept,
    ).grid(row=0, column=0, padx=8)
    tk.Button(button_frame, text="취소", width=12, command=root.destroy).grid(
        row=0, column=1, padx=8
    )
    root.mainloop()
    return accepted["value"]


def _prepare_zip_update(info: UpdateInfo, ui: ProgressWindow) -> None:
    workspace = create_transaction_workspace()
    archive_path = workspace["download_dir"] / "update.zip"
    apply_worker_started = False
    try:
        ui.set_status("업데이트 파일 다운로드 중")
        download_file(
            info.package.url,
            archive_path,
            expected_size=info.package.size,
            progress=ui.progress,
        )
        ui.set_status("다운로드 파일 검증 중")
        actual_sha256 = verify_download(
            archive_path,
            info.package.sha256,
            info.package.size,
            progress=ui.progress,
        )
        if not info.package.sha256:
            ui.log(f"참고: 원격 SHA-256 미지정, 계산값={actual_sha256}")

        ui.set_status("압축 해제 중")
        safe_extract_zip(archive_path, workspace["extract_dir"], progress=ui.progress)
        update_root = find_update_root(workspace["extract_dir"])

        ui.set_status("패키지 파일 검증 중")
        manifest, manifest_path, generated = load_or_build_package_manifest(
            update_root,
            info.version,
            manifest_name=info.package.package_manifest,
            transactional=info.package.transactional,
            progress=ui.progress,
        )
        if generated:
            ui.log("레거시 ZIP에서 임시 파일 매니페스트를 생성했습니다.")
        else:
            ui.log(f"파일 매니페스트 로드: {manifest_path.name}")
        verify_payload(update_root, manifest, progress=ui.progress)

        journal_path = create_transaction_journal(
            workspace,
            BASE_DIR,
            update_root,
            manifest,
            load_current_version(BASE_DIR),
            info.version,
        )
        ui.log(f"Transaction 준비 완료: {journal_path}")
        ui.set_status("관리자 권한으로 파일 교체 준비 중")
        launch_apply_worker(journal_path, elevated=IS_WINDOWS)
        apply_worker_started = True
        ui.on_ui(ui.root.destroy)
    finally:
        if not apply_worker_started:
            shutil.rmtree(workspace["transaction_dir"], ignore_errors=True)


def _prepare_installer_update(info: UpdateInfo, ui: ProgressWindow) -> None:
    workspace = create_transaction_workspace()
    installer_path = workspace["download_dir"] / "GTMate_Installer.exe"
    download_file(
        info.package.url,
        installer_path,
        expected_size=info.package.size,
        progress=ui.progress,
    )
    verify_download(
        installer_path,
        info.package.sha256,
        info.package.size,
        progress=ui.progress,
    )
    terminate_program(TARGET_PROGRAM)
    subprocess.Popen([str(installer_path)], cwd=str(installer_path.parent))
    ui.complete("설치 프로그램을 실행했습니다.")


def _prepare_exe_update(info: UpdateInfo, ui: ProgressWindow) -> None:
    workspace = create_transaction_workspace()
    downloaded = workspace["download_dir"] / TARGET_PROGRAM
    download_file(
        info.package.url,
        downloaded,
        expected_size=info.package.size,
        progress=ui.progress,
    )
    verify_download(
        downloaded,
        info.package.sha256,
        info.package.size,
        progress=ui.progress,
    )
    terminate_program(TARGET_PROGRAM)
    target = BASE_DIR / TARGET_PROGRAM
    backup = workspace["backup_dir"] / TARGET_PROGRAM
    if target.is_file():
        shutil.copy2(target, backup)
    temporary = target.with_name(target.name + ".new")
    try:
        shutil.copy2(downloaded, temporary)
        os.replace(temporary, target)
        atomic_write_json(BASE_DIR / CURRENT_VERSION_FILE, {"version": info.version})
    except Exception:
        temporary.unlink(missing_ok=True)
        if backup.is_file():
            shutil.copy2(backup, target)
        raise
    ui.complete(f"v{info.version} 실행 파일 업데이트 완료", BASE_DIR)


def run_prepare_update(info: UpdateInfo) -> None:
    ui = ProgressWindow("GTMate 업데이트 진행")

    def worker():
        try:
            ui.log(f"업데이트 형식: {info.package.package_type}")
            ui.log(f"Transactional package: {info.package.transactional}")
            if info.package.package_type == "zip":
                _prepare_zip_update(info, ui)
            elif info.package.package_type == "installer":
                _prepare_installer_update(info, ui)
            else:
                _prepare_exe_update(info, ui)
        except Exception as error:
            ui.fail(error)

    threading.Thread(target=worker, daemon=True).start()
    ui.run()


def _launch_recovery_worker(journal_path: Path) -> None:
    host_dir = journal_path.parent / "recovery-host"
    launch_worker(
        ["--rollback", str(journal_path), "--parent-pid", str(os.getpid())],
        host_dir,
        elevated=IS_WINDOWS,
    )


def handle_incomplete_transaction() -> bool:
    incomplete = find_incomplete_transactions()
    if not incomplete:
        return False
    journal_path = incomplete[0]
    journal = load_transaction(journal_path)
    root = tk.Tk()
    root.withdraw()
    should_restore = messagebox.askyesno(
        "중단된 업데이트 발견",
        "완료되지 않은 GTMate 업데이트가 있습니다.\n\n"
        f"대상 버전: {journal.get('to_version')}\n"
        f"상태: {journal.get('state')}\n\n"
        "이전 버전으로 복구하시겠습니까?",
        parent=root,
    )
    root.destroy()
    if should_restore:
        _launch_recovery_worker(journal_path)
    return True


def run_normal_mode(repair: bool = False) -> None:
    cleanup_finalized_transactions()
    if handle_incomplete_transaction():
        return
    try:
        current_version = load_current_version(BASE_DIR)
        info = fetch_update_info(UPDATE_INFO_URL)
    except Exception as error:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "업데이트 확인 실패",
            f"업데이트 정보를 가져올 수 없습니다:\n{error}",
            parent=root,
        )
        root.destroy()
        return

    if not repair and not is_newer_version(info.version, current_version):
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "GTMate",
            f"현재 최신 버전(v{current_version})을 사용 중입니다.",
            parent=root,
        )
        root.destroy()
        return
    if not show_update_prompt(info, current_version, repair=repair):
        return
    run_prepare_update(info)


def run_apply_mode(journal_path: Path, parent_pid: int) -> None:
    ui = ProgressWindow("GTMate 업데이트 적용")

    def worker():
        try:
            ui.set_status("기존 Updater 종료 대기 중")
            if parent_pid and not wait_for_process_exit(parent_pid, timeout=45):
                raise RuntimeError("기존 Updater가 종료되지 않았습니다.")
            ui.set_status("GTMate 종료 중")
            terminate_program(TARGET_PROGRAM)

            def registration_callback(path: Path, version: str):
                if not update_install_registration(path, version, create_if_missing=False):
                    raise RuntimeError("기존 제어판 등록 정보를 찾지 못했습니다.")

            result = apply_transaction(
                journal_path,
                progress=ui.progress,
                registration_callback=registration_callback,
            )
            if result.get("registration_updated") is False:
                ui.log(
                    "제어판 정보 갱신은 보류되었습니다: "
                    + str(result.get("registration_error") or "unknown")
                )
            try:
                cleanup_transaction_payload(journal_path)
                ui.log("업데이트 임시 파일 정리 완료")
            except Exception as cleanup_error:
                ui.log(f"업데이트 임시 파일 정리 보류: {cleanup_error}")
            ui.complete(
                f"v{result['to_version']} 업데이트 완료",
                Path(result["install_dir"]),
            )
        except Exception as error:
            ui.fail(error)

    threading.Thread(target=worker, daemon=True).start()
    ui.run()


def run_rollback_mode(journal_path: Path, parent_pid: int) -> None:
    ui = ProgressWindow("GTMate 업데이트 복구")

    def worker():
        try:
            if parent_pid and not wait_for_process_exit(parent_pid, timeout=45):
                raise RuntimeError("기존 Updater가 종료되지 않았습니다.")
            terminate_program(TARGET_PROGRAM)
            result = rollback_transaction(journal_path, progress=ui.progress)
            install_dir = Path(result["install_dir"])
            version = load_current_version(install_dir)
            try:
                update_install_registration(install_dir, version, create_if_missing=False)
            except Exception as registration_error:
                ui.log(f"제어판 정보 복구 보류: {registration_error}")
            try:
                cleanup_transaction_payload(journal_path)
                ui.log("복구 임시 파일 정리 완료")
            except Exception as cleanup_error:
                ui.log(f"복구 임시 파일 정리 보류: {cleanup_error}")
            ui.complete("이전 버전 복구 완료", install_dir)
        except Exception as error:
            ui.fail(error)

    threading.Thread(target=worker, daemon=True).start()
    ui.run()


def run_registration_mode(install_dir: Path, version: str, quiet: bool) -> None:
    install_dir = install_dir.resolve()
    if IS_WINDOWS and not is_admin():
        host_dir = updater_state_root() / "registration" / str(int(time.time()))
        arguments = [
            "--repair-registration",
            "--install-dir",
            str(install_dir),
            "--version",
            version,
            "--quiet",
        ]
        try:
            launch_worker(arguments, host_dir, elevated=True)
        except Exception:
            if not quiet:
                raise
        return
    try:
        repair_install_registration(install_dir, version)
        if not quiet:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("GTMate", "제어판 설치 정보를 복구했습니다.", parent=root)
            root.destroy()
    except Exception as error:
        if quiet:
            error_path = updater_state_root() / "registration_error.json"
            atomic_write_json(
                error_path,
                {
                    "install_dir": str(install_dir),
                    "version": str(version),
                    "error": str(error),
                },
            )
            return
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("GTMate", f"설치 정보 복구 실패:\n{error}", parent=root)
        root.destroy()


def run_uninstall_mode(install_dir: Path, quiet: bool) -> None:
    remove_user_data = False
    if not quiet:
        root = tk.Tk()
        root.title("GTMate 제거")
        root.geometry("430x220")
        root.resizable(False, False)
        confirmed = {"value": False}
        selection = {"remove_user_data": False}
        remove_var = tk.BooleanVar(value=False)
        tk.Label(root, text="GTMate를 제거하시겠습니까?", font=("Arial", 14, "bold")).pack(
            pady=(22, 10)
        )
        tk.Label(root, text=f"제거 경로: {install_dir}", wraplength=390).pack(pady=4)
        tk.Checkbutton(
            root,
            text="AppData에 저장된 사용자 설정과 스킨도 제거",
            variable=remove_var,
        ).pack(pady=12)

        buttons = tk.Frame(root)
        buttons.pack()

        def confirm():
            confirmed["value"] = True
            selection["remove_user_data"] = bool(remove_var.get())
            root.destroy()

        tk.Button(buttons, text="제거", width=12, command=confirm).grid(
            row=0, column=0, padx=8
        )
        tk.Button(buttons, text="취소", width=12, command=root.destroy).grid(
            row=0, column=1, padx=8
        )
        root.mainloop()
        if not confirmed["value"]:
            return
        remove_user_data = selection["remove_user_data"]

    launch_uninstall_worker(
        install_dir,
        remove_user_data=remove_user_data,
        elevated=IS_WINDOWS,
    )


def run_uninstall_worker(
    install_dir: Path,
    parent_pid: int,
    remove_user_data: bool,
    quiet: bool,
) -> None:
    if quiet:
        try:
            if parent_pid and not wait_for_process_exit(parent_pid, timeout=45):
                raise RuntimeError("기존 Updater가 종료되지 않았습니다.")
            perform_uninstall(install_dir, remove_user_data=remove_user_data)
        except Exception as error:
            atomic_write_json(
                updater_state_root() / "uninstall_error.json",
                {
                    "install_dir": str(install_dir),
                    "remove_user_data": bool(remove_user_data),
                    "error": str(error),
                },
            )
        return

    ui = ProgressWindow("GTMate 제거")

    def worker():
        try:
            if parent_pid and not wait_for_process_exit(parent_pid, timeout=45):
                raise RuntimeError("기존 Updater가 종료되지 않았습니다.")
            perform_uninstall(
                install_dir,
                remove_user_data=remove_user_data,
                progress=ui.progress,
            )
            ui.complete("GTMate 제거 완료")
        except Exception as error:
            ui.fail(error)

    threading.Thread(target=worker, daemon=True).start()
    ui.run()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--apply", type=Path)
    parser.add_argument("--rollback", type=Path)
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--repair-registration", action="store_true")
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--uninstall-worker", type=Path)
    parser.add_argument("--install-dir", type=Path)
    parser.add_argument("--version", default="")
    parser.add_argument("--remove-user-data", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--self-test", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.self_test:
        atomic_write_json(
            arguments.self_test.resolve(),
            {
                "ok": True,
                "updater_protocol": UPDATER_PROTOCOL_VERSION,
                "base_dir": str(BASE_DIR),
                "frozen": bool(getattr(sys, "frozen", False)),
            },
        )
        return
    if arguments.apply:
        run_apply_mode(arguments.apply, arguments.parent_pid)
        return
    if arguments.rollback:
        run_rollback_mode(arguments.rollback, arguments.parent_pid)
        return
    if arguments.uninstall_worker:
        run_uninstall_worker(
            arguments.uninstall_worker,
            arguments.parent_pid,
            arguments.remove_user_data,
            arguments.quiet,
        )
        return

    install_dir = (arguments.install_dir or BASE_DIR).resolve()
    if arguments.repair_registration:
        version = arguments.version or load_current_version(install_dir)
        run_registration_mode(install_dir, version, arguments.quiet)
        return
    if arguments.uninstall:
        run_uninstall_mode(install_dir, arguments.quiet)
        return
    run_normal_mode(repair=arguments.repair)


if __name__ == "__main__":
    main()
