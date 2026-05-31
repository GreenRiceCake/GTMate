import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
import uuid
from urllib.parse import urlparse

import requests
import tkinter as tk
from tkinter import messagebox, ttk


if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CURR_VER_FILE = os.path.join(BASE_DIR, "curr_ver.json")
TARGET_PROGRAM = "GTMate.exe"
UPDATE_INFO_URL = os.environ.get(
    "GTMATE_UPDATE_INFO_URL",
    "https://raw.githubusercontent.com/GreenRiceCake/GTMate/main/update_manifest.json",
)
PENDING_SELF_UPDATE = None


def load_current_version():
    try:
        with open(CURR_VER_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("version", "0.0.0")
    except FileNotFoundError:
        return "0.0.0"
    except Exception:
        return "0.0.0"


def parse_version(version_text):
    parts = []
    for part in str(version_text).split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        parts.append(int(digits or 0))

    while len(parts) < 3:
        parts.append(0)

    return tuple(parts[:3])


def get_update_info():
    response = requests.get(UPDATE_INFO_URL, timeout=15)
    response.raise_for_status()
    return response.json()


def kill_program(process_name):
    try:
        subprocess.run(
            ["taskkill", "/f", "/im", process_name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except subprocess.CalledProcessError:
        pass


def download_file(url, destination, progress=None):
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress and total_size:
                        progress(downloaded, total_size)


def find_update_root(extract_dir):
    entries = [
        os.path.join(extract_dir, name)
        for name in os.listdir(extract_dir)
        if not name.startswith("__MACOSX")
    ]

    if len(entries) == 1 and os.path.isdir(entries[0]):
        nested_exe = os.path.join(entries[0], TARGET_PROGRAM)
        if os.path.exists(nested_exe):
            return entries[0]

    return extract_dir


def should_skip_path(relative_path):
    normalized = relative_path.replace("\\", "/").lower()
    preserved_files = {
        "bot_config.json",
    }

    return normalized in preserved_files


def write_current_version(new_version):
    with open(CURR_VER_FILE, "w", encoding="utf-8") as f:
        json.dump({"version": new_version}, f, indent=4)


def run_installer(installer_path):
    subprocess.Popen([installer_path], cwd=os.path.dirname(installer_path))


def update_from_exe(download_url, new_version, ui=None):
    if ui:
        ui.set_step("GTMate 종료 중...")
    kill_program(TARGET_PROGRAM)

    temp_dir = tempfile.mkdtemp(prefix="gtmate_update_")
    try:
        exe_path = os.path.join(temp_dir, TARGET_PROGRAM)
        if ui:
            ui.set_step("새 실행 파일 다운로드 중...")
        download_file(download_url, exe_path, ui.set_download_progress if ui else None)

        if ui:
            ui.set_step("실행 파일 교체 중...")
        shutil.copy2(exe_path, os.path.join(BASE_DIR, TARGET_PROGRAM))

        if ui:
            ui.set_step("버전 정보 갱신 중...")
        write_current_version(new_version)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def update_from_zip(download_url, new_version, ui=None):
    if ui:
        ui.set_step("GTMate 종료 중...")
    kill_program(TARGET_PROGRAM)
    time.sleep(0.5)

    temp_dir = tempfile.mkdtemp(prefix="gtmate_update_")
    try:
        zip_path = os.path.join(temp_dir, "update.zip")
        extract_dir = os.path.join(temp_dir, "extract")

        if ui:
            ui.set_step("업데이트 파일 다운로드 중...")
        download_file(download_url, zip_path, ui.set_download_progress if ui else None)

        if ui:
            ui.set_step("압축 해제 중...")
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.infolist()
            total_members = max(len(members), 1)
            for index, member in enumerate(members, start=1):
                zf.extract(member, extract_dir)
                if ui:
                    ui.set_file_progress(index, total_members, f"압축 해제: {member.filename}")

        update_root = find_update_root(extract_dir)
        if not os.path.exists(os.path.join(update_root, TARGET_PROGRAM)):
            raise RuntimeError("업데이트 zip 안에서 GTMate.exe를 찾을 수 없습니다.")

        if ui:
            ui.set_step("파일 교체 중...")
        copy_update_tree(update_root, BASE_DIR, ui)

        if ui:
            ui.set_step("버전 정보 갱신 중...")
        write_current_version(new_version)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def update_from_installer(download_url, ui=None):
    temp_dir = tempfile.mkdtemp(prefix="gtmate_installer_")
    parsed = urlparse(download_url)
    file_name = os.path.basename(parsed.path) or "GTMate_Installer.exe"
    if not file_name.lower().endswith(".exe"):
        file_name = "GTMate_Installer.exe"

    installer_path = os.path.join(temp_dir, file_name)
    if ui:
        ui.set_step("설치 파일 다운로드 중...")
    download_file(download_url, installer_path, ui.set_download_progress if ui else None)

    if ui:
        ui.set_step("설치 파일 실행 중...")
    run_installer(installer_path)


def collect_copy_jobs(source_dir, target_dir):
    jobs = []
    for root, dirs, files in os.walk(source_dir):
        relative_root = os.path.relpath(root, source_dir)
        if relative_root == ".":
            relative_root = ""

        for file_name in files:
            relative_path = os.path.normpath(os.path.join(relative_root, file_name))
            if should_skip_path(relative_path):
                continue

            source_path = os.path.join(root, file_name)
            target_path = os.path.join(target_dir, relative_path)
            jobs.append((source_path, target_path, relative_path))

    return jobs


def copy_update_tree(source_dir, target_dir, ui=None):
    global PENDING_SELF_UPDATE

    jobs = collect_copy_jobs(source_dir, target_dir)
    total_jobs = max(len(jobs), 1)

    for index, (source_path, target_path, relative_path) in enumerate(jobs, start=1):
        if relative_path.replace("\\", "/").lower() == "updater.exe":
            staged_path = os.path.join(target_dir, "Updater.new.exe")
            shutil.copy2(source_path, staged_path)
            PENDING_SELF_UPDATE = staged_path
            if ui:
                ui.set_file_progress(index, total_jobs, "Updater.exe는 종료 후 교체하도록 예약됨")
            continue

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(source_path, target_path)
        if ui:
            ui.set_file_progress(index, total_jobs, f"파일 교체: {relative_path}")


def schedule_self_update():
    if not PENDING_SELF_UPDATE:
        return

    updater_path = os.path.join(BASE_DIR, "Updater.exe")
    script_path = os.path.join(BASE_DIR, f"gtmate_updater_finish_{uuid.uuid4().hex}.cmd")
    commands = [
        "@echo off",
        "timeout /t 1 /nobreak >nul",
        "for /l %%i in (1,1,30) do (",
        f'    if not exist "{PENDING_SELF_UPDATE}" goto move_done',
        f'    move /y "{PENDING_SELF_UPDATE}" "{updater_path}" >nul 2>nul',
        f'    if not exist "{PENDING_SELF_UPDATE}" goto move_done',
        "    timeout /t 1 /nobreak >nul",
        ")",
        ":move_done",
        'del "%~f0" >nul 2>nul',
    ]

    with open(script_path, "w", encoding="utf-8") as f:
        f.write("\r\n".join(commands))

    subprocess.Popen(
        ["cmd", "/c", script_path],
        cwd=BASE_DIR,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


class UpdateUI:
    def __init__(self, root, update_info):
        self.root = root
        self.update_info = update_info
        self.completed = False

        self.status_var = tk.StringVar(value="업데이트 준비 중...")
        self.download_var = tk.StringVar(value="")
        self.file_var = tk.StringVar(value="")

        self.window = tk.Toplevel(root)
        self.window.title("GTMate 업데이트 진행")
        self.window.geometry("500x360")
        self.window.resizable(False, False)
        self.window.protocol("WM_DELETE_WINDOW", lambda: None)

        tk.Label(self.window, text="업데이트 진행 중", font=("Arial", 14, "bold")).pack(pady=(14, 4))
        tk.Label(self.window, textvariable=self.status_var, font=("Arial", 11)).pack(pady=(0, 8))

        self.download_bar = ttk.Progressbar(self.window, length=430, mode="determinate")
        self.download_bar.pack(pady=(2, 2))
        tk.Label(self.window, textvariable=self.download_var, font=("Arial", 9)).pack()

        self.file_bar = ttk.Progressbar(self.window, length=430, mode="determinate")
        self.file_bar.pack(pady=(10, 2))
        tk.Label(self.window, textvariable=self.file_var, font=("Arial", 9)).pack()

        self.log_text = tk.Text(self.window, height=9, width=58)
        self.log_text.config(state=tk.DISABLED)
        self.log_text.pack(pady=(10, 8))

        self.close_button = tk.Button(self.window, text="Close", state=tk.DISABLED, command=self.close)
        self.close_button.pack()

    def run_on_ui(self, callback):
        self.window.after(0, callback)

    def log(self, message):
        def update():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)

        self.run_on_ui(update)

    def set_step(self, message):
        self.run_on_ui(lambda: self.status_var.set(message))
        self.log(message)

    def set_download_progress(self, current, total):
        percent = int(current * 100 / total) if total else 0

        def update():
            self.download_bar["value"] = percent
            self.download_var.set(f"다운로드: {percent}% ({current // 1024 // 1024}MB / {total // 1024 // 1024}MB)")

        self.run_on_ui(update)

    def set_file_progress(self, current, total, message):
        percent = int(current * 100 / total) if total else 0

        def update():
            self.file_bar["value"] = percent
            self.file_var.set(f"{current} / {total}")

        self.run_on_ui(update)
        if current == 1 or current == total or current % 20 == 0:
            self.log(message)

    def complete(self, message):
        def update():
            self.completed = True
            self.status_var.set(message)
            self.download_bar["value"] = 100
            self.file_bar["value"] = 100
            self.close_button.config(state=tk.NORMAL)

        self.run_on_ui(update)
        self.log(message)
        if PENDING_SELF_UPDATE:
            self.log("Updater.exe는 창을 닫은 뒤 자동으로 교체됩니다.")

    def fail(self, error):
        def update():
            self.status_var.set("업데이트 실패")
            self.close_button.config(state=tk.NORMAL)
            messagebox.showerror("업데이트 실패", f"업데이트 중 오류가 발생했습니다:\n{error}")

        self.run_on_ui(update)
        self.log(f"오류: {error}")

    def close(self):
        self.window.destroy()
        if self.completed:
            schedule_self_update()
            sys.exit()


def update_program(update_info, parent):
    ui = UpdateUI(parent, update_info)

    def worker():
        try:
            new_version = update_info["version"]
            download_url = update_info["download_url"]
            update_type = update_info.get("update_type", "exe").lower()

            ui.log(f"업데이트 방식: {update_type}")

            if update_type == "zip":
                update_from_zip(download_url, new_version, ui)
                ui.complete(f"v{new_version} 업데이트 완료. 다시 실행해주세요.")
                return

            if update_type == "installer":
                ui.set_step("GTMate 종료 중...")
                kill_program(TARGET_PROGRAM)
                update_from_installer(download_url, ui)
                ui.complete("설치 파일을 실행했습니다. 설치 마법사를 따라 진행해주세요.")
                return

            if update_type == "exe":
                update_from_exe(download_url, new_version, ui)
                ui.complete(f"v{new_version} 업데이트 완료. 다시 실행해주세요.")
                return

            raise RuntimeError(f"지원하지 않는 업데이트 방식입니다: {update_type}")

        except Exception as e:
            ui.fail(e)

    threading.Thread(target=worker, daemon=True).start()


def main():
    try:
        current_version = load_current_version()
        data = get_update_info()
    except Exception as e:
        messagebox.showerror("업데이트 확인 실패", f"업데이트 정보를 가져올 수 없습니다:\n{e}")
        return

    latest_version = data.get("version", "0.0.0")
    changelog = data.get("changelog", "")
    title = data.get("title", "GTMate 업데이트")

    if parse_version(latest_version) <= parse_version(current_version):
        messagebox.showinfo("GTMate", f"현재 최신 버전(v{current_version})을 사용 중입니다.")
        return

    root = tk.Tk()
    root.title("GTMate 업데이트")
    root.geometry("440x360")

    label = tk.Label(root, text=f"업데이트 확인: {current_version} -> {latest_version}", font=("Arial", 14))
    label.pack(pady=10)

    title_label = tk.Label(root, text=title, font=("Arial", 11, "bold"))
    title_label.pack()

    changelog_text = tk.Text(root, height=13, width=52)
    changelog_text.insert(tk.END, changelog)
    changelog_text.config(state=tk.DISABLED)
    changelog_text.pack(pady=8)

    frame = tk.Frame(root)
    frame.pack(pady=10)

    def start_update():
        update_button.config(state=tk.DISABLED)
        ignore_button.config(state=tk.DISABLED)
        update_program(data, root)

    update_button = tk.Button(frame, text="Update", command=start_update)
    update_button.grid(row=0, column=0, padx=10)

    ignore_button = tk.Button(frame, text="Ignore", command=root.destroy)
    ignore_button.grid(row=0, column=1, padx=10)

    root.mainloop()


if __name__ == "__main__":
    main()
