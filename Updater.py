import requests
import sys
import subprocess
import os
import json
import tkinter as tk
from tkinter import messagebox

# 파일 경로 설정
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CURR_VER_FILE = os.path.join(BASE_DIR, "curr_ver.json")
TARGET_PROGRAM = "GTMate.exe"

# 최신 버전 정보 URL (예: GitHub raw 링크)
UPDATE_INFO_URL = "https://raw.githubusercontent.com/GreenRiceCake/GTMate/main/version.json"

# 1. 현재 버전 로드
def load_current_version():
    try:
        with open(CURR_VER_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("version")
    except FileNotFoundError:
        return "0.0.0"

# 2. 최신 버전 정보 가져오기
def get_update_info():
    response = requests.get(UPDATE_INFO_URL)
    response.raise_for_status()
    return response.json()

# 3. 대상 프로그램 강제 종료
def kill_program(process_name):
    try:
        subprocess.run(["taskkill", "/f", "/im", process_name], check=True)
    except subprocess.CalledProcessError:
        pass  # 이미 종료됐거나 실행 중이 아니면 무시

# 4. 업데이트 실행
def update_program(download_url, new_version):
    try:
        kill_program(TARGET_PROGRAM)

        # 파일 다운로드
        response = requests.get(download_url)
        response.raise_for_status()

        # exe 덮어쓰기
        with open(os.path.join(BASE_DIR, TARGET_PROGRAM), "wb") as f:
            f.write(response.content)

        # 버전 정보 업데이트
        with open(CURR_VER_FILE, "w", encoding="utf-8") as f:
            json.dump({"version": new_version}, f, indent=4)

        messagebox.showinfo("업데이트 완료", f"v{new_version}로 업데이트되었습니다. 다시 실행해주세요.")
        sys.exit()

    except Exception as e:
        messagebox.showerror("업데이트 실패", f"업데이트 중 오류가 발생했습니다:\n{e}")

# 5. 메인 로직 (GUI)
def main():
    current_version = load_current_version()
    data = get_update_info()
    latest_version = data["version"]
    changelog = data.get("changelog", "")
    download_url = data["download_url"]

    if latest_version > current_version:
        root = tk.Tk()
        root.title("GTMate 업데이트")
        root.geometry("400x300")

        label = tk.Label(root, text=f"업데이트 확인: {current_version} ➔ {latest_version}", font=("Arial", 14))
        label.pack(pady=10)

        changelog_label = tk.Label(root, text="변경사항:", font=("Arial", 12))
        changelog_label.pack()

        changelog_text = tk.Text(root, height=10, width=45)
        changelog_text.insert(tk.END, changelog)
        changelog_text.config(state=tk.DISABLED)
        changelog_text.pack()

        frame = tk.Frame(root)
        frame.pack(pady=10)

        update_button = tk.Button(frame, text="Update", command=lambda: update_program(download_url, latest_version))
        update_button.grid(row=0, column=0, padx=10)

        ignore_button = tk.Button(frame, text="Ignore", command=root.destroy)
        ignore_button.grid(row=0, column=1, padx=10)

        root.mainloop()
    else:
        messagebox.showinfo("GTMate", f"현재 최신 버전(v{current_version})을 사용 중입니다.")

if __name__ == "__main__":
    main()
