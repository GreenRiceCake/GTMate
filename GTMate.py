import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
import socket
import os
import sys
import struct
import time
import requests
import subprocess
import asyncio
import json
import threading
import numpy as np
import discord
from discord import opus
from discord.ext import commands, voice_recv
import vosk
from dataclasses import dataclass
from typing import Optional
import threading

# pycryptodome의 Salsa20
try:
    from Crypto.Cipher import Salsa20
except ImportError:
    print("[오류] pycryptodome이 설치되지 않았습니다!")
    exit(1)

SHARED_GAME_STATE = {
    "fuel_liters": 0.0,
    "fuel_percent": 0.0,
    "laps_remain": 0.0, # 계산된 남은 랩 수
    "current_lap": 0,
    "total_laps": 0,
    "best_lap_ms": -1,
    "last_lap_ms": -1,
    "rank": 0,
    "total_cars": 0,
    "on_track": False
}

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, "bot_config.json")
PIPER_EXE = os.path.join(BASE_DIR, "bin", "piper.exe")
PIPER_MODEL = os.path.join(BASE_DIR, "models", "piper", "ttsmodel.onnx") 
VOSK_MODEL_PATH = os.path.join(BASE_DIR, "models", "vosk")
FFMPEG_EXE = os.path.join(BASE_DIR, "bin", "ffmpeg.exe")
UPDATER_EXE = os.path.join(BASE_DIR, "Updater.exe")
opus_path = os.path.join(BASE_DIR, "bin", "libopus.dll")

try:
    if not discord.opus.is_loaded():
        # 파일 이름을 직접 주거나 풀 경로를 줍니다.
        discord.opus.load_opus(opus_path)
        print(f">>> [성공] Opus 로드 완료: {opus_path}")
except Exception as e:
    print(f">>> [실패] Opus 로드 에러: {e}")

# [추가] 봇 상태 상수
STATE_IDLE = 0
STATE_WAITING_COMMAND = 1
STATE_WAITING_FOLLOWUP = 2

# [추가] 동의어 사전 (Alias)
COMMAND_ALIASES = {
    "wake": ["engineer", "mate", "chief", "radio", "hello", "hey", "new", "near", "gin", "beer", "ate"],
    "fuel": ["fuel", "gas", "petrol", "consumption", "tank", "few", "fill", "few all"],
    "rank": ["rank", "position", "place", "where am i"],
    "current_lap": ["current lap", "lap", "current"],
    "best_lap": ["best", "fastest", "record", "lap time"],
    "last_lap": ["last", "previous", "lap time"],
    "no": ["no", "nope", "negative", "cancel", "nothing", "done", "thanks", "thank you"]
}

@dataclass
class TelemetryPacket:
    position: tuple; velocity: tuple; rotation: tuple
    speed: float; rpm: float; fuel_level: float; fuel_capacity: float
    throttle: int; brake: int; current_gear: int; suggested_gear: int
    tire_temps: tuple; tire_radius: tuple; wheel_rps: tuple
    packet_id: int; lap_count: int; total_laps: int; best_lap: int; last_lap: int
    race_rank: int; total_cars: int
    flags: int; boost: float; oil_pressure: float; water_temp: float; oil_temp: float
    timestamp: float

class GT7Flags:
    CAR_ON_TRACK = 1 << 0
    PAUSED = 1 << 1
    HAS_TURBO = 1 << 4
    REV_LIMITER = 1 << 5
    HANDBRAKE = 1 << 6
    HIGH_BEAM = 1 << 8
    ASM_ACTIVE = 1 << 10
    TCS_ACTIVE = 1 << 11
    @staticmethod
    def check(flags, flag): return bool(flags & flag)

class EngineerBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
        
        # Vosk 초기화
        if not os.path.exists(VOSK_MODEL_PATH):
            print(f"[오류] Vosk 모델 없음: {VOSK_MODEL_PATH}")
        self.vosk_model = vosk.Model(VOSK_MODEL_PATH)
        self.recognizer = vosk.KaldiRecognizer(self.vosk_model, 16000)
        self.recognizer.SetWords(True) 
        
        self.state = STATE_IDLE
        self.last_interaction_time = 0
        self.audio_queue = asyncio.Queue()
        self.is_speaking = False
        self.processing_task = None

        # self.recognizer = vosk.KaldiRecognizer(self.vosk_model, 16000) 아래에 추가
        self.recognizer.SetWords(True)
        self.recognizer.SetPartialWords(True) # 부분 단어 인식을 더 공격적으로 수행

        # 기존 self.recognizer 부분을 아래처럼 변경해 보세요.
        # 허용할 단어 리스트 (레이싱 관련 및 호출어)
        grammar_list = [
            "engineer", "mate", "radio", "hello", # 호출어
            "fuel", "rank", "position", "lap", "previous", "gap", "best", # 명령어
            "yes", "no", "copy", "thanks", "cancel", # 응답
            "[unk]" # 리스트에 없는 단어 처리용
        ]
        grammar_json = json.dumps(grammar_list)
        self.recognizer = vosk.KaldiRecognizer(self.vosk_model, 16000, grammar_json)

    async def on_ready(self):
        print(f'>>> [Bot] Logged in as {self.user}')
        # 채널 자동 접속 로직 (설정 파일의 채널 ID 우선)
        cfg = self.load_config()
        channel_id = cfg.get("CHANNEL_ID")
        channel = self.get_channel(int(channel_id)) if channel_id else None
        
        if not channel: # 설정 없으면 첫 번째 음성 채널 찾기
            for guild in self.guilds:
                if guild.voice_channels:
                    channel = guild.voice_channels[0]
                    break
        
        if channel:
            print(f">>> [Bot] Joining {channel.name}")
            self.voice_client = await channel.connect(cls=voice_recv.VoiceRecvClient)
            self.voice_client.listen(voice_recv.BasicSink(self.sink_callback))
            await self.speak_tts("Radio check. Connected.")
            if not self.processing_task:
                self.processing_task = self.loop.create_task(self.process_audio_queue())

    def load_config(self):
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f: return json.load(f)
        return {}

    def sink_callback(self, user, data: voice_recv.VoiceData):
        if user == self.user: return
        
        # 1. PCM 데이터를 numpy 배열로 변환
        audio_array = np.frombuffer(data.pcm, dtype=np.int16)
        if audio_array.size == 0: return

        # 2. 스테레오 -> 모노 변환 (더 선명한 방식)
        # 단순히 평균을 내기보다 왼쪽/오른쪽 채널 중 하나만 쓰거나 더 정교하게 합칩니다.
        mono_audio = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)

        # 3. 48000Hz -> 16000Hz (3:1 다운샘플링)
        # 슬라이싱을 사용하여 오디오 데이터의 선명도를 유지합니다.
        resampled_audio = mono_audio[::3] 

        asyncio.run_coroutine_threadsafe(
            self.audio_queue.put(resampled_audio.tobytes()), 
            self.loop
        )

    def match_keyword(self, text, target_keys):
        for key in target_keys:
            for alias in COMMAND_ALIASES.get(key, []):
                if alias in text: return key
        return None

    async def process_audio_queue(self):
        print(">>> [Bot] Listening...")
        while True:
            # 1. 큐 관리 (너무 쌓이면 비우기)
            if self.audio_queue.qsize() > 20:
                while self.audio_queue.qsize() > 1:
                    self.audio_queue.get_nowait()
            
            # 2. 데이터 가져오기 (여기서 딱 한 번만!)
            pcm_data = await self.audio_queue.get()
            
            try:
                # 3. 말하는 중이면 무시
                if self.is_speaking:
                    continue # 위에서 이미 get()을 했으므로 그냥 넘어가면 됨

                # 4. 인식 처리
                self.recognizer.AcceptWaveform(pcm_data)
                partial = json.loads(self.recognizer.PartialResult()).get("partial", "").lower()
                
                if not partial: continue

                # [중요] 타임아웃 리셋
                self.last_interaction_time = time.time()

                # 상태 머신 로직
                if self.state == STATE_IDLE:
                    if self.match_keyword(partial, ["wake"]):
                        self.recognizer.Reset()
                        await self.handle_interaction("wake")

                elif self.state == STATE_WAITING_COMMAND:
                    cmd = self.match_keyword(partial, ["fuel", "rank", "best_lap", "last_lap", "current_lap", "no"])
                    if cmd:
                        self.recognizer.Reset()
                        await self.handle_interaction(cmd)

                elif self.state == STATE_WAITING_FOLLOWUP:
                    cmd = self.match_keyword(partial, ["no", "fuel", "rank", "best_lap", "last_lap", "current_lap"])
                    if cmd:
                        self.recognizer.Reset()
                        await self.handle_interaction(cmd)

            except Exception as e:
                print(f"Loop Error: {e}")
                await asyncio.sleep(0.1)

    async def handle_interaction(self, key):
        self.is_speaking = True
        
        # 호출어 대응
        if key == "wake":
            await self.speak_tts("Yes mate, Go ahead.")
            self.state = STATE_WAITING_COMMAND
            # 봇이 말을 마친 '지금' 시간을 기록
            self.last_interaction_time = time.time() 
            self.loop.create_task(self.check_timeout(15, STATE_WAITING_COMMAND))
        
        elif key == "no":
            self.state = STATE_IDLE
            await self.speak_tts("Copy that. Standing by.")
            self.recognizer.Reset()

        else:
            response = self.get_telemetry_response(key)
            if response:
                await self.speak_tts(response)
                await self.speak_tts("Anything else?") # 이 말이 끝날 때까지 아래 코드는 대기함
                
                # [수정 핵심] 봇이 "Anything else?"라고 물어본 직후에 시간을 초기화
                self.last_interaction_time = time.time() 
                self.state = STATE_WAITING_FOLLOWUP
                
                # 대기 시간을 10초에서 15초로 늘리는 것을 추천합니다.
                self.loop.create_task(self.check_timeout(15, STATE_WAITING_FOLLOWUP))
            else:
                await self.speak_tts("I don't have that data.")
                self.last_interaction_time = time.time() # 여기서도 시간 갱신
                self.state = STATE_WAITING_COMMAND

        self.is_speaking = False
        self.recognizer.Reset()

    async def check_timeout(self, duration, monitor_state):
        await asyncio.sleep(duration)
        if self.state == monitor_state:
            if time.time() - self.last_interaction_time >= duration:
                if monitor_state == STATE_WAITING_COMMAND:
                    await self.speak_tts("Standing by.")
                elif monitor_state == STATE_WAITING_FOLLOWUP:
                    await self.speak_tts("Radio out.")
                self.state = STATE_IDLE
                self.recognizer.Reset()

    def reset_audio_engine(self):
        try:
            # 1. 말하기/듣기 플래그 강제 리셋
            self.is_speaking = False
            self.is_listening = True
            
            # 2. 쌓여있는 오디오 데이터(Queue)가 있다면 모두 비우기
            # (인식기가 피트 소음을 처리하느라 밀려있는 걸 방지)
            if hasattr(self, 'audio_queue'):
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                    except:
                        break
            
            # 3. 인식기(Recognizer) 초기화 (사용 중인 엔진에 따라 다름)
            # Vosk 등을 사용한다면 여기서 AcceptWaveform을 초기화하는 로직이 들어갈 수 있습니다.
            
            print(">>> [Bot] Audio engine has been hard-reset.")
        except Exception as e:
            print(f">>> [Bot] Reset failed: {e}")

    async def restart_listener(self):
        try:
            # 1. 기존 인식 태스크가 있다면 안전하게 취소
            if self.processing_task and not self.processing_task.done():
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
                await asyncio.sleep(0.1)
            
            # 2. 쌓여있는 피트 소음(오디오 큐) 비우기
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except:
                    break
            
            # 3. [핵심] 실제 루프 함수인 'process_audio_queue' 다시 실행
            self.processing_task = self.loop.create_task(self.process_audio_queue())
            
            # 4. 상태 초기화
            self.is_speaking = False
            self.state = STATE_IDLE # 대기 상태로 리셋
            self.recognizer.Reset()
            
            print(">>> [Bot] 음성 인식 루프(process_audio_queue) 재시작 성공.")
        except Exception as e:
            print(f">>> [Bot] 재시작 실패: {e}")

    # [핵심] 실제 GTMate 데이터 읽기
    def get_telemetry_response(self, key):
        data = SHARED_GAME_STATE
        
        if key == "fuel":
            liters = int(data['fuel_liters'])
            laps = f"{data['laps_remain']:.1f}"
            return f"Fuel is {liters} liters. That's about {data['laps_remain']:.1f} laps."
        
        elif key == "rank":
            return f"Position {data['rank']} out of {data['total_cars']}."

        elif key == "current_lap":
            return f"Lap {data['current_lap']}."
        
        elif key == "best_lap":
            if data['best_lap_ms'] <= 0: return "No best lap set yet."
            return f"Best lap is {self.format_time_tts(data['best_lap_ms'])}."
            
        elif key == "last_lap":
            if data['last_lap_ms'] <= 0: return "No last lap data."
            return f"Last lap was {self.format_time_tts(data['last_lap_ms'])}."
            
        return None

    def format_time_tts(self, ms):
        minutes = ms // 60000
        seconds = (ms % 60000) // 1000
        point = (ms % 1000) // 100
        text = ""
        if minutes > 0: text += f"{minutes} minute "
        text += f"{seconds} point {point} seconds"
        return text

    async def speak_tts(self, text):
        if not self.voice_client or not self.voice_client.is_connected(): return
        if not text.strip(): return
        
        self.is_speaking = True
        creation_flags = subprocess.CREATE_NO_WINDOW 
        
        import os
        import traceback  # 에러 추적을 위해 추가
        
        current_env = os.environ.copy()
        current_env["PYTHONIOENCODING"] = "utf-8"
        
        current_ffmpeg = os.path.abspath(os.path.join(BASE_DIR, "bin", "ffmpeg.exe"))
        current_piper = os.path.abspath(os.path.join(BASE_DIR, "bin", "piper.exe"))
        abs_model_path = os.path.abspath(PIPER_MODEL if os.path.isabs(PIPER_MODEL) else os.path.join(BASE_DIR, PIPER_MODEL))

        try:
            # 1. Piper 실행
            piper_cmd = [current_piper, "--model", abs_model_path, "--output-raw"]
            process = subprocess.Popen(
                piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=creation_flags, cwd=os.path.dirname(current_piper), env=current_env
            )
            out_data, err = process.communicate(input=text.encode('utf-8'), timeout=15)
            
            # 2. FFmpeg 실행
            ffmpeg_cmd = [current_ffmpeg, "-f", "s16le", "-ar", "22050", "-ac", "1", "-i", "-",
                          "-f", "s16le", "-ar", "48000", "-ac", "2", "-"]
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=creation_flags, cwd=os.path.dirname(current_ffmpeg), env=current_env
            )
            pcm_converted, ff_err = ffmpeg_proc.communicate(input=out_data, timeout=15)
            
            if not pcm_converted:
                print(">>> [에러] 변환된 PCM 데이터가 없습니다.")
                return

            # 3. 재생 단계 (여기가 핵심 의심 구간)
            import io
            print(">>> [디버그] 재생 시도 직전...")
            audio_source = discord.PCMAudio(io.BytesIO(pcm_converted))
            
            if self.voice_client.is_playing(): 
                self.voice_client.stop()
            
            self.voice_client.play(audio_source)
            print(">>> [디버그] 재생 함수 호출 성공")
            
            while self.voice_client.is_playing():
                await asyncio.sleep(0.1)
                
        except Exception as e:
            # 에러의 타입과 상세한 발생 위치를 콘솔에 뿌립니다.
            print(f">>> [Bot] TTS Fatal Error 발생!")
            print(f">>> 에러 종류: {type(e).__name__}")
            print(f">>> 에러 메시지: {e}")
            traceback.print_exc() # <--- 이게 범인을 잡아줄 겁니다.
            
        finally:
            self.is_speaking = False
            print(">>> [디버그] speak_tts 종료")

class GT7TelemetryReceiver:
    KEY = b'Simulator Interface Packet GT7 ver 0.0'
    XOR_MAGIC = {'A': 0xDEADBEAF, 'B': 0xDEADBEEF, '~': 0x55FABB4F}
    
    def __init__(self, ps_ip: str = '192.168.0.1', packet_type: str = '~'):
        self.ps_ip = ps_ip
        self.packet_type = packet_type
        self.packet_count = 0
        self.running = False
        self.sock = None
        
    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', 33740))
        self.sock.settimeout(2.0)
        self.send_heartbeat()
        
    def send_heartbeat(self):
        if self.sock:
            self.sock.sendto(self.packet_type.encode(), (self.ps_ip, 33739))
    
    def decrypt_packet(self, data: bytes) -> Optional[bytes]:
        try:
            if len(data) < 0x128: return None
            oiv = data[0x40:0x44]
            iv1 = int.from_bytes(oiv, byteorder='little')
            iv2 = iv1 ^ self.XOR_MAGIC[self.packet_type]
            IV = bytearray()
            IV.extend(iv2.to_bytes(4, 'little'))
            IV.extend(iv1.to_bytes(4, 'little'))
            cipher = Salsa20.new(key=self.KEY[0:32], nonce=bytes(IV))
            ddata = cipher.decrypt(data)
            if int.from_bytes(ddata[0:4], byteorder='little') != 0x47375330: return None
            return ddata
        except: return None
    
    def parse_packet(self, data: bytes) -> Optional[TelemetryPacket]:
        try:
            return TelemetryPacket(
                position=struct.unpack_from('fff', data, 0x04),
                velocity=struct.unpack_from('fff', data, 0x10),
                rotation=struct.unpack_from('fff', data, 0x1C),
                speed=struct.unpack_from('f', data, 0x4C)[0],
                rpm=struct.unpack_from('f', data, 0x3C)[0],
                fuel_level=struct.unpack_from('f', data, 0x44)[0],
                fuel_capacity=struct.unpack_from('f', data, 0x48)[0],
                throttle=struct.unpack_from('B', data, 0x91)[0],
                brake=struct.unpack_from('B', data, 0x92)[0],
                current_gear=struct.unpack_from('B', data, 0x90)[0] & 0x0F,
                suggested_gear=(struct.unpack_from('B', data, 0x90)[0] >> 4) & 0x0F,
                tire_temps=struct.unpack_from('ffff', data, 0x60),
                tire_radius=struct.unpack_from('ffff', data, 0xB4),
                wheel_rps=struct.unpack_from('ffff', data, 0xA4),
                packet_id=struct.unpack_from('i', data, 0x70)[0],
                lap_count=struct.unpack_from('h', data, 0x74)[0],
                total_laps=struct.unpack_from('h', data, 0x76)[0],
                best_lap=struct.unpack_from('i', data, 0x78)[0],
                last_lap=struct.unpack_from('i', data, 0x7C)[0],
                race_rank=struct.unpack_from('B', data, 0x84)[0],
                total_cars=struct.unpack_from('B', data, 0x86)[0],
                flags=struct.unpack_from('h', data, 0x8E)[0],
                boost=struct.unpack_from('f', data, 0x50)[0],
                oil_pressure=struct.unpack_from('f', data, 0x54)[0],
                water_temp=struct.unpack_from('f', data, 0x58)[0],
                oil_temp=struct.unpack_from('f', data, 0x5C)[0],
                timestamp=time.time()
            )
        except: return None
    
    def start(self, callback):
        self.running = True
        self.connect()
        while self.running:
            if self.packet_count % 100 == 0: self.send_heartbeat()
            try:
                data, addr = self.sock.recvfrom(4096)
                decrypted = self.decrypt_packet(data)
                if decrypted:
                    packet = self.parse_packet(decrypted)
                    if packet:
                        callback(packet)
                        self.packet_count += 1
            except socket.timeout: self.send_heartbeat()
            except: pass

    def stop(self):
        self.running = False
        if self.sock: self.sock.close()

class RaceDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("GTMate 1.0.1")
        self.root.geometry("1200x800")
        self.root.configure(bg='#000000')

        self.current_version = "1.0.1"
        self.check_for_update_st()
        
        self.receiver = None
        self.current_packet = None
        self.last_packet_time = 0
        self.last_data_val = (0, 0)
        self.last_change_time = 0
        
        self.create_widgets()
        self.max_rpm_seen = 5000

        self.bot = None
        self.bot_running = False

        # 부스트 게이지를 담을 프레임 (나중에 이 프레임 통째로 숨기거나 보임)
        self.boost_frame = tk.Frame(root, bg='black')

        # 라벨 표시 (Boost: 0.00 bar)
        self.boost_label = tk.Label(self.boost_frame, text="BOOST: 0.00 bar", 
                                    font=("Arial", 12, "bold"), fg="white", bg="black")
        self.boost_label.pack()

        # 부스트 바 (가로 200px 정도의 작은 바)
        self.boost_canvas = tk.Canvas(self.boost_frame, width=200, height=20, 
                                      bg='#333333', highlightthickness=0)
        self.boost_canvas.pack()

        self.has_turbo_active = False # 현재 터보 게이지가 켜져 있는지 상태 저장

        # [연료 계산용 변수 초기화]
        self.last_lap_count = -1       # 마지막으로 체크한 랩 수
        self.fuel_at_lap_start = -1    # 랩 '시작' 시점의 연료량 (고정값)
        
        self.fuel_consumption_history = [] 
        self.avg_fuel_per_lap = 0       
        
        # [화면 표시용 노이즈 필터]
        self.display_fuel_pct = 100.0

        # 폰트 설정
        self.font_small = tkfont.Font(family="Arial", size=12, weight="bold")
        self.font_huge = tkfont.Font(family="Arial", size=25, weight="bold")

        if not hasattr(self, 'display_fuel_pct'):
            self.display_fuel_pct = pct

        # 순위 표시 라벨 (예: POS: 01 / 16)
        self.pos_label = tk.Label(
            root, 
            text="POS: -- / --", 
            font=self.font_huge,
            fg="white", 
            bg="black"
        )

        self.pos_label.place(relx=0.015, rely=0.5, anchor='w')

        # 1. 상태의 시작점: 처음엔 무조건 트랙 위라고 가정합니다.
        self.pit_status = "TRACK"
        
        # 2. 속도 낙차 계산용: 첫 프레임에서 speed_drop 계산 시 에러 방지
        self.last_speed_kmh = 0.0

    def check_for_update_st(self):
            try:
                url = "https://raw.githubusercontent.com/GreenRiceCake/GTMate/main/version.json"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    latest_version = data["version"]

                    if self.current_version < latest_version:
                        # 버전이 낮으면 업데이트 프로그램 실행
                        subprocess.Popen(["Updater.exe"])
                    else:
                        print("최신 버전입니다.")
                else:
                    print("업데이트 서버에 연결할 수 없습니다.")

            except Exception as e:
                print(f"업데이트 확인 중 오류 발생: {e}")
        
    def create_widgets(self):
        # 상단 설정
        top_frame = tk.Frame(self.root, bg='#1a1a1a', pady=5)
        top_frame.pack(fill=tk.X)
        
        # 1. PS IP 설정
        tk.Label(top_frame, text="PS IP:", bg='#1a1a1a', fg='gray').pack(side=tk.LEFT, padx=5)
        self.ip_entry = tk.Entry(top_frame, font=('Arial', 10), width=12)
        self.ip_entry.insert(0, "192.168.0.1")
        self.ip_entry.pack(side=tk.LEFT, padx=5)
        
        self.connect_btn = tk.Button(top_frame, text="Connect PS", command=self.toggle_connection, bg='#00ff00', font=('Arial', 10, 'bold'))
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(top_frame, text="● PS: Ready", bg='#1a1a1a', fg='gray', font=('Arial', 10))
        self.status_label.pack(side=tk.LEFT, padx=10)

        # ----------------------------------------------------
        # [봇 제어 버튼 추가]
        # ----------------------------------------------------
        tk.Frame(top_frame, width=2, bg='#333').pack(side=tk.LEFT, fill=tk.Y, padx=10) # 구분선

        self.btn_bot_config = tk.Button(top_frame, text="Bot Config", command=self.open_bot_config, bg='#444', fg='white', font=('Arial', 9))
        self.btn_bot_config.pack(side=tk.LEFT, padx=5)

        self.btn_bot_toggle = tk.Button(top_frame, text="Start Radio", command=self.toggle_bot, bg='#0080ff', fg='white', font=('Arial', 10, 'bold'))
        self.btn_bot_toggle.pack(side=tk.LEFT, padx=5)
        
        self.lbl_bot_status = tk.Label(top_frame, text="● Radio: OFF", bg='#1a1a1a', fg='gray', font=('Arial', 10))
        self.lbl_bot_status.pack(side=tk.LEFT, padx=5)

        main_frame = tk.Frame(self.root, bg='black')
        main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # [왼쪽 열: 랩타임 & 연료]
        self.left_col = tk.Frame(main_frame, bg='black', width=250)
        self.left_col.pack(side=tk.LEFT, fill=tk.Y)
        
        # 랩타임 정보 (왼쪽 중앙)
        self.lap_info_frame = tk.Frame(self.left_col, bg='#1a1a1a', relief=tk.RIDGE, bd=2, pady=15)
        self.lap_info_frame.pack(side=tk.TOP, fill=tk.X, pady=(50, 20))
        tk.Label(self.lap_info_frame, text="BEST LAP", bg='#1a1a1a', fg='yellow', font=('Arial', 12, 'bold')).pack()
        self.best_lap_label = tk.Label(self.lap_info_frame, text="--:--:---", bg='#1a1a1a', fg='white', font=('Arial', 24))
        self.best_lap_label.pack(pady=(0, 10))
        tk.Label(self.lap_info_frame, text="LAST LAP", bg='#1a1a1a', fg='white', font=('Arial', 12, 'bold')).pack()
        self.last_lap_label = tk.Label(self.lap_info_frame, text="--:--:---", bg='#1a1a1a', fg='white', font=('Arial', 24))
        self.last_lap_label.pack()

        self.fuel_frame = tk.Frame(self.left_col, bg='#1a1a1a', relief=tk.RIDGE, bd=2)
        self.fuel_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)
        tk.Label(self.fuel_frame, text="FUEL", bg='#1a1a1a', fg='cyan', font=('Arial', 14, 'bold')).pack()
        self.fuel_label = tk.Label(self.fuel_frame, text="-- %", bg='#1a1a1a', fg='#00ff00', font=('Arial', 30))
        self.fuel_label.pack()
        self.fuel_bar = ttk.Progressbar(self.fuel_frame, length=180, mode='determinate')
        self.fuel_bar.pack(pady=5)

        # [중앙 열: RPM, 기어, 속도, 타이어]
        self.center_col = tk.Frame(main_frame, bg='black')
        self.center_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20)
        self.rpm_canvas = tk.Canvas(self.center_col, height=50, bg='#222', highlightthickness=0)
        self.rpm_canvas.pack(fill=tk.X, pady=(10, 0))
        self.gear_label = tk.Label(self.center_col, text="N", fg='#00ff00', bg='black', font=('Arial', 180, 'bold'))
        self.gear_label.pack()
        self.speed_label = tk.Label(self.center_col, text="0", fg='white', bg='black', font=('Arial', 80))
        self.speed_label.pack()
        self.speed_kmh_label = tk.Label(self.center_col, text="km/h", fg='gray', bg='black', font=('arial', 15))
        self.speed_kmh_label.pack()
        self.tire_frame = tk.Frame(self.center_col, bg='black', pady=20)
        self.tire_frame.pack(side=tk.BOTTOM)
        self.tire_labels = []
        for i, name in enumerate(["FL", "FR", "RL", "RR"]):
            lbl = tk.Label(self.tire_frame, text=f"{name}\n--", font=('Arial', 16, 'bold'), width=8, height=3, relief=tk.RAISED, bd=2, bg='#1a1a1a', fg='white')
            lbl.grid(row=i//2, column=i%2, padx=10, pady=5)
            self.tire_labels.append(lbl)

        # [오른쪽 열: 입력바, 상세 상태]
        self.right_col = tk.Frame(main_frame, bg='black', width=300)
        self.right_col.pack(side=tk.LEFT, fill=tk.Y)
        
        # 입력바
        input_sub = tk.Frame(self.right_col, bg='black')
        input_sub.pack(pady=10)
        tk.Label(input_sub, text="THR  BRK", bg='black', fg='white', font=('Arial', 12, 'bold')).pack()
        self.thr_canvas = tk.Canvas(input_sub, width=40, height=250, bg='#222', highlightthickness=0)
        self.thr_canvas.pack(side=tk.LEFT, padx=10)
        self.brk_canvas = tk.Canvas(input_sub, width=40, height=250, bg='#222', highlightthickness=0)
        self.brk_canvas.pack(side=tk.LEFT, padx=10)

        # 상태 섹션 (확장)
        self.status_box = tk.Frame(self.right_col, bg='#1a1a1a', pady=10, relief=tk.RIDGE, bd=2)
        self.status_box.pack(fill=tk.X, pady=10)
        self.replay_label = tk.Label(self.status_box, text="STATUS: STANDBY", bg='#1a1a1a', fg='#666666', font=('Arial', 12, 'bold'))
        self.replay_label.pack()
        self.lap_count_label = tk.Label(self.status_box, text="LAP: -- / --", bg='#1a1a1a', fg='white', font=('Arial', 12))
        self.lap_count_label.pack(pady=(0, 10))

        # 플래그 표시부
        self.flag_asm = tk.Label(self.status_box, text="ASM", bg='#1a1a1a', fg='#333', font=('Arial', 11, 'bold'))
        self.flag_asm.pack()
        self.flag_tcs = tk.Label(self.status_box, text="TCS", bg='#1a1a1a', fg='#333', font=('Arial', 11, 'bold'))
        self.flag_tcs.pack()
        self.flag_beam = tk.Label(self.status_box, text="HIGH BEAM", bg='#1a1a1a', fg='#333', font=('Arial', 11, 'bold'))
        self.flag_beam.pack()
        self.flag_hand = tk.Label(self.status_box, text="HANDBRAKE", bg='#1a1a1a', fg='#333', font=('Arial', 11, 'bold'))
        self.flag_hand.pack()

    def toggle_connection(self):
        if self.receiver and self.receiver.running: self.stop_connection()
        else: self.start_connection()

    def start_connection(self):
        ps_ip = self.ip_entry.get().strip()
        if not ps_ip: return
        self.status_label.config(text="● 연결 시도 중", fg='orange')
        self.receiver = GT7TelemetryReceiver(ps_ip=ps_ip)
        threading.Thread(target=self.receiver.start, args=(self.update_data,), daemon=True).start()
        self.connect_btn.config(text="연결 끊기", bg='#ff0000')
        self.ip_entry.config(state='disabled')

    def stop_connection(self):
        if self.receiver: self.receiver.stop()
        self.status_label.config(text="● 연결 끊김", fg='red')
        self.connect_btn.config(text="연결", bg='#00ff00')
        self.ip_entry.config(state='normal')

    def update_data(self, packet: TelemetryPacket):
        self.current_packet = packet
        self.last_packet_time = time.time()
        SHARED_GAME_STATE['fuel_liters'] = packet.fuel_level
        SHARED_GAME_STATE['fuel_percent'] = (packet.fuel_level / packet.fuel_capacity * 100) if packet.fuel_capacity else 0
        SHARED_GAME_STATE['current_lap'] = packet.lap_count
        SHARED_GAME_STATE['total_laps'] = packet.total_laps
        SHARED_GAME_STATE['best_lap_ms'] = packet.best_lap
        SHARED_GAME_STATE['last_lap_ms'] = packet.last_lap
        SHARED_GAME_STATE['rank'] = packet.race_rank
        SHARED_GAME_STATE['total_cars'] = packet.total_cars
        self.root.after(0, self.update_gui)

    def update_gui(self):
        now = time.time()
        if not self.current_packet or (now - self.last_packet_time > 2.0):
            self.status_label.config(text="● 연결 실패/대기", fg='red')
            self.show_empty_data()
            return
        
        self.status_label.config(text="● 연결됨", fg='#00ff00')
        p = self.current_packet
        on_track = GT7Flags.check(p.flags, GT7Flags.CAR_ON_TRACK)
        
        if abs(p.speed - self.last_data_val[0]) > 0.1 or abs(p.rpm - self.last_data_val[1]) > 1:
            self.last_change_time = now
            self.last_data_val = (p.speed, p.rpm)

        is_moving = (now - self.last_change_time < 5.0)
        
        if on_track:
            self.replay_label.config(text="STATUS: ON TRACK", fg='#00ff00')
            self.render_dashboard(p)
        elif is_moving:
            self.replay_label.config(text="STATUS: REPLAY MODE", fg='yellow')
            self.render_dashboard(p)
        else:
            self.replay_label.config(text="STATUS: IDLE", fg='#666666')
            self.show_empty_data()

        has_turbo_flag = bool(p.flags & GT7Flags.HAS_TURBO)
        is_actually_boosting = (p.boost > 1.05) # 대기압보다 높은 압력이 감지되면 터보로 간주
        
        should_show_boost = has_turbo_flag or is_actually_boosting

        if should_show_boost:
            if not self.has_turbo_active:
                self.boost_frame.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)
                self.has_turbo_active = True
            self.draw_boost_bar(p.boost)
        else:
            # 자연흡기 차로 판단되면 즉시 숨김
            if self.has_turbo_active:
                self.boost_frame.place_forget()
                self.has_turbo_active = False

    def render_dashboard(self, p):
        self.speed_label.config(text=f"{int(p.speed * 3.6)}")
        self.gear_label.config(text="R" if p.current_gear == 0 else ("N" if p.current_gear == 15 else str(p.current_gear)))
        self.draw_rpm_bar(p.rpm, GT7Flags.check(p.flags, GT7Flags.REV_LIMITER))
        
        # 랩타임 업데이트
        if p.best_lap > 0: self.best_lap_label.config(text=self.format_time(p.best_lap))
        if p.last_lap > 0: self.last_lap_label.config(text=self.format_time(p.last_lap))
        
        # 플래그 상태 업데이트
        self.flag_asm.config(fg='#00ff00' if GT7Flags.check(p.flags, GT7Flags.ASM_ACTIVE) else '#333')
        self.flag_tcs.config(fg='#00ff00' if GT7Flags.check(p.flags, GT7Flags.TCS_ACTIVE) else '#333')
        self.flag_beam.config(fg='#00ffff' if GT7Flags.check(p.flags, GT7Flags.HIGH_BEAM) else '#333')
        self.flag_hand.config(fg='#ff0000' if GT7Flags.check(p.flags, GT7Flags.HANDBRAKE) else '#333')

        names = ["FL", "FR", "RL", "RR"]
        for i, temp in enumerate(p.tire_temps): self.update_tire(self.tire_labels[i], temp, names[i])

        # 1. 속도 단위 변환 (m/s -> km/h) 및 변수 초기화
        kmh = p.speed * 3.6
        if not hasattr(self, 'pit_status'): self.pit_status = "TRACK"
        if not hasattr(self, 'last_speed_kmh'): self.last_speed_kmh = kmh
        
        new_status = self.pit_status
        
        # -------------------------------------------------------------------------
        # [0. 피트 진입 감지]
        # -------------------------------------------------------------------------
        if self.pit_status == "TRACK":
            # 이전 프레임과 현재 프레임의 속도 차이 (낙차)
            speed_drop = self.last_speed_kmh - kmh
            
            # [핵심 조건]
            # 1. 속도가 한 번에 50km/h 이상 증발 (물리적으로 불가능한 감속 = 워프)
            # 2. 혹은 속도가 1km/h 미만으로 거의 멈춤 (0km/h 지점 통과)
            if (speed_drop > 50 or kmh < 1.0) and p.lap_count > 0:
                # 단, 아주 저속(10km/h 미만)에서 멈춘 건 사고일 수 있으니 
                # 직전 속도가 어느 정도(40km/h 이상) 있었을 때만 피트 진입으로 인정
                if self.last_speed_kmh > 40:
                    print(f">>> [PIT DETECTED] Drop: {speed_drop:.1f}, Current: {kmh:.1f}")
                    new_status = "PIT_ENTRY"
                    if self.bot and self.bot.loop:
                        asyncio.run_coroutine_threadsafe(self.bot.speak_tts("Box, Box, Box."), self.bot.loop)
                        asyncio.run_coroutine_threadsafe(self.bot.restart_listener(), self.bot.loop)

        # -------------------------------------------------------------------------
        # [1. 피트 작업 및 탈출]
        # -------------------------------------------------------------------------
        elif self.pit_status == "PIT_ENTRY":
            # 속도가 0이 되거나 타이어 온도가 60도 이하로 리셋되면 작업 중으로 간주
            if kmh < 1.0  or all(t <= 60.0 for t in p.tire_temps):
                new_status = "PIT_STOP"
            
            # 만약 다시 속도가 나면 (드라이브 스루나 감지 오류 대응)
            if kmh > 90: new_status = "TRACK"

        elif self.pit_status == "PIT_STOP":
            # 80km/h(리미터 속도)를 넘어서 가속하기 시작하면 트랙 복귀
            if kmh > 85:
                new_status = "TRACK"
                self.fuel_at_lap_start = p.fuel_level
                if self.bot and self.bot.loop:
                    asyncio.run_coroutine_threadsafe(self.bot.speak_tts("Push now! Clear track."), self.bot.loop)
                    asyncio.run_coroutine_threadsafe(self.bot.restart_listener(), self.bot.loop)

                if self.bot and self.bot.loop and self.bot.loop.is_running():
                    # 단순히 변수만 바꾸는 게 아니라 루프 자체를 재시작 명령
                    asyncio.run_coroutine_threadsafe(
                        self.bot.restart_listener(), 
                        self.bot.loop
                    )
                print(">>> [System] 피트 아웃: 음성 인식 루프 재시작 명령 전송")

        if new_status != self.pit_status:
            print(f">>> [PIT STATUS] {self.pit_status} -> {new_status}")
            self.pit_status = new_status

        self.last_speed_kmh = kmh


        # -------------------------------------------------------------------------
        # [연료 및 전략 시스템 로직 - 기존 코드]
        # -------------------------------------------------------------------------
        # 피트 작업 중이 아닐 때만(TRACK) 랩당 소모량을 계산하여 데이터 오염 방지
        if p.fuel_capacity > 0:
            
            # 1. 초기화 로직 (게임 시작 직후 한 번만 실행)
            if self.last_lap_count == -1:
                self.last_lap_count = p.lap_count
                self.fuel_at_lap_start = p.fuel_level
                if not hasattr(self, 'display_fuel_pct'):
                    self.display_fuel_pct = (p.fuel_level / p.fuel_capacity) * 100

            # 2. 랩 변경 감지 및 평균 소모량 계산
            if p.lap_count > self.last_lap_count:
                if self.last_lap_count == 0 and p.lap_count == 1 and self.pit_status == "TRACK":
                    print("스타트 랩 감지: 첫 데이터 제외")
                else:
                    if self.fuel_at_lap_start > 0:
                        fuel_used = self.fuel_at_lap_start - p.fuel_level
                        
                        # 피트인 상태가 아닐 때만 히스토리에 추가 (2중 방어)
                        if 0.5 < fuel_used < p.fuel_capacity * 0.3:
                            self.fuel_consumption_history.append(fuel_used)
                            if len(self.fuel_consumption_history) > 5:
                                self.fuel_consumption_history.pop(0)
                            self.avg_fuel_per_lap = sum(self.fuel_consumption_history) / len(self.fuel_consumption_history)

                self.fuel_at_lap_start = p.fuel_level
                self.last_lap_count = p.lap_count
            
            # 주행 중 급유 감지 (피트 상태 업데이트를 놓쳤을 경우를 대비한 백업)
            elif p.fuel_level > self.fuel_at_lap_start + 1.0:
                 self.fuel_at_lap_start = p.fuel_level


            # 3. 화면 표시용 데이터 가공 (노이즈 필터링)
            # -------------------------------------------------
            raw_pct = (p.fuel_level / p.fuel_capacity) * 100
            
            if (raw_pct < self.display_fuel_pct) or (self.pit_status != "TRACK") or (raw_pct - self.display_fuel_pct > 5.0):
                self.display_fuel_pct = raw_pct
                
            display_int_pct = int(self.display_fuel_pct)


            # 4. 남은 랩 수 텍스트 생성
            # -------------------------------------------------
            sub_text = "CALC..."
            
            # 데이터가 1개 이상 있고 평균값이 정상적일 때
            if len(self.fuel_consumption_history) >= 1 and self.avg_fuel_per_lap > 0.5:
                laps_remain = p.fuel_level / self.avg_fuel_per_lap

                # [봇 공유 데이터 업데이트]
                SHARED_GAME_STATE['laps_remain'] = laps_remain
                
                # 현실적인 랩 수 표시 제한
                if laps_remain > 50:
                    laps_str = "50+"
                else:
                    laps_str = f"{laps_remain:.1f}"
                
                sub_text = f"({laps_str} LAPS)"
            else:
                # 데이터 수집 전에는 현재 리터 표시
                sub_text = f"({int(p.fuel_level)}L)"


            # 5. UI 최종 업데이트
            # -------------------------------------------------
            fuel_color = '#00ff00' 
            if display_int_pct < 20: fuel_color = '#ffff00'
            if display_int_pct < 10: fuel_color = '#ff0000'
            
            self.fuel_label.config(
                text=f"FUEL: {display_int_pct}%\n{sub_text}", 
                fg=fuel_color,
                font=self.font_small
            )
            self.fuel_bar['value'] = self.display_fuel_pct

        # ---------------------------------------------------------
        # [순위 및 참가자 수 업데이트]
        # ---------------------------------------------------------
        # 패킷 구조에 따라 변수명이 다를 수 있으므로 getattr로 안전하게 가져옵니다.
        current_pos = getattr(p, 'race_rank', 0)
        total_cars = getattr(p, 'total_cars', 0)

        # 0인 경우는 데이터가 아직 안 들어온 것이므로 대기 표시
        if current_pos < 255:
            # 포지션이 1~9위일 때 앞에 0을 붙여 "01"처럼 보이게 하면 가독성이 좋아집니다.
            pos_str = f"{current_pos:02d}" if current_pos < 10 else f"{current_pos}"
            toca_str = f"{total_cars:02d}" if total_cars < 10 else f"{total_cars}"
            self.pos_label.config(text=f"POS: {pos_str} / {toca_str}")
            
            # [추가 효과] 1위일 때는 금색(또는 밝은 노란색)으로 강조
            if current_pos == 1:
                self.pos_label.config(fg="#FFD700") # Gold
            else:
                self.pos_label.config(fg="white")
        else:
            self.pos_label.config(text="POS: -- / --", fg="white")
            
        self.draw_vertical_bar(self.thr_canvas, p.throttle / 255, '#00ff00')
        self.draw_vertical_bar(self.brk_canvas, p.brake / 255, '#ff0000')
        self.lap_count_label.config(text=f"LAP: {p.lap_count} / {max(0, p.total_laps)}")

    def draw_rpm_bar(self, rpm, is_limiter):
        self.rpm_canvas.delete("all")
        w = self.rpm_canvas.winfo_width()
        
        # 1. 동적 최대치 업데이트: 현재 RPM이 기록된 최대치보다 크면 갱신
        if rpm > self.max_rpm_seen:
            self.max_rpm_seen = rpm
        
        # 2. 비율 계산 (갱신된 max_rpm_seen 기준)
        # 분모가 0이 되는 것을 방지하기 위해 max(1, ...) 사용
        pct = min(rpm / max(1, self.max_rpm_seen), 1.0)
        
        fill_w = w * pct
        
        # 색상 로직 (비율 기반이므로 그대로 유지해도 무방합니다)
        color = '#00ff00'
        if pct > 0.9: color = '#87CEEB' # 시프트 라이트 느낌 (하늘색)
        elif pct > 0.8: color = '#ff0000' # 레드존 근처 (빨간색)
        elif pct > 0.6: color = '#ffff00' # 중간 (노란색)
        
        # 리미터 작동 시 깜빡임 로직
        if pct > 0.95 and (int(time.time() * 15) % 2 == 0):
            return
            
        self.rpm_canvas.create_rectangle(0, 0, fill_w, 50, fill=color, outline="")

    def draw_boost_bar(self, raw_boost):
        # 1. 부스트압 보정: GT7은 절대압력을 주므로 대기압(1.0)을 빼야 실제 게이지 압력이 됨
        # raw_boost가 1.0이면 0 bar, 0.2이면 -0.8 bar가 됩니다.
        boost = raw_boost - 1.0
        
        self.boost_canvas.delete("all")
        w = 150
        
        # 2. 범위 설정: -1.0(진공) ~ 2.0(과급) bar 기준
        # min_val을 -1.0으로 잡아야 -0.8 bar 같은 진공 상태가 중간 아래로 표현됩니다.
        min_val = -1.0
        max_val = 2.0
        
        # 비율 계산: (현재값 - 최소값) / (최대값 - 최소값)
        pct = (boost - min_val) / (max_val - min_val)
        pct = max(0, min(pct, 1.0)) # 0.0 ~ 1.0 사이로 제한
        
        fill_w = w * pct
        
        # 3. 색상 설정: 0 bar(대기압) 이상이면 하늘색, 이하면 어두운 회색
        color = '#00FFFF' if boost >= 0 else '#555555'
        if boost > 1.2: color = '#FF4500' # 고부스트 시 주황색
        
        self.boost_canvas.create_rectangle(0, 0, fill_w, 12, fill=color, outline="")
        self.boost_label.config(text=f"BOOST: {boost:.2f} bar")

    def draw_vertical_bar(self, canvas, pct, color):
        canvas.delete("all")
        h = canvas.winfo_height()
        fill_h = h * pct
        canvas.create_rectangle(0, h - fill_h, 40, h, fill=color, outline="")

    def update_tire(self, label, temp, position):
        color = '#00ffff' if temp < 70 else '#00ff00' if temp < 85 else '#ffff00' if temp < 105 else '#ff0000'
        label.config(text=f"{position}\n{int(temp)}°C", fg=color)

    def show_empty_data(self):
        self.speed_label.config(text="--")
        self.gear_label.config(text="--")
        self.fuel_label.config(text="-- L")
        self.best_lap_label.config(text="--:--:---")
        self.last_lap_label.config(text="--:--:---")
        self.rpm_canvas.delete("all")
        self.thr_canvas.delete("all")
        self.brk_canvas.delete("all")
        self.max_rpm_seen = 5000
        for f in [self.flag_asm, self.flag_tcs, self.flag_beam, self.flag_hand]: f.config(fg='#333')

    def format_time(self, ms):
        minutes, seconds = divmod(ms // 1000, 60)
        return f"{minutes:02d}:{seconds:02d}.{ms % 1000:03d}"

    # ---------------------------------------------------
    # [봇 관련 메서드]
    # ---------------------------------------------------
    def open_bot_config(self):
        # 간단한 팝업창으로 토큰/채널ID 입력
        win = tk.Toplevel(self.root)
        win.title("Bot Configuration")
        win.geometry("400x200")
        
        config = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f: config = json.load(f)
            
        tk.Label(win, text="Discord Bot Token:").pack(pady=5)
        e_token = tk.Entry(win, width=40)
        e_token.insert(0, config.get("TOKEN", ""))
        e_token.pack()
        
        tk.Label(win, text="Voice Channel ID:").pack(pady=5)
        e_channel = tk.Entry(win, width=40)
        e_channel.insert(0, config.get("CHANNEL_ID", ""))
        e_channel.pack()
        
        def save():
            new_cfg = {"TOKEN": e_token.get().strip(), "CHANNEL_ID": e_channel.get().strip()}
            with open(CONFIG_PATH, 'w') as f: json.dump(new_cfg, f, indent=4)
            win.destroy()
            
        tk.Button(win, text="Save & Close", command=save, bg='#00ff00').pack(pady=20)

    def toggle_bot(self):
        # 1. 실행 중이라면 끄기 (빨간 버튼 상태일 때)
        if self.bot_running:
            self.stop_bot()
            return

        # 2. 실행 중이 아니라면 켜기
        config = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f: 
                config = json.load(f)
        self.config = config
        
        token = self.config.get("TOKEN", "")
        if not token:
            self.open_bot_config()
            return

        # --- 버튼을 빨간색 "Disable Radio"로 변경 ---
        self.lbl_bot_status.config(text="● Radio: ON", fg='#00ff00')
        self.btn_bot_toggle.config(
            text="Disable Radio", 
            bg='#ff4444', # 빨간색
            fg='white',
            activebackground='#cc0000'
        )
        
        self.bot_running = True
        self.bot_thread = threading.Thread(target=self.run_bot_process, args=(token,), daemon=True)
        self.bot_thread.start()

    def stop_bot(self):
        """봇을 종료하고 버튼을 다시 파란색으로"""
        if self.bot:
            # 봇에게 종료 신호 보냄
            asyncio.run_coroutine_threadsafe(self.bot.close(), self.bot.loop)
            
        self.bot_running = False
        self.lbl_bot_status.config(text="● Radio: OFF", fg='#ff0000')
        self.btn_bot_toggle.config(
            text="Start Radio", 
            bg='#007bff', # 파란색 (원래 색상)
            fg='white',
            activebackground='#0056b3'
        )

    def run_bot_process(self, token):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # self.config를 빼고 EngineerBot()만 호출하세요.
        self.bot = EngineerBot() 
        
        try:
            loop.run_until_complete(self.bot.start(token))
        except Exception as e:
            print(f"Bot Error: {e}")
        finally:
            loop.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = RaceDashboard(root)
    root.mainloop()
