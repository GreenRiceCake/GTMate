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
import importlib.metadata
import ipaddress
import numpy as np
try:
    import sounddevice as sd
except ImportError:
    sd = None
import discord
from discord import opus
from discord.ext import commands, voice_recv
import vosk
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional
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
    "laps_remain_ready": False,
    "current_lap": 0,
    "current_lap_ms": -1,
    "total_laps": 0,
    "best_lap_ms": -1,
    "last_lap_ms": -1,
    "rank": 0,
    "total_cars": 0,
    "on_track": False,
    "race_active": False,
}

RADIO_UI_STATE = {
    "status": "OFF",
    "detail": "Radio off",
    "heard": "",
    "updated_at": 0.0,
}

def set_radio_ui_state(status=None, detail=None, heard=None):
    if status is not None:
        RADIO_UI_STATE["status"] = status
    if detail is not None:
        RADIO_UI_STATE["detail"] = detail
    if heard is not None:
        RADIO_UI_STATE["heard"] = heard
    RADIO_UI_STATE["updated_at"] = time.time()

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
PENDING_UPDATER_EXE = os.path.join(BASE_DIR, "Updater.new.exe")
opus_path = os.path.join(BASE_DIR, "bin", "libopus.dll")


def load_bot_config_file():
    if not os.path.exists(CONFIG_PATH):
        return normalize_settings({})

    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if not isinstance(config, dict):
            raise ValueError("configuration root must be an object")
        return normalize_settings(config)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f">>> [Config] Failed to load bot configuration: {type(e).__name__}")
        return normalize_settings({})


def normalize_audio_device(value):
    if not isinstance(value, dict):
        return None
    name = str(value.get("name") or "").strip()
    hostapi = str(value.get("hostapi") or "").strip()
    if not name:
        return None
    return {"name": name, "hostapi": hostapi}


def normalize_settings(config):
    config = dict(config) if isinstance(config, dict) else {}
    playstation = config.get("playstation")
    playstation = dict(playstation) if isinstance(playstation, dict) else {}
    radio = config.get("radio")
    radio = dict(radio) if isinstance(radio, dict) else {}
    discord_settings = radio.get("discord")
    discord_settings = dict(discord_settings) if isinstance(discord_settings, dict) else {}
    native_settings = radio.get("native")
    native_settings = dict(native_settings) if isinstance(native_settings, dict) else {}

    mode = str(radio.get("mode") or config.get("RADIO_MODE") or "discord").lower()
    if mode not in {"discord", "native"}:
        mode = "discord"

    normalized = {
        key: value
        for key, value in config.items()
        if key not in {"TOKEN", "CHANNEL_ID", "PS_IP", "RADIO_MODE"}
    }
    normalized.update(
        {
            "schema_version": 2,
            "playstation": {
                **playstation,
                "ip": str(playstation.get("ip") or config.get("PS_IP") or "192.168.0.1"),
            },
            "radio": {
                **radio,
                "mode": mode,
                "discord": {
                    **discord_settings,
                    "token": str(discord_settings.get("token") or config.get("TOKEN") or ""),
                    "channel_id": str(
                        discord_settings.get("channel_id") or config.get("CHANNEL_ID") or ""
                    ),
                },
                "native": {
                    **native_settings,
                    "input_device": normalize_audio_device(native_settings.get("input_device")),
                    "output_device": normalize_audio_device(native_settings.get("output_device")),
                },
            },
        }
    )
    return normalized


def save_bot_config_file(config):
    config = normalize_settings(config)
    temporary_path = CONFIG_PATH + ".tmp"
    try:
        with open(temporary_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, CONFIG_PATH)
    finally:
        try:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        except OSError:
            pass

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

@dataclass(frozen=True)
class RadioCommandDefinition:
    aliases: tuple[str, ...]
    response_builder: Optional[Callable] = None


def build_fuel_response(data, _format_time):
    liters = int(data['fuel_liters'])
    if not data.get('laps_remain_ready'):
        return f"Fuel is {liters} liters. I don't have enough data to estimate laps remaining yet."
    return f"Fuel is {liters} liters. That's about {data['laps_remain']:.1f} laps."


def build_rank_response(data, _format_time):
    if not data.get('race_active'):
        return ENGINEER_LINES["not_in_race"]
    return f"Position {data['rank']} out of {data['total_cars']}."


def build_current_lap_response(data, _format_time):
    if not data.get('race_active'):
        return ENGINEER_LINES["not_in_race"]
    return f"Lap {data['current_lap']}."


def build_best_lap_response(data, format_time):
    if not data.get('race_active'):
        return ENGINEER_LINES["not_in_race"]
    if data['best_lap_ms'] <= 0:
        return "No best lap set yet."
    return f"Best lap is {format_time(data['best_lap_ms'])}."


def build_last_lap_response(data, format_time):
    if not data.get('race_active'):
        return ENGINEER_LINES["not_in_race"]
    if data['last_lap_ms'] <= 0:
        return "No last lap data."
    return f"Last lap was {format_time(data['last_lap_ms'])}."


RADIO_COMMANDS = {
    "wake": RadioCommandDefinition((
        "engineer", "mate", "chief", "radio", "hello", "hey",
        "new", "near", "gin", "beer", "ate",
    )),
    "fuel": RadioCommandDefinition(
        ("fuel", "gas", "petrol", "consumption", "tank", "few", "fill", "few all"),
        build_fuel_response,
    ),
    "rank": RadioCommandDefinition(
        ("rank", "position", "place", "where am i"),
        build_rank_response,
    ),
    "current_lap": RadioCommandDefinition(
        ("current lap", "lap", "current"),
        build_current_lap_response,
    ),
    "best_lap": RadioCommandDefinition(
        ("best", "fastest", "record", "lap time"),
        build_best_lap_response,
    ),
    "last_lap": RadioCommandDefinition(
        ("last", "previous", "lap time"),
        build_last_lap_response,
    ),
    "no": RadioCommandDefinition(
        ("no", "nope", "negative", "cancel", "nothing", "done", "thanks", "thank you")
    ),
}

TELEMETRY_COMMAND_KEYS = tuple(
    key for key, definition in RADIO_COMMANDS.items() if definition.response_builder
)

ENGINEER_LINES = {
    "wake": "Yes mate, Go ahead.",
    "cancel": "Copy that. Standing by.",
    "follow_up": "Anything else?",
    "missing_data": "I don't have that data.",
    "command_timeout": "Standing by.",
    "follow_up_timeout": "Radio out.",
    "discord_connected": "Radio check. Connected.",
    "native_connected": "Radio check. Native audio connected.",
    "not_in_race": "You're not currently in a race.",
    "new_best_lap": (
        "New best lap. {lap_time}. "
        "That's {delta_time} faster than your previous best."
    ),
    "low_fuel": "Fuel warning. About {laps} laps remaining.",
    "pit_box": "Box, Box, Box.",
    "pit_exit": "Push now! Clear track.",
    "tire_cold": "{subject} {verb} still cold.",
    "tire_hot": "{subject} {verb} still running hot.",
}


TIRE_POSITION_NAMES = (
    "Front left tire",
    "Front right tire",
    "Rear left tire",
    "Rear right tire",
)


def describe_tire_positions(indices):
    remaining = set(indices)
    if not remaining:
        return "", False
    if remaining == {0, 1, 2, 3}:
        return "All tires", True

    parts = []
    for label, pair in (
        ("Front tires", {0, 1}),
        ("Rear tires", {2, 3}),
        ("Left tires", {0, 2}),
        ("Right tires", {1, 3}),
    ):
        if pair <= remaining:
            parts.append(label)
            remaining -= pair
            break

    for index in range(4):
        if index in remaining:
            parts.append(TIRE_POSITION_NAMES[index])

    return " and ".join(parts), len(indices) > 1


def build_tire_temperature_warning(cold_indices, hot_indices):
    sentences = []
    cold_subject, cold_plural = describe_tire_positions(cold_indices)
    hot_subject, hot_plural = describe_tire_positions(hot_indices)
    if cold_subject:
        sentences.append(
            ENGINEER_LINES["tire_cold"].format(
                subject=cold_subject,
                verb="are" if cold_plural else "is",
            )
        )
    if hot_subject:
        sentences.append(
            ENGINEER_LINES["tire_hot"].format(
                subject=hot_subject,
                verb="are" if hot_plural else "is",
            )
        )
    return " ".join(sentences)


def list_audio_devices(kind):
    if sd is None:
        return []
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    hostapis = sd.query_hostapis()
    devices = []
    for index, device in enumerate(sd.query_devices()):
        if int(device.get(channel_key, 0)) <= 0:
            continue
        hostapi_index = int(device.get("hostapi", 0))
        hostapi_name = str(hostapis[hostapi_index].get("name") or "")
        descriptor = {"name": str(device.get("name") or ""), "hostapi": hostapi_name}
        label = f"{descriptor['name']} ({hostapi_name})"
        devices.append((label, descriptor, index))
    return devices


def get_default_audio_device_label(kind):
    if sd is None:
        return "System default (unavailable)"

    try:
        default_devices = sd.default.device
        default_index = default_devices[0 if kind == "input" else 1]
        if default_index is None or int(default_index) < 0:
            return "System default (not configured)"

        default_index = int(default_index)
        device = sd.query_devices(default_index)
        hostapis = sd.query_hostapis()
        hostapi_index = int(device.get("hostapi", 0))
        hostapi_name = str(hostapis[hostapi_index].get("name") or "")
        device_name = str(device.get("name") or "Unknown device")
        return f"System default - {device_name} ({hostapi_name})"
    except Exception:
        return "System default (unknown device)"


def resolve_audio_device(selection, kind):
    selection = normalize_audio_device(selection)
    if selection is None:
        return None
    devices = list_audio_devices(kind)
    for _label, descriptor, index in devices:
        if descriptor == selection:
            return index
    for _label, descriptor, index in devices:
        if descriptor["name"] == selection["name"]:
            return index
    return None


def resample_mono_pcm(pcm_bytes, source_rate, target_rate=16000):
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if samples.size == 0:
        return b""
    source_rate = int(round(source_rate))
    if source_rate == target_rate:
        return samples.tobytes()
    output_size = max(1, int(round(samples.size * target_rate / source_rate)))
    source_positions = np.arange(samples.size, dtype=np.float64)
    target_positions = np.arange(output_size, dtype=np.float64) * source_rate / target_rate
    target_positions = np.minimum(target_positions, samples.size - 1)
    output = np.interp(target_positions, source_positions, samples.astype(np.float64))
    return np.clip(output, -32768, 32767).astype(np.int16).tobytes()

@dataclass
class TelemetryPacket:
    position: tuple; velocity: tuple; rotation: tuple
    speed: float; rpm: float; max_rpm: float; fuel_level: float; fuel_capacity: float
    clutch: float; throttle: int; brake: int; current_gear: int; suggested_gear: int
    tire_temps: tuple; tire_radius: tuple; wheel_rps: tuple
    packet_id: int; lap_count: int; total_laps: int; best_lap: int; last_lap: int
    current_lap_ms: int; surface_type: str; wheel_steering_angle: tuple; wheel_base: float; car_category: str
    vehicle_dynamics_raw: float
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

def _parse_version(version_text):
    parts = []
    for part in version_text.replace("-", ".").split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])

def is_newer_version(latest_version, current_version):
    return _parse_version(latest_version) > _parse_version(current_version)

def verify_discord_voice_stack():
    try:
        discord_version = importlib.metadata.version("discord.py")
    except importlib.metadata.PackageNotFoundError:
        discord_version = getattr(discord, "__version__", "unknown")

    if _parse_version(discord_version) < (2, 7, 0):
        raise RuntimeError(
            "discord.py 2.7.0 이상이 필요합니다. "
            f"현재 버전: {discord_version}. "
            "DAVE 음성 연결을 위해 `py -3.13 -m pip install -U \"discord.py[voice]\" davey`로 갱신하세요."
        )

    try:
        davey_version = importlib.metadata.version("davey")
        import davey  # noqa: F401 - discord.py가 DAVE 연결에서 내부적으로 사용합니다.
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "davey가 설치되어 있지 않습니다. "
            "Discord의 DAVE/E2EE 음성 연결을 위해 `py -3.13 -m pip install -U davey`를 실행하세요."
        ) from exc

    try:
        voice_recv_version = importlib.metadata.version("discord-ext-voice-recv")
    except importlib.metadata.PackageNotFoundError:
        voice_recv_version = "unknown"

    print(
        ">>> [Voice] stack ready: "
        f"discord.py={discord_version}, davey={davey_version}, "
        f"discord-ext-voice-recv={voice_recv_version}"
    )

class GTMateAudioSink(voice_recv.AudioSink):
    def __init__(self, callback: Callable):
        super().__init__()
        self.callback = callback
        self.packet_count = 0
        self.last_log_time = 0
        self.decoders = {}
        self.decode_error_count = 0
        self.dave_decrypt_error_count = 0
        self.last_decode_error_time = 0
        self.logged_dave_state = False

    def wants_opus(self):
        return True

    def decrypt_dave_if_needed(self, user, ssrc, opus_data):
        voice_client = self.voice_client
        connection = getattr(voice_client, "_connection", None) if voice_client else None
        dave_session = getattr(connection, "dave_session", None)
        can_decrypt = bool(getattr(connection, "can_encrypt", False))

        if not dave_session or not can_decrypt:
            return opus_data

        if not self.logged_dave_state:
            self.logged_dave_state = True
            version = getattr(connection, "dave_protocol_version", "unknown")
            print(f">>> [Voice] DAVE decrypt enabled: protocol={version}")

        user_id = getattr(user, "id", None)
        if user_id is None and voice_client is not None:
            try:
                user_id = voice_client._get_id_from_ssrc(ssrc)
            except Exception:
                user_id = None

        if user_id is None:
            return opus_data

        try:
            import davey
            return dave_session.decrypt(user_id, davey.MediaType.audio, opus_data)
        except Exception as e:
            self.dave_decrypt_error_count += 1
            now = time.time()
            if now - self.last_decode_error_time > 2:
                self.last_decode_error_time = now
                print(
                    ">>> [Voice] DAVE 복호화 실패 - 패킷 건너뜀: "
                    f"{type(e).__name__}: {e} (errors={self.dave_decrypt_error_count})"
                )
            return None

    def write(self, user, data: voice_recv.VoiceData):
        opus_data = getattr(data, "opus", None)
        if not opus_data:
            return

        packet = getattr(data, "packet", None)
        ssrc = getattr(packet, "ssrc", id(user))
        opus_data = self.decrypt_dave_if_needed(user, ssrc, opus_data)
        if not opus_data:
            return

        decoder = self.decoders.get(ssrc)
        if decoder is None:
            decoder = opus.Decoder()
            self.decoders[ssrc] = decoder

        try:
            pcm = decoder.decode(opus_data, fec=False)
        except opus.OpusError as e:
            self.decode_error_count += 1
            now = time.time()
            if now - self.last_decode_error_time > 2:
                self.last_decode_error_time = now
                print(
                    ">>> [Voice] Opus 패킷 디코딩 실패 - 패킷만 건너뜀: "
                    f"{e} (errors={self.decode_error_count})"
                )
            return

        data.pcm = pcm

        self.packet_count += 1
        now = time.time()
        if self.packet_count == 1 or now - self.last_log_time > 5:
            self.last_log_time = now
            print(f">>> [Voice] PCM 수신 중: user={user}, bytes={len(pcm)}, packets={self.packet_count}")

        try:
            self.callback(user, data)
        except Exception as e:
            print(f">>> [Voice] sink callback error: {type(e).__name__}: {e}")

    def cleanup(self):
        print(">>> [Voice] audio sink cleanup")

class GTMateVoiceClient(voice_recv.VoiceRecvClient):
    def stop_playback_only(self):
        if hasattr(self, "stop_playing"):
            self.stop_playing()
        else:
            discord.VoiceClient.stop(self)

class EngineerBot(commands.Bot):
    def __init__(self, mode="discord", config=None):
        self.radio_mode = mode if mode in {"discord", "native"} else "discord"
        self.config = normalize_settings(config or load_bot_config_file())
        if self.radio_mode == "discord":
            verify_discord_voice_stack()

        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
        
        # Vosk 초기화
        if not os.path.exists(VOSK_MODEL_PATH):
            print(f"[오류] Vosk 모델 없음: {VOSK_MODEL_PATH}")
        self.vosk_model = vosk.Model(VOSK_MODEL_PATH)
        
        self.state = STATE_IDLE
        self.last_interaction_time = 0
        self.audio_queue = asyncio.Queue(maxsize=12)
        self.is_speaking = False
        self.processing_task = None
        self.audio_sink = None
        self.listener_restarting = False
        self.listener_error_count = 0
        self.last_listener_error_time = 0
        self.stt_buffer = bytearray()
        self.stt_chunk_size = 1600
        self.stt_idle_flush_seconds = 0.18
        self.stt_endpoint_silence_chunks = 6
        self.stt_voice_rms_threshold = 320.0
        self.stt_silence_rms_threshold = 260.0
        self.stt_silence_chunks_needed = 5
        self.stt_had_voice = False
        self.stt_quiet_chunks = 0
        self.stt_last_partial_text = ""
        self.last_handled_stt_text = ""
        self.last_handled_stt_time = 0.0
        self.last_partial_log_time = 0
        self.last_audio_level_log_time = 0
        self.native_input_stream = None
        self.native_input_rate = 16000
        self.native_input_device = None
        self.native_output_device = None
        self.native_stop_event = None
        self.last_native_status_log_time = 0
        self.recognizer = self.create_stt_recognizer()

    def create_stt_recognizer(self):
        grammar_items = []
        seen = set()

        def add_item(item):
            item = item.strip().lower()
            if item and item not in seen:
                seen.add(item)
                grammar_items.append(item)

        for definition in RADIO_COMMANDS.values():
            for alias in definition.aliases:
                add_item(alias)
                for word in alias.split():
                    add_item(word)

        for phrase in (
            "what is", "how much", "fuel left", "remaining fuel",
            "current lap", "last lap", "best lap", "fastest lap",
            "where am i", "what position", "yes please", "no thanks",
            "[unk]",
        ):
            add_item(phrase)

        recognizer = vosk.KaldiRecognizer(self.vosk_model, 16000, json.dumps(grammar_items))
        recognizer.SetWords(True)
        recognizer.SetPartialWords(True)
        print(f">>> [STT] grammar ready: {len(grammar_items)} entries, chunk={self.stt_chunk_size} bytes")
        return recognizer

    async def on_ready(self):
        print(f'>>> [Bot] Logged in as {self.user}')
        set_radio_ui_state("CONNECTING", f"Logged in as {self.user}")
        # 채널 자동 접속 로직 (설정 파일의 채널 ID 우선)
        cfg = self.load_config()
        channel_id = str(cfg["radio"]["discord"].get("channel_id") or "").strip()
        channel = None

        if channel_id:
            if channel_id.isdigit():
                configured_channel = self.get_channel(int(channel_id))
                if isinstance(configured_channel, discord.VoiceChannel):
                    channel = configured_channel
                else:
                    print(">>> [Bot] Configured channel is unavailable or is not a voice channel; using fallback.")
            else:
                print(">>> [Bot] Invalid voice channel ID; using fallback.")
        
        if not channel: # 설정 없으면 첫 번째 음성 채널 찾기
            for guild in self.guilds:
                if guild.voice_channels:
                    channel = guild.voice_channels[0]
                    break
        
        if channel:
            print(f">>> [Bot] Joining {channel.name}")
            set_radio_ui_state("CONNECTING", f"Joining {channel.name}")
            self.voice_client = await channel.connect(cls=GTMateVoiceClient, reconnect=True)
            self.start_voice_listener()
            await self.speak_tts(ENGINEER_LINES["discord_connected"])
            set_radio_ui_state("STANDBY", "Waiting for wake word")
            if not self.processing_task:
                self.processing_task = self.loop.create_task(self.process_audio_queue())
        else:
            print(">>> [Bot] No accessible voice channel found.")
            set_radio_ui_state("ERROR", "No accessible voice channel")

    async def start_native(self):
        if sd is None:
            raise RuntimeError("Native radio requires the sounddevice package.")

        native_config = self.config["radio"]["native"]
        self.native_input_device = resolve_audio_device(
            native_config.get("input_device"), "input"
        )
        self.native_output_device = resolve_audio_device(
            native_config.get("output_device"), "output"
        )
        input_info = sd.query_devices(self.native_input_device, "input")
        output_info = sd.query_devices(self.native_output_device, "output")
        default_rate = int(round(float(input_info.get("default_samplerate") or 48000)))
        self.native_input_rate = 16000
        try:
            sd.check_input_settings(
                device=self.native_input_device,
                channels=1,
                dtype="int16",
                samplerate=self.native_input_rate,
            )
        except Exception:
            self.native_input_rate = default_rate

        output_rate = int(
            round(float(output_info.get("default_samplerate") or 48000))
        )
        sd.check_output_settings(
            device=self.native_output_device,
            channels=1,
            dtype="int16",
            samplerate=output_rate,
        )

        blocksize = max(160, int(self.native_input_rate * 0.05))
        self.native_stop_event = asyncio.Event()
        self.native_input_stream = sd.RawInputStream(
            device=self.native_input_device,
            channels=1,
            samplerate=self.native_input_rate,
            dtype="int16",
            blocksize=blocksize,
            callback=self.native_input_callback,
        )
        self.native_input_stream.start()
        if not self.processing_task:
            self.processing_task = self.loop.create_task(self.process_audio_queue())

        input_name = str(input_info.get("name") or "System default")
        output_name = str(output_info.get("name") or "System default")
        print(
            f">>> [Native] audio ready: input={input_name}, output={output_name}, "
            f"input_rate={self.native_input_rate}, block={blocksize}"
        )
        set_radio_ui_state("CONNECTING", "Native audio ready")
        if not await self.speak_tts(ENGINEER_LINES["native_connected"]):
            raise RuntimeError("Native output test failed.")
        set_radio_ui_state("STANDBY", "Waiting for wake word")

    async def wait_for_native_stop(self):
        if self.native_stop_event is not None:
            await self.native_stop_event.wait()

    def native_input_callback(self, indata, _frames, _time_info, status):
        if status:
            now = time.time()
            if now - self.last_native_status_log_time > 2:
                self.last_native_status_log_time = now
                print(f">>> [Native] input status: {status}")
        if self.is_speaking or not getattr(self, "loop", None):
            return
        pcm = resample_mono_pcm(bytes(indata), self.native_input_rate, 16000)
        if not pcm:
            return
        try:
            self.loop.call_soon_threadsafe(self.enqueue_stt_audio, pcm)
        except RuntimeError:
            pass

    def start_voice_listener(self):
        if not self.voice_client or not self.voice_client.is_connected():
            print(">>> [Voice] listener 시작 실패: voice client가 연결되어 있지 않습니다.")
            set_radio_ui_state("ERROR", "Voice client disconnected")
            return

        if hasattr(self.voice_client, "is_listening") and self.voice_client.is_listening():
            return

        self.audio_sink = GTMateAudioSink(self.sink_callback)
        self.voice_client.listen(self.audio_sink, after=self.on_listener_finished)
        if self.state == STATE_IDLE:
            set_radio_ui_state("STANDBY", "Waiting for wake word")
        else:
            set_radio_ui_state("LISTENING", "Listening for command")
        print(">>> [Voice] listener started with GTMateAudioSink")

    def on_listener_finished(self, error):
        if error:
            print(f">>> [Voice] listener stopped with error: {type(error).__name__}: {error}")
            set_radio_ui_state("ERROR", f"{type(error).__name__}: {error}")
            self.schedule_voice_listener_restart(error)
        else:
            print(">>> [Voice] listener stopped")
            set_radio_ui_state("STOPPED", "Listener stopped")

    def schedule_voice_listener_restart(self, error):
        if not hasattr(self, "loop") or self.loop.is_closed() or self.is_closed():
            return

        now = time.time()
        if now - self.last_listener_error_time > 30:
            self.listener_error_count = 0

        self.last_listener_error_time = now
        self.listener_error_count += 1
        delay = min(0.5 * self.listener_error_count, 5.0)
        reason = f"{type(error).__name__}: {error}"

        try:
            self.loop.call_soon_threadsafe(
                lambda: self.loop.create_task(self.restart_voice_listener_after_error(delay, reason))
            )
        except RuntimeError as e:
            print(f">>> [Voice] listener 재시작 예약 실패: {e}")

    async def restart_voice_listener_after_error(self, delay, reason):
        if self.listener_restarting:
            return

        self.listener_restarting = True
        try:
            print(f">>> [Voice] listener 재시작 대기 {delay:.1f}s ({reason})")
            await asyncio.sleep(delay)

            if not self.voice_client or not self.voice_client.is_connected():
                print(">>> [Voice] listener 재시작 중단: voice client가 연결되어 있지 않습니다.")
                return

            if hasattr(self.voice_client, "is_listening") and self.voice_client.is_listening():
                return

            self.start_voice_listener()
        finally:
            self.listener_restarting = False

    def load_config(self):
        self.config = load_bot_config_file()
        return self.config

    def sink_callback(self, user, data: voice_recv.VoiceData):
        if user == self.user: return
        if not getattr(data, "pcm", None): return

        stt_audio = self.prepare_stt_audio(data.pcm)
        if not stt_audio:
            return

        try:
            self.loop.call_soon_threadsafe(self.enqueue_stt_audio, stt_audio)
        except RuntimeError as e:
            print(f">>> [Voice] audio queue put 실패: {e}")

    def prepare_stt_audio(self, pcm):
        audio_array = np.frombuffer(pcm, dtype=np.int16)
        if audio_array.size == 0:
            return b""

        if audio_array.size % 2 == 0:
            stereo = audio_array.reshape(-1, 2)
            left = stereo[:, 0].astype(np.int32)
            right = stereo[:, 1].astype(np.int32)
            left_rms = np.mean(left * left)
            right_rms = np.mean(right * right)
            mono_audio = left if left_rms >= right_rms else right
        else:
            mono_audio = audio_array.astype(np.int32)

        usable = (mono_audio.size // 3) * 3
        if usable <= 0:
            return b""

        resampled = mono_audio[:usable].reshape(-1, 3).mean(axis=1)
        resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
        return resampled.tobytes()

    def enqueue_stt_audio(self, pcm_bytes):
        if not pcm_bytes:
            return

        while self.audio_queue.qsize() > 8:
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        try:
            self.audio_queue.put_nowait(pcm_bytes)
        except asyncio.QueueFull:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(pcm_bytes)
            except asyncio.QueueEmpty:
                pass

    def match_keyword(self, text, target_keys):
        for key in target_keys:
            definition = RADIO_COMMANDS.get(key)
            if not definition:
                continue
            for alias in definition.aliases:
                if alias in text: return key
        return None

    async def handle_recognized_text(self, text):
        text = text.strip().lower()
        if not text:
            return

        now = time.time()
        if text == self.last_handled_stt_text and now - self.last_handled_stt_time < 0.8:
            return

        self.last_handled_stt_text = text
        self.last_handled_stt_time = now

        if now - self.last_partial_log_time > 1:
            self.last_partial_log_time = now
            print(f">>> [STT] text='{text}' state={self.state}")

        # 상태 머신 로직
        if self.state == STATE_IDLE:
            if not self.match_keyword(text, ["wake"]):
                set_radio_ui_state("STANDBY", "Waiting for wake word", "")
                print(f">>> [Radio] ignored while on standby: '{text}'")
                return

            self.last_interaction_time = now
            self.recognizer.Reset()
            self.stt_buffer.clear()
            inline_command = self.match_keyword(
                text, [*TELEMETRY_COMMAND_KEYS, "no"]
            )
            if inline_command and inline_command != "no":
                self.state = STATE_WAITING_COMMAND
                set_radio_ui_state(
                    "COMMAND", f"Command: {inline_command}", text
                )
                print(f">>> [Radio] inline command accepted: {inline_command}")
                await self.handle_interaction(inline_command)
            else:
                set_radio_ui_state("WAKE", f'Wake word: "{text}"', text)
                await self.handle_interaction("wake")

        elif self.state == STATE_WAITING_COMMAND:
            set_radio_ui_state("HEARD", f'Heard "{text}"', text)
            cmd = self.match_keyword(text, [*TELEMETRY_COMMAND_KEYS, "no"])
            if cmd:
                self.last_interaction_time = now
                set_radio_ui_state("COMMAND", f"Command: {cmd}", text)
                print(f">>> [Radio] command accepted: {cmd}")
                self.recognizer.Reset()
                self.stt_buffer.clear()
                await self.handle_interaction(cmd)
            else:
                set_radio_ui_state("LISTENING", "Command not recognized", text)
                print(f">>> [Radio] command not recognized: '{text}'")

        elif self.state == STATE_WAITING_FOLLOWUP:
            set_radio_ui_state("HEARD", f'Heard "{text}"', text)
            cmd = self.match_keyword(text, ["no", *TELEMETRY_COMMAND_KEYS])
            if cmd:
                self.last_interaction_time = now
                set_radio_ui_state("COMMAND", f"Command: {cmd}", text)
                print(f">>> [Radio] follow-up accepted: {cmd}")
                self.recognizer.Reset()
                self.stt_buffer.clear()
                await self.handle_interaction(cmd)
            else:
                set_radio_ui_state("LISTENING", "Follow-up not recognized", text)
                print(f">>> [Radio] follow-up not recognized: '{text}'")

    async def process_audio_queue(self):
        print(">>> [Bot] Listening...")
        while True:
            try:
                if self.audio_queue.qsize() > 8:
                    while self.audio_queue.qsize() > 3:
                        try:
                            self.audio_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break

                try:
                    pcm_data = await asyncio.wait_for(
                        self.audio_queue.get(),
                        timeout=self.stt_idle_flush_seconds,
                    )
                except asyncio.TimeoutError:
                    if self.is_speaking:
                        self.stt_buffer.clear()
                        continue

                    if not self.stt_buffer:
                        continue

                    pcm_data = b""

                if self.is_speaking:
                    self.stt_buffer.clear()
                    continue

                if pcm_data:
                    self.stt_buffer.extend(pcm_data)

                if len(self.stt_buffer) < self.stt_chunk_size:
                    if pcm_data:
                        continue

                    chunks = [bytes(self.stt_buffer)]
                    self.stt_buffer.clear()
                else:
                    chunks = []
                    while len(self.stt_buffer) >= self.stt_chunk_size:
                        chunks.append(bytes(self.stt_buffer[:self.stt_chunk_size]))
                        del self.stt_buffer[:self.stt_chunk_size]

                if not pcm_data:
                    chunks.extend([b"\x00" * self.stt_chunk_size] * self.stt_endpoint_silence_chunks)

                final_texts = []
                partial_text = ""
                should_force_endpoint = not pcm_data
                for chunk in chunks:
                    samples = np.frombuffer(chunk, dtype=np.int16)
                    rms = 0.0
                    peak = 0
                    if samples.size:
                        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                        peak = int(np.max(np.abs(samples)))

                    if rms >= self.stt_voice_rms_threshold:
                        self.stt_had_voice = True
                        self.stt_quiet_chunks = 0
                    elif self.stt_had_voice and rms <= self.stt_silence_rms_threshold:
                        self.stt_quiet_chunks += 1
                        if self.stt_quiet_chunks >= self.stt_silence_chunks_needed:
                            should_force_endpoint = True

                    waveform_complete = self.recognizer.AcceptWaveform(chunk)
                    if waveform_complete:
                        result = json.loads(self.recognizer.Result()).get("text", "").lower()
                        if result:
                            self.stt_had_voice = True
                            final_texts.append(result)
                            self.stt_last_partial_text = ""
                    else:
                        partial = json.loads(self.recognizer.PartialResult()).get("partial", "").lower()
                        if partial:
                            self.stt_had_voice = True
                            partial_text = partial
                            self.stt_last_partial_text = partial

                    now = time.time()
                    if now - self.last_audio_level_log_time > 2:
                        self.last_audio_level_log_time = now
                        print(f">>> [Audio] chunk={len(chunk)} rms={rms:.1f} peak={peak} queue={self.audio_queue.qsize()}")

                endpoint_ready = should_force_endpoint and (self.stt_had_voice or self.stt_last_partial_text or not pcm_data)
                if endpoint_ready:
                    forced_result = json.loads(self.recognizer.FinalResult()).get("text", "").lower()
                    if forced_result:
                        final_texts.append(forced_result)

                recognized_text = " ".join(final_texts) if final_texts else (partial_text or self.stt_last_partial_text)
                if endpoint_ready and recognized_text:
                    print(f">>> [STT] endpoint text='{recognized_text}'")
                await self.handle_recognized_text(recognized_text)

                if endpoint_ready:
                    self.recognizer.Reset()
                    self.stt_buffer.clear()
                    self.stt_last_partial_text = ""
                    self.stt_had_voice = False
                    self.stt_quiet_chunks = 0

            except Exception as e:
                print(f"Loop Error: {e}")
                await asyncio.sleep(0.1)

    async def handle_interaction(self, key):
        self.is_speaking = True
        set_radio_ui_state("RESPONDING", f"Responding: {key}")
        
        # 호출어 대응
        if key == "wake":
            await self.speak_tts(ENGINEER_LINES["wake"])
            self.state = STATE_WAITING_COMMAND
            set_radio_ui_state("LISTENING", "Listening for command")
            # 봇이 말을 마친 '지금' 시간을 기록
            self.last_interaction_time = time.time() 
            self.loop.create_task(self.check_timeout(15, STATE_WAITING_COMMAND))
        
        elif key == "no":
            self.state = STATE_IDLE
            await self.speak_tts(ENGINEER_LINES["cancel"])
            set_radio_ui_state("STANDBY", "Waiting for wake word")
            self.recognizer.Reset()

        else:
            response = self.get_telemetry_response(key)
            if response:
                await self.speak_tts(response)
                await self.speak_tts(ENGINEER_LINES["follow_up"])
                
                # [수정 핵심] 봇이 "Anything else?"라고 물어본 직후에 시간을 초기화
                self.last_interaction_time = time.time() 
                self.state = STATE_WAITING_FOLLOWUP
                set_radio_ui_state("LISTENING", "Listening for follow-up")
                
                # 대기 시간을 10초에서 15초로 늘리는 것을 추천합니다.
                self.loop.create_task(self.check_timeout(15, STATE_WAITING_FOLLOWUP))
            else:
                await self.speak_tts(ENGINEER_LINES["missing_data"])
                self.last_interaction_time = time.time() # 여기서도 시간 갱신
                self.state = STATE_WAITING_COMMAND
                set_radio_ui_state("LISTENING", "Listening for command")

        self.is_speaking = False
        if self.state == STATE_IDLE:
            set_radio_ui_state("STANDBY", "Waiting for wake word")
        else:
            set_radio_ui_state("LISTENING", "Listening for command")
        self.recognizer.Reset()

    async def check_timeout(self, duration, monitor_state):
        await asyncio.sleep(duration)
        if self.state == monitor_state:
            if time.time() - self.last_interaction_time >= duration:
                if monitor_state == STATE_WAITING_COMMAND:
                    await self.speak_tts(ENGINEER_LINES["command_timeout"])
                elif monitor_state == STATE_WAITING_FOLLOWUP:
                    await self.speak_tts(ENGINEER_LINES["follow_up_timeout"])
                self.state = STATE_IDLE
                set_radio_ui_state("STANDBY", "Waiting for wake word")
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
            self.stt_buffer.clear()
            
            # 3. 실제 입력 listener와 공통 인식 루프 다시 실행
            if self.radio_mode == "discord":
                self.start_voice_listener()
            self.processing_task = self.loop.create_task(self.process_audio_queue())
            
            # 4. 상태 초기화
            self.is_speaking = False
            self.state = STATE_IDLE # 대기 상태로 리셋
            self.recognizer.Reset()
            
            print(">>> [Radio] 음성 인식 루프(process_audio_queue) 재시작 성공.")
        except Exception as e:
            print(f">>> [Radio] 재시작 실패: {e}")

    # [핵심] 실제 GTMate 데이터 읽기
    def get_telemetry_response(self, key):
        definition = RADIO_COMMANDS.get(key)
        if not definition or not definition.response_builder:
            return None
        return definition.response_builder(SHARED_GAME_STATE, self.format_time_tts)

    def format_time_tts(self, ms):
        minutes = ms // 60000
        seconds = (ms % 60000) // 1000
        point = (ms % 1000) // 100
        text = ""
        if minutes > 0: text += f"{minutes} minute "
        text += f"{seconds} point {point} seconds"
        return text

    async def speak_tts(self, text):
        if self.radio_mode == "native":
            return await self.speak_native_tts(text)
        if not self.voice_client or not self.voice_client.is_connected(): return
        if not text.strip(): return
        
        self.is_speaking = True
        set_radio_ui_state("SPEAKING", text)
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
                if hasattr(self.voice_client, "stop_playback_only"):
                    self.voice_client.stop_playback_only()
                elif hasattr(self.voice_client, "stop_playing"):
                    self.voice_client.stop_playing()
                else:
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
            if self.state == STATE_IDLE:
                set_radio_ui_state("STANDBY", "Waiting for wake word")
            else:
                set_radio_ui_state("LISTENING", "Listening for command")
            print(">>> [디버그] speak_tts 종료")

    async def speak_native_tts(self, text):
        if sd is None or not text.strip():
            return False

        self.is_speaking = True
        playback_succeeded = False
        set_radio_ui_state("SPEAKING", text)
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        current_env = os.environ.copy()
        current_env["PYTHONIOENCODING"] = "utf-8"
        current_piper = os.path.abspath(PIPER_EXE)
        abs_model_path = os.path.abspath(PIPER_MODEL)

        try:
            piper_cmd = [current_piper, "--model", abs_model_path, "--output-raw"]
            process = subprocess.Popen(
                piper_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creation_flags,
                cwd=os.path.dirname(current_piper),
                env=current_env,
            )
            raw_pcm, error_output = process.communicate(
                input=text.encode("utf-8"), timeout=15
            )
            if process.returncode or not raw_pcm:
                error_text = error_output.decode("utf-8", errors="replace").strip()
                raise RuntimeError(error_text or "Piper returned no audio data.")

            output_info = sd.query_devices(self.native_output_device, "output")
            output_rate = int(
                round(float(output_info.get("default_samplerate") or 48000))
            )
            output_pcm = resample_mono_pcm(raw_pcm, 22050, output_rate)
            await asyncio.to_thread(
                self.play_native_pcm,
                output_pcm,
                output_rate,
                self.native_output_device,
            )
            await asyncio.sleep(0.2)
            playback_succeeded = True
        except Exception as e:
            print(f">>> [Native] TTS error: {type(e).__name__}: {e}")
            set_radio_ui_state("ERROR", f"Native output error: {e}")
        finally:
            self.is_speaking = False
            if playback_succeeded:
                if self.state == STATE_IDLE:
                    set_radio_ui_state("STANDBY", "Waiting for wake word")
                else:
                    set_radio_ui_state("LISTENING", "Listening for command")
        return playback_succeeded

    @staticmethod
    def play_native_pcm(pcm_bytes, sample_rate, device):
        with sd.RawOutputStream(
            device=device,
            channels=1,
            samplerate=sample_rate,
            dtype="int16",
        ) as output_stream:
            output_stream.write(pcm_bytes)

    async def close(self):
        if self.native_input_stream is not None:
            try:
                self.native_input_stream.stop()
                self.native_input_stream.close()
            except Exception as e:
                print(f">>> [Native] input close warning: {e}")
            self.native_input_stream = None

        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            if self.processing_task is not asyncio.current_task():
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
        self.processing_task = None

        if self.native_stop_event is not None:
            self.native_stop_event.set()
        await super().close()

class GT7TelemetryReceiver:
    KEY = b'Simulator Interface Packet GT7 ver 0.0'
    XOR_MAGIC = {'A': 0xDEADBEAF, 'B': 0xDEADBEEF, '~': 0x55FABB4F, 'C': 0xDEADBEEF}
    
    def __init__(self, ps_ip: str = '192.168.0.1', packet_type: str = 'C'):
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
            current_lap_ms = struct.unpack_from('i', data, 0x15C)[0] if len(data) >= 0x160 else -1
            surface_type = data[0x158:0x15C].decode('ascii', errors='ignore').strip('\x00 ') if len(data) >= 0x15C else ""
            wheel_steering_angle = struct.unpack_from('ff', data, 0x160) if len(data) >= 0x168 else (0.0, 0.0)
            wheel_base = struct.unpack_from('f', data, 0x168)[0] if len(data) >= 0x16C else 0.0
            car_category = data[0x16C:0x170].decode('ascii', errors='ignore').strip('\x00 ') if len(data) >= 0x170 else ""
            vehicle_dynamics_raw = struct.unpack_from('f', data, 0x154)[0] if len(data) >= 0x158 else 0.0

            return TelemetryPacket(
                position=struct.unpack_from('fff', data, 0x04),
                velocity=struct.unpack_from('fff', data, 0x10),
                rotation=struct.unpack_from('fff', data, 0x1C),
                speed=struct.unpack_from('f', data, 0x4C)[0],
                rpm=struct.unpack_from('f', data, 0x3C)[0],
                max_rpm=struct.unpack_from('h', data, 0x8A)[0],
                fuel_level=struct.unpack_from('f', data, 0x44)[0],
                fuel_capacity=struct.unpack_from('f', data, 0x48)[0],
                clutch=struct.unpack_from('<f', data, 0xF4)[0],
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
                current_lap_ms=current_lap_ms,
                surface_type=surface_type,
                wheel_steering_angle=wheel_steering_angle,
                wheel_base=wheel_base,
                car_category=car_category,
                vehicle_dynamics_raw=vehicle_dynamics_raw,
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
    BASE_WINDOW_WIDTH = 1200
    BASE_WINDOW_HEIGHT = 800
    PIT_ENTRY_LOOKBACK_SECONDS = 0.5
    FUEL_LAP_RESET_CONFIRM_PACKETS = 3
    TIRE_COLD_THRESHOLD = 60.0
    TIRE_HOT_THRESHOLD = 80.0
    TIRE_BOUNDARY_CAPTURE_WINDOW_MS = 5000
    TIRE_WARNING_DELAY_MS = 20000
    TIRE_WARNING_BUSY_RETRY_MS = 1000
    TIRE_WARNING_MAX_BUSY_RETRIES = 5

    def __init__(self, root):
        self.root = root
        self.root.title("GTMate 1.2.0")
        self.root.geometry(f"{self.BASE_WINDOW_WIDTH}x{self.BASE_WINDOW_HEIGHT}")
        self.root.configure(bg='#000000')
        self.fullscreen_enabled = False
        self.ui_scale = 1.0
        self.scale_after_id = None
        self.scalable_widgets = []
        self.scalable_canvases = []
        self.scalable_progressbars = []
        self.scalable_frames = []
        self.settings_window = None
        self.settings_notebook = None

        self.current_version = "1.2.0"
        self.update_check_started = False
        
        self.receiver = None
        self.current_packet = None
        self.last_packet_time = 0
        self.connection_started_at = 0
        self.connection_timed_out = False
        self.connection_watchdog_after_id = None
        self.last_data_val = (0, 0)
        self.last_change_time = 0

        self.bot = None
        self.bot_running = False
        self.bot_loop = None
        self.bot_thread = None
        self.radio_stop_requested = False
        self.radio_worker_finished = False
        self.radio_worker_error = None
        
        self.create_widgets()
        self.update_radio_status_display()
        self.max_rpm_seen = 5000

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
        self.fuel_strategy_has_data = False
        self.low_fuel_alerts_triggered = set()
        self.best_lap_seen_ms = -1
        self.fuel_strategy_reset_for_standby = False
        self.fuel_lap_reset_candidate = None
        self.fuel_lap_reset_count = 0

        # 랩 경계 시점 기준 타이어 온도 경고 상태
        self.tire_warning_observed_lap = None
        self.tire_warning_boundary_state = None
        self.tire_warning_after_id = None
        self.tire_warning_pending_lap = None
        self.tire_warning_last_announced_lap = None
        
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

        self.register_scalable_ui()
        self.bind_fullscreen_controls()

        # 1. 상태의 시작점: 처음엔 무조건 트랙 위라고 가정합니다.
        self.pit_status = "TRACK"
        
        # 2. 피트 시퀀스 감지용 이전 패킷 값
        self.last_speed_kmh = 0.0
        self.last_vehicle_dynamics_raw = 0.0
        self.last_fuel_level_for_pit = None
        self.last_tire_temps_for_pit = None
        self.pit_stopped_since = None
        self.pit_box_announced = False
        self.pit_sequence_hide_after_id = None
        self.pit_entry_history = deque()
        self.pit_entry_started_at = None
        self.pit_lane_started_at = None
        self.connection_watchdog_after_id = self.root.after(500, self.check_connection_watchdog)
        self.root.after(800, self.finalize_pending_updater)
        self.root.after(2000, self.start_silent_update_check)

    def finalize_pending_updater(self, attempt=0):
        if not os.path.isfile(PENDING_UPDATER_EXE):
            return

        try:
            os.replace(PENDING_UPDATER_EXE, UPDATER_EXE)
            print(">>> [Update] Pending Updater replacement completed")
        except FileNotFoundError:
            # The legacy updater's finishing script completed first.
            return
        except Exception as e:
            if attempt < 119:
                if attempt == 0 or (attempt + 1) % 10 == 0:
                    print(
                        ">>> [Update] Pending Updater is still locked; "
                        f"retrying ({attempt + 1}/120): {e}"
                    )
                self.root.after(
                    500,
                    lambda: self.finalize_pending_updater(attempt + 1),
                )
                return
            print(f">>> [Update] Pending Updater replacement failed: {e}")

    def start_silent_update_check(self):
        if self.update_check_started:
            return

        self.update_check_started = True
        threading.Thread(target=self.check_for_update_silently, daemon=True).start()

    def check_for_update_silently(self):
        try:
            url = "https://raw.githubusercontent.com/GreenRiceCake/GTMate/main/update_manifest.json"
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return

            latest_version = str(response.json().get("version", "")).strip()
            if not latest_version or not is_newer_version(latest_version, self.current_version):
                return

            if os.path.exists(UPDATER_EXE):
                subprocess.Popen([UPDATER_EXE], cwd=BASE_DIR)
            else:
                print(f">>> [Update] Updater not found: {UPDATER_EXE}")
        except Exception as e:
            print(f">>> [Update] Silent check failed: {e}")
        
    def create_widgets(self):
        # 상단 설정
        top_frame = tk.Frame(self.root, bg='#1a1a1a', pady=5)
        top_frame.pack(fill=tk.X)
        
        # 1. PS IP 설정
        tk.Label(top_frame, text="PS IP:", bg='#1a1a1a', fg='gray').pack(side=tk.LEFT, padx=5)
        self.ip_entry = tk.Entry(top_frame, font=('Arial', 10), width=12)
        initial_config = load_bot_config_file()
        self.ip_entry.insert(
            0,
            str(initial_config["playstation"].get("ip") or "192.168.0.1"),
        )
        self.ip_entry.pack(side=tk.LEFT, padx=5)
        
        self.connect_btn = tk.Button(top_frame, text="Connect PS", command=self.toggle_connection, bg='#00ff00', font=('Arial', 10, 'bold'))
        self.connect_btn.pack(side=tk.LEFT, padx=5)

        self.find_ps_btn = tk.Button(top_frame, text="Find PS", command=self.auto_find_ps, bg='#444444', fg='white', font=('Arial', 10, 'bold'))
        self.find_ps_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(top_frame, text="● PS: Ready", bg='#1a1a1a', fg='gray', font=('Arial', 10))
        self.status_label.pack(side=tk.LEFT, padx=10)

        # ----------------------------------------------------
        # [설정 및 라디오 제어]
        # ----------------------------------------------------
        tk.Frame(top_frame, width=2, bg='#333').pack(side=tk.LEFT, fill=tk.Y, padx=10) # 구분선

        self.btn_settings = tk.Button(top_frame, text="Settings", command=self.open_settings, bg='#444', fg='white', font=('Arial', 9))
        self.btn_settings.pack(side=tk.LEFT, padx=5)

        self.btn_bot_toggle = tk.Button(top_frame, text="Start Radio", command=self.toggle_bot, bg='#0080ff', fg='white', font=('Arial', 10, 'bold'))
        self.btn_bot_toggle.pack(side=tk.LEFT, padx=5)
        
        self.lbl_bot_status = tk.Label(top_frame, text="● Radio: OFF", bg='#1a1a1a', fg='gray', font=('Arial', 10))
        self.lbl_bot_status.pack(side=tk.LEFT, padx=5)

        self.radio_info_frame = tk.Frame(top_frame, bg='#1a1a1a')
        self.radio_info_frame.pack(side=tk.LEFT, padx=(5, 0))
        self.radio_state_label = tk.Label(
            self.radio_info_frame,
            text="OFF",
            bg='#1a1a1a',
            fg='gray',
            font=('Arial', 9, 'bold'),
            width=11,
            anchor='w',
        )
        self.radio_state_label.pack(anchor='w')
        self.radio_detail_label = tk.Label(
            self.radio_info_frame,
            text="Radio off",
            bg='#1a1a1a',
            fg='#888888',
            font=('Arial', 8),
            width=32,
            anchor='w',
        )
        self.radio_detail_label.pack(anchor='w')

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
        tk.Label(self.lap_info_frame, text="CURRENT LAP", bg='#1a1a1a', fg='#00ffff', font=('Arial', 12, 'bold')).pack()
        self.current_lap_time_label = tk.Label(self.lap_info_frame, text="--:--:---", bg='#1a1a1a', fg='white', font=('Arial', 24))
        self.current_lap_time_label.pack(pady=(0, 10))
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
        tk.Label(input_sub, text="CLU      BRK      THR", bg='black', fg='white', font=('Arial', 12, 'bold')).pack()
        self.clu_canvas = tk.Canvas(input_sub, width=40, height=250, bg='#222', highlightthickness=0)
        self.clu_canvas.pack(side=tk.LEFT, padx=10)
        self.brk_canvas = tk.Canvas(input_sub, width=40, height=250, bg='#222', highlightthickness=0)
        self.brk_canvas.pack(side=tk.LEFT, padx=10)
        self.thr_canvas = tk.Canvas(input_sub, width=40, height=250, bg='#222', highlightthickness=0)
        self.thr_canvas.pack(side=tk.LEFT, padx=10)

        # 상태 섹션 (확장)
        self.status_box = tk.Frame(self.right_col, bg='#1a1a1a', pady=10, relief=tk.RIDGE, bd=2)
        self.status_box.pack(fill=tk.X, pady=10)
        self.replay_label = tk.Label(self.status_box, text="STATUS: STANDBY", bg='#1a1a1a', fg='#666666', font=('Arial', 12, 'bold'))
        self.replay_label.pack()
        self.lap_count_label = tk.Label(self.status_box, text="LAP: -- / --", bg='#1a1a1a', fg='white', font=('Arial', 12))
        self.lap_count_label.pack(pady=(0, 10))
        self.pit_sequence_title = tk.Label(self.status_box, text="PIT: TRACK", bg='#1a1a1a', fg='#00ff00', font=('Arial', 10, 'bold'))
        self.pit_sequence_frame = tk.Frame(self.status_box, bg='#1a1a1a')
        self.pit_step_labels = {}
        for idx, (key, text) in enumerate((
            ("TRACK", "TRK"),
            ("PIT_ENTRY", "ENT"),
            ("PIT_LANE", "LANE"),
            ("PIT_WORK", "WORK"),
            ("PIT_EXIT", "EXIT"),
        )):
            lbl = tk.Label(
                self.pit_sequence_frame,
                text=text,
                width=5,
                bg='#2a2a2a',
                fg='#777777',
                font=('Arial', 8, 'bold'),
                relief=tk.RIDGE,
                bd=1,
            )
            lbl.grid(row=0, column=idx, padx=1)
            self.pit_step_labels[key] = lbl
        self.pit_sequence_visible = False
        self.update_pit_sequence_display("TRACK")

        # 플래그 표시부
        self.flag_asm = tk.Label(self.status_box, text="ASM", bg='#1a1a1a', fg='#333', font=('Arial', 11, 'bold'))
        self.flag_asm.pack()
        self.flag_tcs = tk.Label(self.status_box, text="TCS", bg='#1a1a1a', fg='#333', font=('Arial', 11, 'bold'))
        self.flag_tcs.pack()
        self.flag_beam = tk.Label(self.status_box, text="HIGH BEAM", bg='#1a1a1a', fg='#333', font=('Arial', 11, 'bold'))
        self.flag_beam.pack()
        self.flag_hand = tk.Label(self.status_box, text="HANDBRAKE", bg='#1a1a1a', fg='#333', font=('Arial', 11, 'bold'))
        self.flag_hand.pack()

    def update_pit_sequence_display(self, status):
        if not hasattr(self, 'pit_step_labels'):
            return

        titles = {
            "TRACK": "PIT: TRACK",
            "PIT_ENTRY": "PIT: ENTRY",
            "PIT_LANE": "PIT: LANE",
            "PIT_WORK": "PIT: WORK",
            "PIT_EXIT": "PIT: EXIT",
        }
        active_colors = {
            "TRACK": "#00ff00",
            "PIT_ENTRY": "#ff9900",
            "PIT_LANE": "#ffff00",
            "PIT_WORK": "#00ffff",
            "PIT_EXIT": "#ff66ff",
        }
        active_color = active_colors.get(status, "#00ff00")

        if hasattr(self, 'pit_sequence_title'):
            self.pit_sequence_title.config(text=titles.get(status, f"PIT: {status}"), fg=active_color)

        for key, label in self.pit_step_labels.items():
            if key == status:
                label.config(bg=active_color, fg='black')
            else:
                label.config(bg='#2a2a2a', fg='#777777')

    def show_pit_sequence_display(self):
        if not hasattr(self, 'pit_sequence_title'):
            return

        if getattr(self, 'pit_sequence_hide_after_id', None) is not None:
            try:
                self.root.after_cancel(self.pit_sequence_hide_after_id)
            except Exception:
                pass
            self.pit_sequence_hide_after_id = None

        if getattr(self, 'pit_sequence_visible', False):
            return

        self.pit_sequence_title.pack(after=self.lap_count_label)
        self.pit_sequence_frame.pack(after=self.pit_sequence_title, pady=(4, 10))
        self.pit_sequence_visible = True

    def hide_pit_sequence_display(self):
        if not hasattr(self, 'pit_sequence_title'):
            return

        if getattr(self, 'pit_sequence_hide_after_id', None) is not None:
            self.pit_sequence_hide_after_id = None

        self.pit_sequence_title.pack_forget()
        self.pit_sequence_frame.pack_forget()
        self.pit_sequence_visible = False

    def schedule_pit_sequence_hide(self, delay_ms=3000):
        if not hasattr(self, 'pit_sequence_title'):
            return

        if getattr(self, 'pit_sequence_hide_after_id', None) is not None:
            try:
                self.root.after_cancel(self.pit_sequence_hide_after_id)
            except Exception:
                pass

        self.pit_sequence_hide_after_id = self.root.after(delay_ms, self.hide_pit_sequence_display)

    def bind_fullscreen_controls(self):
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.exit_fullscreen)
        self.root.bind("<Configure>", self.on_root_configure)

    def toggle_fullscreen(self, event=None):
        self.fullscreen_enabled = not self.fullscreen_enabled
        self.root.attributes("-fullscreen", self.fullscreen_enabled)
        self.schedule_ui_scale_update()
        return "break"

    def exit_fullscreen(self, event=None):
        if self.fullscreen_enabled:
            self.fullscreen_enabled = False
            self.root.attributes("-fullscreen", False)
            self.schedule_ui_scale_update()
        return "break"

    def on_root_configure(self, event):
        if event.widget is self.root:
            self.schedule_ui_scale_update()

    def schedule_ui_scale_update(self):
        if self.scale_after_id is not None:
            try:
                self.root.after_cancel(self.scale_after_id)
            except Exception:
                pass
        self.scale_after_id = self.root.after(80, self.apply_ui_scale)

    def register_scalable_ui(self):
        self.scalable_widgets = []
        self.scalable_canvases = []
        self.scalable_progressbars = []
        self.scalable_frames = []

        self.collect_scalable_widgets(self.root)

        if hasattr(self, 'font_small'):
            self.scalable_widgets.append((self.font_small, "Arial", 12, "bold"))
        if hasattr(self, 'font_huge'):
            self.scalable_widgets.append((self.font_huge, "Arial", 25, "bold"))

        for frame in (getattr(self, 'left_col', None), getattr(self, 'right_col', None)):
            if frame is not None:
                try:
                    width = int(frame.cget("width"))
                    if width > 1:
                        self.scalable_frames.append((frame, width))
                except Exception:
                    pass

        self.apply_ui_scale()

    def collect_scalable_widgets(self, widget):
        try:
            font_name = widget.cget("font")
            if font_name:
                font_info = tkfont.Font(font=font_name)
                family = font_info.actual("family")
                size = abs(int(font_info.actual("size")))
                weight = font_info.actual("weight")
                if size > 0:
                    self.scalable_widgets.append((widget, family, size, weight))
        except Exception:
            pass

        if isinstance(widget, tk.Canvas):
            canvas_width = self.get_positive_widget_int(widget, "width")
            canvas_height = self.get_positive_widget_int(widget, "height")
            self.scalable_canvases.append((widget, canvas_width, canvas_height))

        if isinstance(widget, ttk.Progressbar):
            length = self.get_positive_widget_int(widget, "length")
            if length:
                self.scalable_progressbars.append((widget, length))

        for child in widget.winfo_children():
            self.collect_scalable_widgets(child)

    def get_positive_widget_int(self, widget, option):
        try:
            value = int(float(widget.cget(option)))
            return value if value > 1 else None
        except Exception:
            return None

    def calculate_ui_scale(self):
        width = max(self.root.winfo_width(), self.BASE_WINDOW_WIDTH)
        height = max(self.root.winfo_height(), self.BASE_WINDOW_HEIGHT)
        scale = min(width / self.BASE_WINDOW_WIDTH, height / self.BASE_WINDOW_HEIGHT)
        return max(0.85, min(scale, 1.85))

    def apply_ui_scale(self):
        self.scale_after_id = None
        scale = self.calculate_ui_scale()
        if abs(scale - self.ui_scale) < 0.03:
            return

        self.ui_scale = scale

        for widget, family, base_size, weight in self.scalable_widgets:
            scaled_size = max(7, int(round(base_size * scale)))
            try:
                if isinstance(widget, tkfont.Font):
                    widget.configure(size=scaled_size)
                else:
                    widget.configure(font=(family, scaled_size, weight))
            except Exception:
                pass

        for canvas, base_width, base_height in self.scalable_canvases:
            config = {}
            if base_width:
                config["width"] = max(10, int(round(base_width * scale)))
            if base_height:
                config["height"] = max(10, int(round(base_height * scale)))
            if config:
                try:
                    canvas.configure(**config)
                except Exception:
                    pass

        for progressbar, base_length in self.scalable_progressbars:
            try:
                progressbar.configure(length=max(40, int(round(base_length * scale))))
            except Exception:
                pass

        for frame, base_width in self.scalable_frames:
            try:
                frame.configure(width=max(120, int(round(base_width * scale))))
            except Exception:
                pass

    def toggle_connection(self):
        if self.receiver and self.receiver.running: self.stop_connection()
        else: self.start_connection()

    def auto_find_ps(self):
        if self.receiver and self.receiver.running:
            self.stop_connection()

        self.status_label.config(text="● PS 검색 중...", fg='orange')
        self.connect_btn.config(state='disabled')
        self.find_ps_btn.config(state='disabled', text="Searching...")
        self.ip_entry.config(state='disabled')

        threading.Thread(target=self.run_ps_discovery, daemon=True).start()

    def run_ps_discovery(self):
        ps_ip = self.discover_ps_ip()
        self.root.after(0, lambda: self.finish_ps_discovery(ps_ip))

    def finish_ps_discovery(self, ps_ip):
        self.connect_btn.config(state='normal')
        self.find_ps_btn.config(state='normal', text="Find PS")
        self.ip_entry.config(state='normal')

        if not ps_ip:
            self.status_label.config(text="● PS 검색 실패", fg='red')
            return

        self.ip_entry.delete(0, tk.END)
        self.ip_entry.insert(0, ps_ip)
        self.status_label.config(text=f"● PS 발견: {ps_ip}", fg='#00ff00')
        self.start_connection()

    def get_local_subnet_candidates(self):
        addresses = set()

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                addresses.add(s.getsockname()[0])
        except Exception:
            pass

        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                addresses.add(info[4][0])
        except Exception:
            pass

        candidates = []
        seen = set()
        for address in addresses:
            try:
                ip = ipaddress.ip_address(address)
                if ip.is_loopback or ip.is_link_local:
                    continue

                network = ipaddress.ip_network(f"{address}/24", strict=False)
                for host in network.hosts():
                    host_text = str(host)
                    if host_text == address or host_text in seen:
                        continue
                    seen.add(host_text)
                    candidates.append(host_text)
            except ValueError:
                continue

        return candidates

    def discover_ps_ip(self, timeout=3.0):
        candidates = self.get_local_subnet_candidates()
        if not candidates:
            return None

        probe_receiver = GT7TelemetryReceiver(packet_type='C')
        packet_type = probe_receiver.packet_type.encode()
        deadline = time.time() + timeout
        last_probe_time = 0

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('0.0.0.0', 33740))
                sock.settimeout(0.15)

                while time.time() < deadline:
                    now = time.time()
                    if now - last_probe_time > 0.7:
                        last_probe_time = now
                        for ip in candidates:
                            try:
                                sock.sendto(packet_type, (ip, 33739))
                            except OSError:
                                pass

                    try:
                        data, addr = sock.recvfrom(4096)
                    except socket.timeout:
                        continue

                    sender_ip = addr[0]
                    probe_receiver.ps_ip = sender_ip
                    decrypted = probe_receiver.decrypt_packet(data)
                    if decrypted and probe_receiver.parse_packet(decrypted):
                        return sender_ip

        except OSError as e:
            print(f">>> [PS Auto Find] UDP 포트 사용 실패: {e}")

        return None

    def start_connection(self):
        ps_ip = self.ip_entry.get().strip()
        if not ps_ip: return
        SHARED_GAME_STATE['on_track'] = False
        SHARED_GAME_STATE['race_active'] = False
        self.status_label.config(text="● 연결 시도 중", fg='orange')
        self.current_packet = None
        self.last_packet_time = 0
        self.connection_started_at = time.time()
        self.connection_timed_out = False
        self.receiver = GT7TelemetryReceiver(ps_ip=ps_ip)
        threading.Thread(target=self.receiver.start, args=(self.update_data,), daemon=True).start()
        self.connect_btn.config(text="Disconnect", bg='#ff0000')
        self.find_ps_btn.config(state='disabled')
        self.ip_entry.config(state='disabled')

    def stop_connection(self):
        if self.receiver: self.receiver.stop()
        SHARED_GAME_STATE['on_track'] = False
        SHARED_GAME_STATE['race_active'] = False
        self.receiver = None
        self.current_packet = None
        self.last_packet_time = 0
        self.connection_started_at = 0
        self.connection_timed_out = False
        self.status_label.config(text="● 연결 끊김", fg='red')
        self.connect_btn.config(text="Connect PS", bg='#00ff00')
        self.find_ps_btn.config(state='normal')
        self.ip_entry.config(state='normal')
        self.show_empty_data()

    def check_connection_watchdog(self):
        if self.receiver and self.receiver.running:
            reference_time = self.last_packet_time or self.connection_started_at
            if reference_time and time.time() - reference_time > 2.0:
                if not self.connection_timed_out:
                    self.connection_timed_out = True
                    SHARED_GAME_STATE['on_track'] = False
                    SHARED_GAME_STATE['race_active'] = False
                    self.status_label.config(text="● 연결 실패/대기", fg='red')
                    self.show_connection_lost_data()
                    print(">>> [PS] Packet timeout: no telemetry received for 2 seconds.")

        self.connection_watchdog_after_id = self.root.after(500, self.check_connection_watchdog)

    def update_data(self, packet: TelemetryPacket):
        self.current_packet = packet
        self.last_packet_time = time.time()
        self.connection_timed_out = False
        SHARED_GAME_STATE['fuel_liters'] = packet.fuel_level
        SHARED_GAME_STATE['fuel_percent'] = (packet.fuel_level / packet.fuel_capacity * 100) if packet.fuel_capacity else 0
        SHARED_GAME_STATE['current_lap'] = packet.lap_count
        SHARED_GAME_STATE['current_lap_ms'] = packet.current_lap_ms
        SHARED_GAME_STATE['total_laps'] = packet.total_laps
        SHARED_GAME_STATE['best_lap_ms'] = packet.best_lap
        SHARED_GAME_STATE['last_lap_ms'] = packet.last_lap
        SHARED_GAME_STATE['rank'] = packet.race_rank
        SHARED_GAME_STATE['total_cars'] = packet.total_cars
        SHARED_GAME_STATE['on_track'] = GT7Flags.check(
            packet.flags, GT7Flags.CAR_ON_TRACK
        )
        SHARED_GAME_STATE['race_active'] = (
            packet.lap_count != -1 and packet.race_rank not in (-1, 0xFF)
        )
        self.root.after(0, self.update_gui)

    def speak_engineer(self, text):
        if self.bot and self.bot_loop and self.bot_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.bot.speak_tts(text), self.bot_loop
            )
            return True
        return False

    def cancel_pending_tire_warning(self):
        if self.tire_warning_after_id is not None:
            try:
                self.root.after_cancel(self.tire_warning_after_id)
            except Exception:
                pass
        self.tire_warning_after_id = None
        self.tire_warning_pending_lap = None

    def reset_tire_temperature_warning(self):
        self.cancel_pending_tire_warning()
        self.tire_warning_observed_lap = None
        self.tire_warning_boundary_state = None
        self.tire_warning_last_announced_lap = None

    def classify_tire_temperatures(self, temperatures):
        cold = {
            index
            for index, temperature in enumerate(temperatures)
            if temperature < self.TIRE_COLD_THRESHOLD
        }
        hot = {
            index
            for index, temperature in enumerate(temperatures)
            if temperature >= self.TIRE_HOT_THRESHOLD
        }
        return cold, hot

    def update_tire_temperature_warning(self, packet):
        race_is_active = bool(SHARED_GAME_STATE.get('race_active'))
        car_is_on_track = bool(SHARED_GAME_STATE.get('on_track'))
        if (
            not race_is_active
            or not car_is_on_track
            or self.pit_status != "TRACK"
            or packet.lap_count <= 0
        ):
            if (
                self.tire_warning_observed_lap is not None
                or self.tire_warning_after_id is not None
            ):
                self.reset_tire_temperature_warning()
            return

        current_lap = packet.lap_count
        current_state = self.classify_tire_temperatures(packet.tire_temps)

        if self.tire_warning_observed_lap is None:
            self.tire_warning_observed_lap = current_lap
            # 랩 시작 직후 연결됐다면 첫 패킷도 경계 측정값으로 사용할 수 있습니다.
            if (
                0 <= packet.current_lap_ms
                <= self.TIRE_BOUNDARY_CAPTURE_WINDOW_MS
            ):
                self.tire_warning_boundary_state = current_state
                print(
                    f">>> [Tires] Initial boundary captured at lap {current_lap}"
                )
            return

        if current_lap == self.tire_warning_observed_lap:
            return

        previous_lap = self.tire_warning_observed_lap
        self.tire_warning_observed_lap = current_lap
        self.cancel_pending_tire_warning()

        if current_lap != previous_lap + 1:
            self.tire_warning_boundary_state = current_state
            print(
                f">>> [Tires] Lap sequence reset: {previous_lap}->{current_lap}"
            )
            return

        if self.tire_warning_boundary_state is None:
            self.tire_warning_boundary_state = current_state
            print(f">>> [Tires] Boundary baseline set at lap {current_lap}")
            return

        previous_cold, previous_hot = self.tire_warning_boundary_state
        current_cold, current_hot = current_state
        candidate_cold = previous_cold & current_cold
        candidate_hot = previous_hot & current_hot
        self.tire_warning_boundary_state = current_state

        print(
            f">>> [Tires] Lap {current_lap} candidates: "
            f"cold={sorted(candidate_cold)}, hot={sorted(candidate_hot)}"
        )
        if not candidate_cold and not candidate_hot:
            return

        self.tire_warning_pending_lap = current_lap
        self.tire_warning_after_id = self.root.after(
            self.TIRE_WARNING_DELAY_MS,
            lambda lap=current_lap,
                   cold=tuple(sorted(candidate_cold)),
                   hot=tuple(sorted(candidate_hot)): self.evaluate_tire_warning(
                       lap, cold, hot
                   ),
        )

    def evaluate_tire_warning(
        self, lap, candidate_cold, candidate_hot, busy_retry=0
    ):
        self.tire_warning_after_id = None
        if (
            self.tire_warning_pending_lap != lap
            or self.tire_warning_last_announced_lap == lap
        ):
            return

        packet = self.current_packet
        if (
            packet is None
            or time.time() - self.last_packet_time > 2.0
            or packet.lap_count != lap
            or not SHARED_GAME_STATE.get('race_active')
            or not SHARED_GAME_STATE.get('on_track')
            or self.pit_status != "TRACK"
        ):
            self.tire_warning_pending_lap = None
            print(f">>> [Tires] Lap {lap} warning discarded: state changed")
            return

        current_cold, current_hot = self.classify_tire_temperatures(
            packet.tire_temps
        )
        confirmed_cold = set(candidate_cold) & current_cold
        confirmed_hot = set(candidate_hot) & current_hot
        if not confirmed_cold and not confirmed_hot:
            self.tire_warning_pending_lap = None
            print(f">>> [Tires] Lap {lap} temperatures returned to normal")
            return

        radio_is_busy = (
            self.bot is not None
            and (
                self.bot.is_speaking
                or self.bot.state != STATE_IDLE
            )
        )
        if radio_is_busy:
            if busy_retry < self.TIRE_WARNING_MAX_BUSY_RETRIES:
                self.tire_warning_after_id = self.root.after(
                    self.TIRE_WARNING_BUSY_RETRY_MS,
                    lambda: self.evaluate_tire_warning(
                        lap,
                        tuple(sorted(confirmed_cold)),
                        tuple(sorted(confirmed_hot)),
                        busy_retry + 1,
                    ),
                )
            else:
                self.tire_warning_pending_lap = None
                print(f">>> [Tires] Lap {lap} warning skipped: radio busy")
            return

        message = build_tire_temperature_warning(
            confirmed_cold, confirmed_hot
        )
        self.tire_warning_pending_lap = None
        if message and self.speak_engineer(message):
            self.tire_warning_last_announced_lap = lap
            print(f">>> [Tires] Lap {lap} warning: {message}")
        elif message:
            print(f">>> [Tires] Lap {lap} warning skipped: radio off")

    def reset_fuel_strategy(self, reason="standby"):
        if reason == "standby" and not self.fuel_strategy_reset_for_standby:
            print(">>> [Fuel] Strategy data reset for standby.")
        elif reason != "standby":
            print(f">>> [Fuel] Strategy data reset for {reason}.")

        self.last_lap_count = -1
        self.fuel_at_lap_start = -1
        self.fuel_consumption_history.clear()
        self.avg_fuel_per_lap = 0
        self.fuel_strategy_has_data = False
        self.low_fuel_alerts_triggered.clear()
        self.best_lap_seen_ms = -1
        self.fuel_lap_reset_candidate = None
        self.fuel_lap_reset_count = 0
        self.fuel_strategy_reset_for_standby = reason == "standby"
        SHARED_GAME_STATE['laps_remain'] = 0.0
        SHARED_GAME_STATE['laps_remain_ready'] = False

    def mark_fuel_strategy_active(self):
        self.fuel_strategy_reset_for_standby = False

    def format_time_tts(self, ms):
        minutes = ms // 60000
        seconds = (ms % 60000) // 1000
        tenths = (ms % 1000) // 100
        if minutes > 0:
            return f"{minutes} minute {seconds} point {tenths} seconds"
        return f"{seconds} point {tenths} seconds"

    def announce_best_lap_if_needed(self, best_lap_ms):
        if best_lap_ms <= 0:
            return

        if self.best_lap_seen_ms > 0 and best_lap_ms < self.best_lap_seen_ms:
            delta_ms = self.best_lap_seen_ms - best_lap_ms
            msg = ENGINEER_LINES["new_best_lap"].format(
                lap_time=self.format_time_tts(best_lap_ms),
                delta_time=self.format_time_tts(delta_ms),
            )
            self.speak_engineer(msg)

        if self.best_lap_seen_ms <= 0 or best_lap_ms < self.best_lap_seen_ms:
            self.best_lap_seen_ms = best_lap_ms

    def announce_low_fuel_if_needed(self, laps_remain):
        threshold = None
        if laps_remain <= 1:
            threshold = 1
        elif laps_remain <= 2:
            threshold = 2
        elif laps_remain <= 3:
            threshold = 3

        if threshold and threshold not in self.low_fuel_alerts_triggered:
            self.low_fuel_alerts_triggered.add(threshold)
            self.speak_engineer(
                ENGINEER_LINES["low_fuel"].format(laps=threshold)
            )

    def update_gui(self):
        now = time.time()
        if not self.current_packet or (now - self.last_packet_time > 2.0):
            self.status_label.config(text="● 연결 실패/대기", fg='red')
            self.show_connection_lost_data()
            return
        
        self.status_label.config(text="● 연결됨", fg='#00ff00')
        p = self.current_packet
        on_track = GT7Flags.check(p.flags, GT7Flags.CAR_ON_TRACK)
        
        if abs(p.speed - self.last_data_val[0]) > 0.1 or abs(p.rpm - self.last_data_val[1]) > 1:
            self.last_change_time = now
            self.last_data_val = (p.speed, p.rpm)

        is_moving = (now - self.last_change_time < 5.0)

        rank_is_standby = p.race_rank in (-1, 0xFF)
        if p.lap_count == -1 or rank_is_standby:
            self.show_empty_data()
            return

        if on_track:
            self.mark_fuel_strategy_active()
            self.replay_label.config(text="STATUS: ON TRACK", fg='#00ff00')
            self.render_dashboard(p)
        elif is_moving:
            self.mark_fuel_strategy_active()
            self.replay_label.config(text="STATUS: REPLAY MODE", fg='yellow')
            self.render_dashboard(p)

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

    def update_radio_status_display(self):
        if not hasattr(self, 'radio_state_label'):
            return

        if self.radio_worker_finished:
            self.radio_worker_finished = False
            self.set_radio_controls_stopped(self.radio_worker_error)
            self.radio_worker_error = None

        status = RADIO_UI_STATE.get("status", "OFF")
        detail = RADIO_UI_STATE.get("detail", "")
        heard = RADIO_UI_STATE.get("heard", "")

        colors = {
            "OFF": "gray",
            "STANDBY": "#888888",
            "STARTING": "orange",
            "CONNECTING": "orange",
            "STOPPING": "orange",
            "LISTENING": "#00ff00",
            "HEARD": "#00ffff",
            "WAKE": "#00ffff",
            "COMMAND": "#ffff00",
            "RESPONDING": "#ff99ff",
            "SPEAKING": "#ff99ff",
            "ERROR": "#ff4444",
            "STOPPED": "#ff4444",
        }
        color = colors.get(status, "white")

        if heard and status in ("HEARD", "WAKE", "COMMAND"):
            detail = f'Heard: "{heard}"'

        if len(detail) > 40:
            detail = detail[:37] + "..."

        self.radio_state_label.config(text=status, fg=color)
        self.radio_detail_label.config(text=detail or "Radio ready", fg=color if status != "OFF" else "#888888")
        self.root.after(250, self.update_radio_status_display)

    def render_dashboard(self, p):
        self.speed_label.config(text=f"{int(p.speed * 3.6)}")
        self.gear_label.config(text="R" if p.current_gear == 0 else ("N" if p.current_gear == 15 else str(p.current_gear)))
        self.draw_rpm_bar(p.rpm, p.max_rpm, GT7Flags.check(p.flags, GT7Flags.REV_LIMITER))
        
        # 랩타임 업데이트
        if p.best_lap > 0:
            self.announce_best_lap_if_needed(p.best_lap)
            self.best_lap_label.config(text=self.format_time(p.best_lap))
        if p.current_lap_ms > 0: self.current_lap_time_label.config(text=self.format_time(p.current_lap_ms))
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
        if not hasattr(self, 'last_vehicle_dynamics_raw'): self.last_vehicle_dynamics_raw = p.vehicle_dynamics_raw
        if not hasattr(self, 'last_fuel_level_for_pit'): self.last_fuel_level_for_pit = p.fuel_level
        if not hasattr(self, 'last_tire_temps_for_pit'): self.last_tire_temps_for_pit = p.tire_temps
        if not hasattr(self, 'pit_stopped_since'): self.pit_stopped_since = None
        if not hasattr(self, 'pit_box_announced'): self.pit_box_announced = False
        if not hasattr(self, 'pit_entry_history'):
            self.pit_entry_history = deque()
        if not hasattr(self, 'pit_entry_started_at'): self.pit_entry_started_at = None
        if not hasattr(self, 'pit_lane_started_at'): self.pit_lane_started_at = None
        
        new_status = self.pit_status
        now = time.monotonic()
        dyn_abs = abs(p.vehicle_dynamics_raw)
        dynamics_near_zero = dyn_abs <= 0.015
        self.pit_entry_history.append((now, kmh, dyn_abs))
        while (
            self.pit_entry_history
            and now - self.pit_entry_history[0][0] > self.PIT_ENTRY_LOOKBACK_SECONDS
        ):
            self.pit_entry_history.popleft()
        recent_peak_speed = max((sample[1] for sample in self.pit_entry_history), default=kmh)
        entry_freeze_detected = (
            recent_peak_speed > 40.0
            and kmh <= 1.0
            and dynamics_near_zero
        )
        if self.pit_status == "TRACK" and kmh <= 1.0 < self.last_speed_kmh:
            print(
                f">>> [PIT ENTRY CHECK] "
                f"window={self.PIT_ENTRY_LOOKBACK_SECONDS:.2f}s, "
                f"peak={recent_peak_speed:.1f}, speed={kmh:.1f}, "
                f"dynamics={p.vehicle_dynamics_raw:.4f}, lap={p.lap_count}, "
                f"detected={entry_freeze_detected and p.lap_count > 0}"
            )
        tires_in_pit_range = all(58.5 <= t <= 60.5 for t in p.tire_temps)
        last_tires_in_pit_range = (
            self.last_tire_temps_for_pit is not None
            and all(58.5 <= t <= 60.5 for t in self.last_tire_temps_for_pit)
        )
        tires_reset_for_pit = tires_in_pit_range and not last_tires_in_pit_range
        refueling_detected = (
            self.last_fuel_level_for_pit is not None
            and p.fuel_level > self.last_fuel_level_for_pit + 0.5
        )
        pit_work_cue = tires_reset_for_pit or refueling_detected
        
        # -------------------------------------------------------------------------
        # [0. 피트 진입/작업 감지]
        # -------------------------------------------------------------------------
        if pit_work_cue and self.pit_status in ("TRACK", "PIT_ENTRY", "PIT_LANE"):
            reason = "tires" if tires_reset_for_pit else "refuel"
            print(f">>> [PIT WORK DETECTED] cue={reason}, speed={kmh:.1f}")
            new_status = "PIT_WORK"
            self.pit_stopped_since = None

        elif self.pit_status == "TRACK":
            if entry_freeze_detected and p.lap_count > 0:
                print(
                    f">>> [PIT ENTRY DETECTED] "
                    f"{self.PIT_ENTRY_LOOKBACK_SECONDS:.2f}s peak {recent_peak_speed:.1f}->{kmh:.1f}, "
                    f"dynamics {self.last_vehicle_dynamics_raw:.4f}->{p.vehicle_dynamics_raw:.4f}"
                )
                new_status = "PIT_ENTRY"
                self.pit_stopped_since = None
                self.pit_entry_started_at = now

        # -------------------------------------------------------------------------
        # [1. 피트 작업 및 탈출]
        # -------------------------------------------------------------------------
        elif self.pit_status == "PIT_ENTRY":
            if kmh >= 50.0:
                new_status = "PIT_LANE"
            elif (
                self.pit_entry_started_at
                and now - self.pit_entry_started_at > 15.0
                and not (kmh <= 2.0 and dynamics_near_zero)
            ):
                print(">>> [PIT ENTRY TIMEOUT] PIT_ENTRY -> TRACK")
                new_status = "TRACK"

        elif self.pit_status == "PIT_LANE":
            if kmh <= 2.0 and dynamics_near_zero:
                if self.pit_stopped_since is None:
                    self.pit_stopped_since = now
                elif now - self.pit_stopped_since >= 2.0:
                    new_status = "PIT_WORK"
                    self.pit_stopped_since = None
            elif self.pit_lane_started_at and now - self.pit_lane_started_at > 60.0:
                print(">>> [PIT LANE TIMEOUT] PIT_LANE -> TRACK")
                new_status = "TRACK"
            else:
                self.pit_stopped_since = None

        elif self.pit_status == "PIT_WORK":
            if kmh >= 100.0:
                new_status = "TRACK"
            elif kmh >= 50.0:
                new_status = "PIT_EXIT"

        elif self.pit_status == "PIT_EXIT":
            if kmh >= 85.0:
                new_status = "TRACK"

        if new_status != self.pit_status:
            print(f">>> [PIT STATUS] {self.pit_status} -> {new_status}")
            if new_status != "TRACK":
                self.show_pit_sequence_display()
            if new_status == "PIT_LANE" and self.pit_status != "PIT_LANE":
                self.pit_lane_started_at = now
            if new_status in ("PIT_LANE", "PIT_WORK") and not self.pit_box_announced:
                self.pit_box_announced = True
                self.speak_engineer(ENGINEER_LINES["pit_box"])
            if new_status == "TRACK" and self.pit_status != "TRACK":
                self.fuel_at_lap_start = p.fuel_level
                self.last_lap_count = p.lap_count
                self.pit_stopped_since = None
                self.pit_entry_started_at = None
                self.pit_lane_started_at = None
                self.pit_box_announced = False
                self.speak_engineer(ENGINEER_LINES["pit_exit"])
                print(f">>> [Fuel] Pit exit baseline reset: lap={p.lap_count}, fuel={p.fuel_level:.2f}")
                print(">>> [System] 피트 아웃: 엔지니어 안내 전송")
            elif new_status != "PIT_ENTRY":
                self.pit_entry_started_at = None
                if new_status != "PIT_LANE":
                    self.pit_lane_started_at = None
            self.pit_status = new_status
            self.update_pit_sequence_display(self.pit_status)
            if self.pit_status == "TRACK":
                self.schedule_pit_sequence_hide(3000)

        self.last_speed_kmh = kmh
        self.last_vehicle_dynamics_raw = p.vehicle_dynamics_raw
        self.last_fuel_level_for_pit = p.fuel_level
        self.last_tire_temps_for_pit = p.tire_temps
        self.update_tire_temperature_warning(p)


        # -------------------------------------------------------------------------
        # [연료 및 전략 시스템 로직 - 기존 코드]
        # -------------------------------------------------------------------------
        # 피트 작업 중이 아닐 때만(TRACK) 랩당 소모량을 계산하여 데이터 오염 방지
        if p.fuel_capacity > 0:
            if 0 <= p.lap_count < self.last_lap_count:
                if self.fuel_lap_reset_candidate == p.lap_count:
                    self.fuel_lap_reset_count += 1
                else:
                    self.fuel_lap_reset_candidate = p.lap_count
                    self.fuel_lap_reset_count = 1

                if self.fuel_lap_reset_count >= self.FUEL_LAP_RESET_CONFIRM_PACKETS:
                    previous_lap = self.last_lap_count
                    self.reset_fuel_strategy(
                        reason=f"new session (lap {previous_lap} -> {p.lap_count})"
                    )
                    self.display_fuel_pct = (p.fuel_level / p.fuel_capacity) * 100
            else:
                self.fuel_lap_reset_candidate = None
                self.fuel_lap_reset_count = 0
            
            # 1. 초기화 로직 (게임 시작 직후 한 번만 실행)
            if self.last_lap_count == -1:
                self.last_lap_count = p.lap_count
                self.fuel_at_lap_start = p.fuel_level
                if not hasattr(self, 'display_fuel_pct'):
                    self.display_fuel_pct = (p.fuel_level / p.fuel_capacity) * 100

            # 2. 랩 변경 감지 및 평균 소모량 계산
            if p.lap_count > self.last_lap_count:
                if self.pit_status != "TRACK":
                    print(f">>> [Fuel] Lap change ignored during pit: {self.last_lap_count}->{p.lap_count}, status={self.pit_status}")
                elif self.last_lap_count == 0 and p.lap_count == 1:
                    print("스타트 랩 감지: 첫 데이터 제외")
                else:
                    if self.fuel_at_lap_start > 0:
                        fuel_used = self.fuel_at_lap_start - p.fuel_level
                        min_fuel_used = max(0.05, p.fuel_capacity * 0.001)
                        
                        # 피트인 상태가 아닐 때만 히스토리에 추가 (2중 방어)
                        if min_fuel_used < fuel_used < p.fuel_capacity * 0.3:
                            self.fuel_consumption_history.append(fuel_used)
                            if len(self.fuel_consumption_history) > 5:
                                self.fuel_consumption_history.pop(0)
                            self.avg_fuel_per_lap = sum(self.fuel_consumption_history) / len(self.fuel_consumption_history)
                            self.fuel_strategy_has_data = True
                            print(
                                f">>> [Fuel] Lap fuel used={fuel_used:.2f}, "
                                f"avg={self.avg_fuel_per_lap:.2f}, samples={len(self.fuel_consumption_history)}"
                            )
                        else:
                            print(
                                f">>> [Fuel] Lap fuel sample rejected: used={fuel_used:.2f}, "
                                f"min={min_fuel_used:.2f}, cap={p.fuel_capacity:.2f}"
                            )

                self.fuel_at_lap_start = p.fuel_level
                self.last_lap_count = p.lap_count
            
            # 주행 중 급유 감지 (피트 상태 업데이트를 놓쳤을 경우를 대비한 백업)
            elif p.fuel_level > self.fuel_at_lap_start + 1.0:
                 self.fuel_at_lap_start = p.fuel_level
                 self.low_fuel_alerts_triggered.clear()


            # 3. 화면 표시용 데이터 가공 (노이즈 필터링)
            # -------------------------------------------------
            raw_pct = (p.fuel_level / p.fuel_capacity) * 100
            
            if (raw_pct < self.display_fuel_pct) or (self.pit_status != "TRACK") or (raw_pct - self.display_fuel_pct > 5.0):
                self.display_fuel_pct = raw_pct
                
            display_int_pct = int(self.display_fuel_pct)


            # 4. 남은 랩 수 텍스트 생성
            # -------------------------------------------------
            sub_text = "CALC..."
            
            # 유효 판정을 통과한 연료 표본이 있으면 차량 소비량과 관계없이 계산 시작
            if (
                self.fuel_strategy_has_data
                and len(self.fuel_consumption_history) >= 1
                and self.avg_fuel_per_lap > 0.0
            ):
                laps_remain = p.fuel_level / self.avg_fuel_per_lap

                # [봇 공유 데이터 업데이트]
                SHARED_GAME_STATE['laps_remain'] = laps_remain
                SHARED_GAME_STATE['laps_remain_ready'] = True
                self.announce_low_fuel_if_needed(laps_remain)
                
                # 현실적인 랩 수 표시 제한
                if laps_remain > 50:
                    laps_str = "50+"
                else:
                    laps_str = f"{laps_remain:.1f}"
                
                sub_text = f"({laps_str} LAPS)"
            else:
                # 데이터 수집 전에는 현재 리터 표시
                SHARED_GAME_STATE['laps_remain'] = 0.0
                SHARED_GAME_STATE['laps_remain_ready'] = False
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
        self.draw_vertical_bar(self.clu_canvas, p.clutch, '#00ffff')
        self.lap_count_label.config(text=f"LAP: {p.lap_count} / {max(0, p.total_laps)}")

    def draw_rpm_bar(self, rpm, max_rpm, is_limiter):
        self.rpm_canvas.delete("all")
        w = self.rpm_canvas.winfo_width()
        
        limit = max_rpm if max_rpm > 0 else 10000
        pct = min(rpm / limit, 1.0)
        
        fill_w = w * pct
        
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

    def clear_dashboard_values(self):
        self.speed_label.config(text="--")
        self.gear_label.config(text="--")
        self.fuel_label.config(text="-- L")
        self.best_lap_label.config(text="--:--:---")
        self.current_lap_time_label.config(text="--:--:---")
        self.last_lap_label.config(text="--:--:---")
        self.lap_count_label.config(text="LAP: -- / --")
        self.pos_label.config(text="POS: -- / --", fg="white")
        self.rpm_canvas.delete("all")
        self.clu_canvas.delete("all")
        self.thr_canvas.delete("all")
        self.brk_canvas.delete("all")
        for name, label in zip(("FL", "FR", "RL", "RR"), self.tire_labels):
            label.config(text=f"{name}\n--°C", fg="#333")
        self.boost_canvas.delete("all")
        self.boost_label.config(text="BOOST: -- bar")
        if self.has_turbo_active:
            self.boost_frame.place_forget()
            self.has_turbo_active = False
        for f in [self.flag_asm, self.flag_tcs, self.flag_beam, self.flag_hand]: f.config(fg='#333')

    def show_connection_lost_data(self):
        SHARED_GAME_STATE['on_track'] = False
        SHARED_GAME_STATE['race_active'] = False
        self.reset_tire_temperature_warning()
        self.replay_label.config(text="STATUS: CONNECTION LOST", fg='#ff4444')
        self.clear_dashboard_values()

    def show_empty_data(self):
        SHARED_GAME_STATE['on_track'] = False
        SHARED_GAME_STATE['race_active'] = False
        self.reset_tire_temperature_warning()
        self.replay_label.config(text="STATUS: STANDBY", fg='#666666')
        self.pit_status = "TRACK"
        self.pit_stopped_since = None
        self.pit_entry_started_at = None
        self.pit_lane_started_at = None
        self.pit_entry_history = deque()
        self.pit_box_announced = False
        self.update_pit_sequence_display("TRACK")
        if getattr(self, 'pit_sequence_hide_after_id', None) is not None:
            try:
                self.root.after_cancel(self.pit_sequence_hide_after_id)
            except Exception:
                pass
            self.pit_sequence_hide_after_id = None
        self.hide_pit_sequence_display()
        self.reset_fuel_strategy()
        self.clear_dashboard_values()
        self.max_rpm_seen = 5000

    def format_time(self, ms):
        minutes, seconds = divmod(ms // 1000, 60)
        return f"{minutes:02d}:{seconds:02d}.{ms % 1000:03d}"

    # ---------------------------------------------------
    # [설정 및 봇 관련 메서드]
    # ---------------------------------------------------
    def open_settings(self, initial_tab="general"):
        if self.settings_window and self.settings_window.winfo_exists():
            self.settings_window.deiconify()
            self.settings_window.lift()
            self.settings_window.focus_force()
            if initial_tab == "radio" and self.settings_notebook:
                self.settings_notebook.select(1)
            return

        win = tk.Toplevel(self.root)
        self.settings_window = win
        win.title("GTMate Settings")
        win.geometry("600x460")
        win.minsize(560, 420)
        win.transient(self.root)
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        config = load_bot_config_file()

        notebook = ttk.Notebook(win)
        self.settings_notebook = notebook
        notebook.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 8))

        general_tab = ttk.Frame(notebook, padding=16)
        radio_tab = ttk.Frame(notebook, padding=16)
        notebook.add(general_tab, text="General")
        notebook.add(radio_tab, text="Team Radio")

        general_tab.columnconfigure(0, weight=1)
        ps_frame = ttk.LabelFrame(general_tab, text="PlayStation", padding=14)
        ps_frame.grid(row=0, column=0, sticky="ew")
        ps_frame.columnconfigure(1, weight=1)
        ttk.Label(ps_frame, text="Default IP address").grid(
            row=0, column=0, sticky="w", padx=(0, 14), pady=5
        )
        e_ps_ip = ttk.Entry(ps_frame)
        e_ps_ip.insert(
            0,
            str(config["playstation"].get("ip") or self.ip_entry.get().strip()),
        )
        e_ps_ip.grid(row=0, column=1, sticky="ew", pady=5)

        radio_tab.columnconfigure(0, weight=1)
        mode_frame = ttk.LabelFrame(radio_tab, text="Mode", padding=10)
        mode_frame.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        mode_var = tk.StringVar(value=config["radio"]["mode"])

        ttk.Radiobutton(
            mode_frame,
            text="Discord",
            value="discord",
            variable=mode_var,
        ).pack(side=tk.LEFT, padx=(4, 18))
        ttk.Radiobutton(
            mode_frame,
            text="Native",
            value="native",
            variable=mode_var,
        ).pack(side=tk.LEFT, padx=4)

        discord_frame = ttk.LabelFrame(radio_tab, text="Discord", padding=14)
        discord_frame.grid(row=1, column=0, sticky="ew")
        discord_frame.columnconfigure(1, weight=1)

        discord_config = config["radio"]["discord"]

        ttk.Label(discord_frame, text="Bot token").grid(
            row=0, column=0, sticky="w", padx=(0, 14), pady=5
        )
        e_token = ttk.Entry(discord_frame, show="*")
        e_token.insert(0, str(discord_config.get("token") or ""))
        e_token.grid(row=0, column=1, sticky="ew", pady=5)

        ttk.Label(discord_frame, text="Voice channel ID").grid(
            row=1, column=0, sticky="w", padx=(0, 14), pady=5
        )
        e_channel = ttk.Entry(discord_frame)
        e_channel.insert(0, str(discord_config.get("channel_id") or ""))
        e_channel.grid(row=1, column=1, sticky="ew", pady=5)

        show_token_var = tk.BooleanVar(value=False)

        def update_token_visibility():
            e_token.configure(show="" if show_token_var.get() else "*")

        ttk.Checkbutton(
            discord_frame,
            text="Show token",
            variable=show_token_var,
            command=update_token_visibility,
        ).grid(row=2, column=1, sticky="w", pady=(8, 0))

        native_frame = ttk.LabelFrame(radio_tab, text="Native audio", padding=14)
        native_frame.columnconfigure(1, weight=1)
        ttk.Label(native_frame, text="Microphone").grid(
            row=0, column=0, sticky="w", padx=(0, 14), pady=5
        )
        input_device_var = tk.StringVar(value="System default")
        input_device_combo = ttk.Combobox(
            native_frame, textvariable=input_device_var, state="readonly"
        )
        input_device_combo.grid(row=0, column=1, sticky="ew", pady=5)

        ttk.Label(native_frame, text="Output device").grid(
            row=1, column=0, sticky="w", padx=(0, 14), pady=5
        )
        output_device_var = tk.StringVar(value="System default")
        output_device_combo = ttk.Combobox(
            native_frame, textvariable=output_device_var, state="readonly"
        )
        output_device_combo.grid(row=1, column=1, sticky="ew", pady=5)

        audio_status_var = tk.StringVar(value="")
        ttk.Label(native_frame, textvariable=audio_status_var).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        input_device_map = {}
        output_device_map = {}
        audio_devices_loaded = False

        def selected_device_label(device_map, saved_selection, default_label):
            saved_selection = normalize_audio_device(saved_selection)
            for label, descriptor in device_map.items():
                if descriptor == saved_selection:
                    return label
            return default_label

        def refresh_audio_devices():
            nonlocal audio_devices_loaded
            previous_input = input_device_map.get(input_device_var.get())
            previous_output = output_device_map.get(output_device_var.get())
            input_device_map.clear()
            output_device_map.clear()
            default_input_label = get_default_audio_device_label("input")
            default_output_label = get_default_audio_device_label("output")
            input_device_map[default_input_label] = None
            output_device_map[default_output_label] = None
            if sd is not None:
                for label, descriptor, _index in list_audio_devices("input"):
                    input_device_map[label] = descriptor
                for label, descriptor, _index in list_audio_devices("output"):
                    output_device_map[label] = descriptor

            input_device_combo["values"] = tuple(input_device_map)
            output_device_combo["values"] = tuple(output_device_map)
            native_config = config["radio"]["native"]
            wanted_input = (
                previous_input
                if audio_devices_loaded
                else native_config.get("input_device")
            )
            wanted_output = (
                previous_output
                if audio_devices_loaded
                else native_config.get("output_device")
            )
            input_device_var.set(
                selected_device_label(
                    input_device_map, wanted_input, default_input_label
                )
            )
            output_device_var.set(
                selected_device_label(
                    output_device_map, wanted_output, default_output_label
                )
            )
            audio_devices_loaded = True
            if sd is None:
                audio_status_var.set("Native audio is unavailable.")
            else:
                audio_status_var.set(
                    f"{len(input_device_map) - 1} inputs, "
                    f"{len(output_device_map) - 1} outputs"
                )

        ttk.Button(
            native_frame,
            text="Refresh devices",
            command=refresh_audio_devices,
        ).grid(row=3, column=1, sticky="e", pady=(12, 0))

        def update_mode_panel(*_args):
            if mode_var.get() == "native":
                discord_frame.grid_remove()
                native_frame.grid(row=1, column=0, sticky="ew")
            else:
                native_frame.grid_remove()
                discord_frame.grid(row=1, column=0, sticky="ew")

        mode_var.trace_add("write", update_mode_panel)
        refresh_audio_devices()
        update_mode_panel()

        config_error_var = tk.StringVar(value="")
        error_label = tk.Label(win, textvariable=config_error_var, fg='#cc2222')
        error_label.grid(row=1, column=0, sticky="w", padx=16)

        button_frame = ttk.Frame(win)
        button_frame.grid(row=2, column=0, sticky="e", padx=12, pady=(8, 12))

        def close_settings():
            self.settings_window = None
            self.settings_notebook = None
            win.destroy()

        def save():
            ps_ip = e_ps_ip.get().strip()
            try:
                parsed_ip = ipaddress.ip_address(ps_ip)
                if parsed_ip.version != 4:
                    raise ValueError
            except ValueError:
                notebook.select(general_tab)
                config_error_var.set("PlayStation IP must be a valid IPv4 address.")
                e_ps_ip.focus_set()
                return

            channel_id = e_channel.get().strip()
            if channel_id and not channel_id.isdigit():
                notebook.select(radio_tab)
                config_error_var.set("Voice Channel ID must contain digits only.")
                e_channel.focus_set()
                return

            if mode_var.get() == "native" and sd is None:
                notebook.select(radio_tab)
                config_error_var.set(
                    "Native audio requires the sounddevice package."
                )
                return

            new_cfg = normalize_settings(config)
            new_cfg["playstation"]["ip"] = ps_ip
            new_cfg["radio"]["mode"] = mode_var.get()
            new_cfg["radio"]["discord"].update(
                {
                    "token": e_token.get().strip(),
                    "channel_id": channel_id,
                }
            )
            new_cfg["radio"]["native"].update(
                {
                    "input_device": input_device_map.get(input_device_var.get()),
                    "output_device": output_device_map.get(output_device_var.get()),
                }
            )
            try:
                save_bot_config_file(new_cfg)
                if str(self.ip_entry.cget("state")) == "normal":
                    self.ip_entry.delete(0, tk.END)
                    self.ip_entry.insert(0, ps_ip)
                radio_settings_changed = new_cfg["radio"] != config["radio"]
                self.config = new_cfg
                if radio_settings_changed and self.bot_running:
                    self.stop_bot()
                close_settings()
            except OSError as e:
                config_error_var.set(f"Could not save configuration: {e}")

        ttk.Button(button_frame, text="Cancel", command=close_settings).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(button_frame, text="Save", command=save).pack(side=tk.LEFT)

        win.protocol("WM_DELETE_WINDOW", close_settings)
        win.bind("<Escape>", lambda _event: close_settings())
        win.bind("<Control-s>", lambda _event: save())
        if initial_tab == "radio":
            notebook.select(radio_tab)
            if mode_var.get() == "native":
                input_device_combo.focus_set()
            else:
                e_token.focus_set()
        else:
            e_ps_ip.focus_set()

    def open_bot_config(self):
        self.open_settings(initial_tab="radio")

    def set_radio_controls_running(self, mode):
        mode_label = "Discord" if mode == "discord" else "Native"
        self.lbl_bot_status.config(text=f"● Radio: {mode_label}", fg='#00ff00')
        self.btn_bot_toggle.config(
            text="Disable Radio",
            bg='#ff4444',
            fg='white',
            activebackground='#cc0000',
            state='normal',
        )

    def set_radio_controls_stopped(self, error=None):
        self.bot_running = False
        if error:
            self.lbl_bot_status.config(text="● Radio: ERROR", fg='#ff4444')
        else:
            self.lbl_bot_status.config(text="● Radio: OFF", fg='gray')
        self.btn_bot_toggle.config(
            text="Start Radio",
            bg='#007bff',
            fg='white',
            activebackground='#0056b3',
            state='normal',
        )

    def toggle_bot(self):
        if self.bot_running:
            self.stop_bot()
            return

        self.config = load_bot_config_file()
        radio_config = self.config["radio"]
        mode = radio_config["mode"]
        token = str(radio_config["discord"].get("token") or "").strip()

        if mode == "discord" and not token:
            self.open_settings(initial_tab="radio")
            return
        if mode == "native" and sd is None:
            self.open_settings(initial_tab="radio")
            return

        mode_label = "Discord" if mode == "discord" else "Native"
        set_radio_ui_state("STARTING", f"Starting {mode_label} radio", "")
        self.bot_running = True
        self.radio_stop_requested = False
        self.radio_worker_finished = False
        self.radio_worker_error = None
        self.set_radio_controls_running(mode)
        self.bot_thread = threading.Thread(
            target=self.run_radio_process,
            args=(mode, token, self.config),
            daemon=True,
        )
        self.bot_thread.start()

    def stop_bot(self):
        self.radio_stop_requested = True
        bot = self.bot
        loop = self.bot_loop
        if bot is not None and loop is not None and loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(bot.close(), loop)
            except RuntimeError as e:
                print(f">>> [Radio] Stop request warning: {e}")

        set_radio_ui_state("STOPPING", "Stopping radio", "")
        self.lbl_bot_status.config(text="● Radio: STOPPING", fg='orange')
        self.btn_bot_toggle.config(
            text="Stopping...",
            bg='#555555',
            fg='white',
            activebackground='#555555',
            state='disabled',
        )

    def run_radio_process(self, mode, token, config):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.bot_loop = loop
        bot = None

        async def run_native_radio():
            async with bot:
                if self.radio_stop_requested:
                    return
                await bot.start_native()
                if self.radio_stop_requested:
                    await bot.close()
                    return
                await bot.wait_for_native_stop()

        try:
            bot = EngineerBot(mode=mode, config=config)
            self.bot = bot
            if self.radio_stop_requested:
                loop.run_until_complete(bot.close())
            elif mode == "native":
                loop.run_until_complete(run_native_radio())
            else:
                loop.run_until_complete(bot.start(token))
        except Exception as e:
            self.radio_worker_error = f"{type(e).__name__}: {e}"
            print(f">>> [Radio] {self.radio_worker_error}")
            set_radio_ui_state("ERROR", self.radio_worker_error)
        finally:
            if bot is not None and not bot.is_closed():
                try:
                    loop.run_until_complete(bot.close())
                except Exception as e:
                    print(f">>> [Radio] Close warning: {e}")

            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )

            if not self.radio_worker_error:
                set_radio_ui_state("OFF", "Radio off", "")
            if self.bot is bot:
                self.bot = None
            self.bot_loop = None
            self.radio_worker_finished = True
            loop.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = RaceDashboard(root)
    root.mainloop()
