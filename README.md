# 🏎️ GTMate
**Your Personal AI Race Engineer & Telemetry Dashboard for Gran Turismo 7**

GTMate는 그란 투리스모 7 드라이버를 위한 올인원 서포트 시스템입니다. 실시간 데이터 시각화 대시보드와 음성 대화형 레이스 엔지니어를 통해 트랙 위에서 최상의 퍼포먼스를 낼 수 있도록 돕습니다.

GTMate는 현재 개발 중에 있습니다. 아래의 기능들은 추후 변경되거나 삭제될 수 있습니다.

# ✨ Key Features
### 🎙️ Interactive Voice Engineer (Under Development)
Hands-free Communication: 운전 중 핸들에서 손을 떼지 않고 음성으로 소통하세요.

Smart Response: "연료 얼마나 남았어?", "내 베스트 랩은?" 같은 질문에 실제 주행 데이터를 바탕으로 답변합니다.

Ultimate Convenience: 디스코드 봇을 통해 음성으로 편리하게 정보를 주고받을수 있습니다.

### 📊 Real-time Telemetry Dashboard
Full Data Visualization: 속도, RPM, 기어, 스로틀/브레이크 입력을 한눈에 확인합니다.

Tire & Fuel Monitoring: 타이어 온도와 연료 잔량을 실시간으로 추적하여 피트 전략을 세울 수 있습니다.

Race Logic: 베스트 랩과 이전 랩 타임을 정밀하게 기록합니다.

### 🔄 Seamless Experience
Auto-Update: 새로운 기능과 버그 수정이 자동으로 업데이트됩니다.

# 🏎️ Command List
엔지니어를 활용하기 위해서는 음성 명령어가 필요합니다.
"호출 명령어": "engineer", "mate", "chief", "radio", "hello", "hey" 등
"연료": "fuel", "gas", "petrol", "consumption", "tank" 등
"현재 순위": "rank", "position", "place", "where am i" 등
"현재 랩": "current lap", "lap", "current" 등
"베스트 랩": "best", "fastest", "record", "lap time" 등
"이전 랩": "last", "previous" 등
"거절 명령어": "no", "nope", "negative", "cancel", "nothing", "done", "thanks", "thank you" 등

# 🛠️ Tech Stack
Language: Python 3.x

GUI: Tkinter

Voice: Vosk (STT), Piper (TTS)

Telemetry: UDP Socket (Salsa20 Decryption)

Audio: FFmpeg, numpy, discord bot
