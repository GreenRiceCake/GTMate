# GTMate

GTMate는 Gran Turismo 7을 위한 실시간 텔레메트리 대시보드이자 디스코드 기반 팀 라디오 엔지니어입니다. GT7 UDP 데이터를 받아 속도, RPM, 기어, 연료, 랩타임, 타이어 상태를 표시하고, 음성 명령으로 레이스 중 필요한 정보를 들을 수 있게 돕습니다.

현재 프로젝트는 개발 중이며, 기능과 UI는 계속 바뀔 수 있습니다.

## 주요 기능

### 실시간 대시보드

- 속도, RPM, 기어, 스로틀, 브레이크, 클러치 입력 표시
- 타이어 온도, 연료 잔량, 현재 랩, 베스트 랩, 이전 랩 표시
- Packet C 기반 현재 랩타임 표시
- F11 전체 화면 토글 및 창 크기에 맞춘 UI 스케일링
- PlayStation IP 자동 검색 및 연결 지원

### 팀 라디오 엔지니어

- Discord 음성 채널에 봇이 접속해 음성 명령을 인식
- Vosk 기반 음성 인식과 Piper 기반 음성 응답 사용
- 연료, 순위, 현재 랩, 베스트 랩, 이전 랩 정보를 음성으로 조회
- 베스트랩 갱신 시 새 기록과 이전 기록 대비 차이를 안내
- 저연료 상황에서 남은 랩 수를 자동 안내
- 라디오 상태 표시로 대기, 호출, 명령 인식, 응답 상태 확인 가능

### 피트 및 연료 전략

- 피트 엔트리, 피트 레인, 핏워크, 피트 아웃 상태 감지
- 피트 시퀀스 UI 표시
- 피트/스탠바이 상태에서 연료 예측 데이터 오염 방지
- 랩당 평균 연료 사용량을 바탕으로 남은 랩 수 계산
- 계산 데이터가 부족한 경우 현재 연료만 안내

### 업데이트

- v1.1.0부터 zip 기반 전체 패키지 업데이트를 지원합니다.
- 업데이트 진행률, 압축 해제, 파일 교체 로그를 볼 수 있습니다.
- v1.0.x 사용자는 전환 업데이트를 거쳐 새 업데이터로 이동합니다.

## 음성 명령

먼저 호출어를 말한 뒤 명령을 말합니다.

호출어:

```text
engineer, mate, chief, radio, hello, hey
```

연료:

```text
fuel, gas, petrol, consumption, tank
```

순위:

```text
rank, position, place, where am i
```

현재 랩:

```text
current lap, lap, current
```

베스트 랩:

```text
best, fastest, record, lap time
```

이전 랩:

```text
last, previous
```

취소 / 종료:

```text
no, nope, negative, cancel, nothing, done, thanks, thank you
```

## 설치 및 사용

1. GitHub Releases에서 최신 GTMate 설치 파일을 다운로드합니다.
2. 설치 후 GTMate를 실행합니다.
3. PlayStation과 PC가 같은 네트워크에 있는지 확인합니다.
4. `Find PS` 또는 직접 IP 입력으로 GT7 UDP 데이터에 연결합니다.
5. `Bot Config`에서 Discord bot token과 voice channel ID를 설정합니다.
6. `Start Radio`를 눌러 팀 라디오를 시작합니다.

주의: `bot_config.json`에는 Discord bot token이 저장됩니다. 이 파일을 공개 저장소나 다른 사람에게 공유하지 마세요.

## v1.1.0 주요 변경점

- Discord 음성 정책 변경에 대응하기 위해 DAVE 기반 음성 수신 구조 적용
- `davey`와 `discord-ext-voice-recv` 조합으로 팀 라디오 음성 인식 복구
- GT7 Packet C 수신 지원
- 현재 랩타임 표시 추가
- 피트 상태 감지 및 피트 시퀀스 UI 추가
- 연료 예측 및 저연료 안내 개선
- 베스트랩 갱신 음성 안내 추가
- PlayStation IP 자동 검색 추가
- zip 기반 새 업데이터 추가
- 전체 화면 토글 및 UI 스케일링 추가

## 개발 메모

`bin`과 `models`는 외부 리소스로 분리되어 있습니다. PyInstaller 빌드 결과물과 함께 배포할 때는 다음 항목들이 같은 설치 폴더에 있어야 합니다.

```text
GTMate.exe
Updater.exe
curr_ver.json
_internal/
bin/
models/
```

업데이트 매니페스트는 레포지토리의 `update_manifest.json`에서 관리합니다.

## 참고한 자료

- [MacManley/gt7-udp](https://github.com/MacManley/gt7-udp): GT7 UDP 패킷 구조 분석에 참고했습니다.
- [Nenkai/PDTools](https://github.com/Nenkai/PDTools): Gran Turismo 계열 데이터 구조와 분석 흐름을 이해하는 데 참고했습니다.
