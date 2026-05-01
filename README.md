# 🧰 MacGyvBot Template

MacGyvBot의 `macgyvbot-*` 레포지토리를 만들 때 사용하는 기본 템플릿입니다.

이 템플릿은 “음성 명령 기반 공구 서랍 관리 로봇팔 어시스턴트” 프로젝트의 공통 협업 파일과 기본 디렉터리 구조를 제공합니다.

## 🚀 사용 방법

1. GitHub에서 이 레포지토리를 **Template repository**로 설정합니다.
2. 새 레포지토리를 만들 때 **Use this template**을 선택합니다.
3. 생성된 레포의 목적에 맞게 README, 패키지 구조, 설정 파일을 수정합니다.

## 📁 기본 구조

```text
.
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   ├── task.md
│   │   ├── experiment.md
│   │   ├── safety_issue.md
│   │   └── config.yml
│   └── PULL_REQUEST_TEMPLATE.md
├── CONTRIBUTING.md
├── README.md
└── .gitignore
```

## 📦 권장 레포지토리 이름

- `macgyvbot-assistant`
- `macgyvbot-perception`
- `macgyvbot-datasets`
- `macgyvbot-docs`
- `macgyvbot-vla`
- `macgyvbot-simulation`

## 🧩 공통 개발 범위

- 🤖 ROS 2 기반 로봇팔 제어
- 👁️ 공구 인식 및 위치 추정
- 🎙️ 음성 명령 처리
- 🧰 서랍 개폐 및 그리퍼 파지 제어
- 🚚 사용자 손 또는 전달 트레이 방향 안전 전달
- 🧪 데이터 수집, 라벨링, 모델 학습
- 🧠 YOLO, GroundingDINO, SAM, OpenVLA 실험
- 🕹️ 시뮬레이션 및 실제 로봇 테스트
- 🚨 안전 제어, 실패 복구, 로깅

## 🖥️ 기본 실행 환경

- OS: `Ubuntu 22.04`
- ROS 2: `ROS2 Humble`
- Python: `Python 3.10`
- 카메라: `Intel® RealSense™ Depth Camera D???`
- 로봇팔: `Doosan-Robotics-M0609`
- 그리퍼: `OnRobot RG2`

카메라 모델명은 실제 장비 확인 후 각 레포 README와 설정 파일에서 구체화합니다.

## 🤝 기여

브랜치, 커밋, PR, 이슈, 안전 규칙은 [CONTRIBUTING.md](./CONTRIBUTING.md)를 따릅니다.
