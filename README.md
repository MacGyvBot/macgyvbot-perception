# MacGyvBot Perception

MacGyvBot의 perception 실험 레포지토리입니다.

현재 구현된 주요 실험은 **사람이 로봇 그리퍼가 들고 있는 공구를 잡았는지 판단하는 hand-tool grasp detection 프로토타입**입니다. MacBook/일반 RGB 카메라를 기본으로 사용하고, RealSense depth 카메라가 있을 경우 depth 기반 접촉 신호를 추가로 사용할 수 있습니다.

## 현재 실험

```text
hand_grasp_detection/
├── README.md
├── requirements.txt
├── requirements-depth.txt
├── src/
│   ├── main.py
│   ├── hand_detector.py
│   ├── tool_detector.py
│   ├── grasp_detector.py
│   ├── depth_camera.py
│   └── utils.py
└── logs/
```

상세 설치 및 실행 방법은 [hand_grasp_detection/README.md](./hand_grasp_detection/README.md)를 참고하세요.

## 핵심 기능

- YOLO 커스텀 모델 기반 공구 bbox 검출
- MediaPipe Hands 기반 다중 손 landmark 검출
- 중복 손 검출 병합
- 공구 ROI와 손 landmark/bbox 접촉 점수 계산
- 선택적 RealSense depth 접촉 신호 반영
- 연속 프레임 기반 `human_grasped_tool` 판단
- CSV 로그 및 스크린샷 저장

## 기본 실행

```bash
cd hand_grasp_detection
pip install -r requirements.txt
python src/main.py
```

기본 YOLO 모델 파일은 `hand_grasp_detection/yolov11_best.pt`입니다.

## Safety Note

이 코드는 실제 로봇 release 신호를 직접 발생시키지 않는 실험용 프로토타입입니다. 로봇팔/그리퍼에 연결할 때는 상위 제어기에서 추가 safety gate를 둬야 합니다.
