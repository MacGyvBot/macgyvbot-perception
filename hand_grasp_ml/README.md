# Hand Grasp ML

MediaPipe Hands와 ML 모델을 이용해 손 상태를 `open` / `grasp`로 분류하는 모듈이다. ToolMate 로봇팔 프로젝트에서 사용자가 공구를 잡았는지 판단하기 위한 확장 기능으로 사용된다.

초기 목표는 MacBook 또는 일반 웹캠에서 단독 실행 가능한 MediaPipe + RandomForest baseline이다. ROS2 연동은 baseline이 완성된 뒤 사용할 수 있는 별도 파일로 제공한다.

## 분류 상태

```text
no_hand       : 손이 감지되지 않음
open          : 손을 편 상태
grasp         : 실제 공구를 잡았거나 잡는 상태
unstable      : 최근 프레임 예측이 충분히 안정되지 않은 상태
```

`no_hand`는 ML 학습 라벨에 포함하지 않아도 된다. 손이 감지되지 않으면 추론 코드에서 직접 `no_hand`로 처리한다.

## 설치

```bash
pip install -r requirements.txt
```

환경에 따라 `python` 명령이 없으면 아래 실행 예시의 `python`을 `python3`로 바꿔 실행한다.

ROS2 노드는 ROS2 Humble 환경에서 실행한다. `rclpy`는 기본 requirements에 포함하지 않는다.

## 데이터 수집

```bash
python src/collect_dataset.py
```

키 입력:

```text
o: open 저장
g: grasp 저장
ESC: 종료
```

CSV는 header 없이 `data/hand_grasp_dataset.csv`에 저장된다.

```text
f0,f1,f2,...,f62,label
```

수집 가이드:

- `open`, `grasp` 각각 최소 200개 이상 수집한다.
- 실제 시연 환경과 비슷한 배경에서 수집한다.
- 실제 사용할 공구를 들고 `grasp` 데이터를 수집한다.
- 손 방향과 카메라 거리를 다양하게 바꿔 수집한다.
- 팀원 여러 명의 손 데이터를 포함하면 일반화 성능이 좋아진다.

## 모델 학습

```bash
python src/train_model.py
```

학습 스크립트는 `data/hand_grasp_dataset.csv`를 읽고 RandomForest 모델을 학습한 뒤 `models/hand_grasp_model.pkl`에 저장한다.

모델 설정:

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    class_weight="balanced",
)
```

## 실시간 추론

```bash
python src/realtime_inference.py
```

화면에는 raw prediction, confidence, stable prediction이 표시된다.

Stable filtering:

- 최근 10프레임을 `deque(maxlen=10)`에 저장한다.
- 가장 많이 나온 상태가 6프레임 이상이면 stable state로 인정한다.
- 그 외에는 `unstable`로 표시한다.
- 손이 감지되지 않으면 buffer를 비우고 `no_hand`로 처리한다.

## ROS2 실행

ROS2 Humble 환경에서만 실행한다.

```bash
python src/ros2_hand_state_node.py
```

노드 정보:

```text
node name: hand_grasp_state_node
topic: /hand_grasp_state
message: std_msgs/msg/String
values: no_hand, open, grasp, unstable
```

향후 로봇 제어 노드에서는 다음과 같은 조건으로 그리퍼 release를 판단할 수 있다.

```python
if hand_state == "grasp" for 0.5 seconds:
    open_gripper()
```

현재 baseline에서는 `grasp`를 release 후보로 사용한다.

## 테스트

```bash
python -m pytest tests
```

현재 테스트는 `extract_features()`의 feature 길이, float 변환, near-zero scale 처리를 검증한다.
