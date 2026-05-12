# Hand-Tool Grasp Detection

카메라 영상에서 공구와 사람 손을 인식하고, 사람이 공구를 잡았는지 `human_grasped_tool = True / False`로 판단하는 실험 코드입니다.

로봇팔 적용 전 판단 로직을 검증하기 위한 프로토타입이며, 실제 그리퍼 release 제어는 포함하지 않습니다. 기본은 RGB 카메라이고, RealSense depth 카메라가 있으면 depth 접촉 신호를 추가로 사용할 수 있습니다.

## 구성

```text
hand_grasp_detection/
├── README.md
├── requirements.txt
├── requirements-depth.txt
├── requirements-sam.txt
├── src/
│   ├── main.py
│   ├── hand_detector.py
│   ├── tool_detector.py
│   ├── tool_mask_tracker.py
│   ├── sam_segmenter.py
│   ├── grasp_detector.py
│   ├── depth_camera.py
│   └── utils.py
└── logs/
```

## 설치

Python 3.10 이상을 권장합니다.

```bash
pip install -r requirements.txt
```

RealSense depth 카메라를 사용할 경우:

```bash
pip install -r requirements-depth.txt
```

SAM 기반 초기 mask 생성을 사용할 경우:

```bash
pip install -r requirements-sam.txt
```

기본 SAM checkpoint 경로:

```text
models/sam_vit_b_01ec64.pth
```

checkpoint 파일은 용량이 크므로 저장소에 커밋하지 않습니다.

## 감지 방식

기본 감지 방식은 커스텀 YOLO best 모델입니다. YOLO 모드에서는 `--tool-classes`로 지정한 클래스만 공구 후보로 사용합니다.

```text
detector_source = yolo
```

기본 모델 파일:

```text
yolov11_best.pt
```

기본 감지 클래스:

```text
scissors
```

`tool_detector.py`는 `yolo11_best.pt`로 잘못 입력한 경우, 프로젝트 디렉토리에 `yolov11_best.pt`가 있으면 자동으로 대체합니다.

## 실행

기본 실행:

```bash
python src/main.py
```

SAM mask를 함께 사용:

```bash
python src/main.py --sam-enabled
```

기본 device는 CUDA입니다. CUDA가 없는 Mac에서 MPS를 쓸 경우:

```bash
python src/main.py --sam-enabled --sam-device mps --yolo-device mps
```

수동 ROI를 사용할 경우:

```bash
python src/main.py \
  --detector-source manual \
  --sam-enabled \
  --sam-device cuda
```

YOLO confidence threshold 조정:

```bash
python src/main.py \
  --detector-source yolo \
  --tool-classes scissors \
  --yolo-device cuda \
  --yolo-conf 0.15
```

YOLO 모드에서는 `--tool-classes`로 전달한 클래스만 공구 후보로 사용합니다. 빈 값은 허용하지 않습니다.

```bash
python src/main.py --detector-source yolo --tool-classes scissors,knife
```

CUDA가 없는 환경에서는 device를 명시합니다.

```bash
python src/main.py --yolo-device cpu --sam-device cpu
```

손 추적 개수 변경:

```bash
python src/main.py --max-hands 4
```

RealSense depth 사용:

```bash
python src/main.py --depth-source realsense
```

Depth 접촉 기준 조정:

```bash
python src/main.py \
  --depth-source realsense \
  --depth-diff-threshold-mm 40 \
  --depth-min-contact-landmarks 2
```

SAM으로 초기 공구 mask를 만들고, 이후 손에 가려져도 locked mask를 유지하려면:

```bash
python src/main.py --sam-enabled
```

다른 SAM checkpoint를 사용할 경우:

```bash
python src/main.py \
  --sam-enabled \
  --sam-checkpoint /path/to/custom_sam_checkpoint.pth \
  --sam-model-type vit_b \
  --sam-device cuda
```

SAM을 켜지 않아도 수동 ROI 또는 YOLO bbox를 rectangular mask로 lock하므로, occlusion 중에도 `locked mask` 기반 판단이 유지됩니다.

## 키보드 조작

| 키 | 동작 |
| --- | --- |
| `q` | 종료 |
| `m` | 현재 프레임에서 수동 ROI 선택 |
| `c` | 수동 ROI와 locked mask 초기화 |
| `r` | grasp counter 초기화 |
| `s` | 현재 프레임 스크린샷 저장 |

스크린샷은 `logs/screenshot_YYYYMMDD_HHMMSS.png`로 저장됩니다.

## 판단 흐름

1. 기본 모드에서는 YOLO로 공구 bbox를 검출합니다.
2. SAM이 켜져 있으면 수동 ROI 또는 YOLO bbox를 prompt로 공구 mask를 생성합니다.
3. `ToolMaskTracker`가 공구 mask 또는 bbox mask를 lock합니다.
4. 손에 가려져 detector/SAM이 잠깐 실패해도 locked mask를 유지합니다.
5. MediaPipe Hands로 손 landmark를 검출합니다.
6. 한 손이 여러 손으로 중복 검출되면 bbox IoU와 손바닥 중심 거리 기준으로 병합합니다.
7. 여러 손이 있으면 locked tool ROI에 가장 가까운 손을 `ACTIVE` 손으로 선택합니다.
8. 손 landmark와 공구 ROI/mask의 접촉/겹침 신호를 점수화합니다.
9. depth 카메라가 켜져 있으면 손 landmark depth와 공구 ROI depth 차이를 추가 점수로 반영합니다.
10. `GRASP_CANDIDATE`가 연속 `GRASP_HOLD_FRAMES` 이상 유지되면 `HUMAN_GRASPED_TOOL`로 전환합니다.

## 상태

| 상태 | 의미 |
| --- | --- |
| `NO_HAND` | 손이 검출되지 않음 |
| `TOOL_NOT_DETECTED` | 손은 있지만 공구 ROI/mask가 초기화되지 않음 |
| `HAND_DETECTED` | 손과 공구가 검출됐지만 접촉 신호가 약함 |
| `HAND_NEAR_TOOL` | 손이 공구 ROI 내부/경계 근처에 있음 |
| `GRASP_CANDIDATE` | grasp score가 기준 이상이며 연속 확인 중 |
| `HUMAN_GRASPED_TOOL` | 후보 상태가 기준 프레임 이상 유지됨 |

## 기본 파라미터

`src/grasp_detector.py` 기준:

```python
GRASP_HOLD_FRAMES = 15
PINCH_DISTANCE_THRESHOLD = 80
PALM_TO_TOOL_THRESHOLD = 80
ROI_MARGIN = 40
LANDMARK_TO_TOOL_THRESHOLD = 70
MIN_CONTACT_LANDMARKS = 2
HAND_TOOL_OVERLAP_THRESHOLD = 0.08
GRASP_SCORE_THRESHOLD = 3
DEPTH_GRASP_SCORE_BONUS = 2
MASK_GRASP_SCORE_BONUS = 2
```

`src/main.py` 기준:

```text
camera_index = 0
max_hands = 2
detector_source = yolo
yolo_model = yolov11_best.pt
yolo_device = cuda
tool_classes = scissors
yolo_conf = 0.20
yolo_imgsz = 640
depth_source = none
depth_diff_threshold_mm = 50
depth_min_contact_landmarks = 2
sam_enabled = false
sam_checkpoint = models/sam_vit_b_01ec64.pth
sam_device = cuda
mask_contact_radius = 6
mask_min_contact_landmarks = 2
mask_max_missing_frames = -1
mask_min_lock_area = 100
mask_smoothing_alpha = 0.7
```

## 로그

실행 시 `logs/grasp_log_YYYYMMDD_HHMMSS.csv`가 생성됩니다.

CSV 컬럼:

```text
timestamp,state,grasp_counter,pinch_distance,palm_to_tool_distance,min_landmark_to_tool_distance,contact_count,hand_tool_overlap_ratio,grasp_score,depth_available,tool_depth_mm,min_hand_tool_depth_diff_mm,depth_contact_count,depth_grasp_confirmed,mask_available,mask_contact_count,hand_mask_overlap_ratio,mask_grasp_confirmed,tool_mask_state,tool_mask_missing_frames,tool_mask_source,human_grasped_tool,active_hand_index,active_handedness,tool_source,tool_label,tool_confidence
```

## 주의

- 이 코드는 실험용 perception 판단 코드입니다.
- 실제 로봇 release 신호로 바로 사용하지 마세요.
- 로봇팔/그리퍼에 연결할 때는 로봇 정지 상태, force/torque, depth 안정성, 사용자 확인 등 별도의 safety gate가 필요합니다.
- 커스텀 YOLO 모델의 클래스명이 바뀌면 `--tool-classes`도 함께 맞춰야 합니다.
