# MacGyvBot Perception

MacGyvBot의 perception 실험 레포지토리입니다.

현재 구현된 주요 실험은 **사람이 로봇 그리퍼가 들고 있는 공구를 잡았는지 판단하는 hand-tool grasp detection 프로토타입**입니다. MacBook/일반 RGB 카메라를 기본으로 사용하고, RealSense depth 카메라가 있을 경우 depth 기반 접촉 신호를 추가로 사용할 수 있습니다.

## 현재 실험

## Model Checkpoint
[Roboflow Tools dataset](https://universe.roboflow.com/drone-fcvjd/tools-bynck) (5,204 images, CC BY 4.0)으로 fine-tuning한 YOLO detection 모델 가중치입니다.
| Model | Base | Dataset | Epochs | Image size | Batch | Weights |
|---|---|---|---|---|---|---|
| YOLOv8m | `yolov8m.pt` (COCO pretrained) | Roboflow Tools v7 | 150 (patience 35) | 640 | 16 |[`ckeckpoint/yolo_v8/yolov8_best.pt`](https://drive.google.com/drive/folders/1RecvinWnF_LqmyD_syDmNfpkQvmM-d1H?usp=drive_link) |
| YOLO11m | `yolo11m.pt` (COCO pretrained) | Roboflow Tools v7 | 150 (patience 35) | 640 | 16 |[`ckeckpoint/yolo_v11/yolov11_best.pt`](https://drive.google.com/drive/folders/11N1dqIcpYFgg0XZ7PRkor_hm2kVurQ2N?usp=drive_link) |

학습에 사용한 전체 하이퍼파라미터는 각 모델 폴더의 `*_args.yaml`을 참조하세요.

### SAM (Segment Anything Model) Checkpoint

YOLO로 검출된 bbox를 마스크로 정밀화하는 데 사용하는 SAM 가중치입니다.  
용량·속도 트레이드오프에 따라 아래 세 가지 중 선택하세요.

#### MobileSAM (권장 — 경량)
| 파일명 | 크기 | 출처 | 다운로드 |
|---|---|---|---|
| `mobile_sam.pt` | ~39 MB | [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM) | [mobile_sam.pt](https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt) |

MobileSAM은 SAM과 동일한 인터페이스를 유지하면서 모델 크기를 대폭 줄인 경량 버전입니다.  
GPU 없이도 실시간 추론이 가능하며, 실험용 프로토타입에 적합합니다.

#### SAM — Meta/Facebook Research (고성능)
| Variant | 파일명 | 크기 | 다운로드 |
|---|---|---|---|
| `vit_b` (기본) | `sam_vit_b_01ec64.pth` | 358 MB | [다운로드](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) |
| `vit_l` | `sam_vit_l_0b3195.pth` | 1.2 GB | [다운로드](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) |
| `vit_h` | `sam_vit_h_4b8939.pth` | 2.4 GB | [다운로드](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |

- **vit_b**: 속도·정확도 균형이 좋은 기본 선택지입니다.  
- **vit_l / vit_h**: 마스크 품질이 더 높지만 메모리와 추론 시간이 증가합니다. GPU 환경에서 사용하세요.

다운로드한 가중치는 `checkpoint/sam/` 디렉토리에 저장하는 것을 권장합니다.


## 📁 기본 구조
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
