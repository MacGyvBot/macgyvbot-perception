# 🤝 기여 가이드

이 레포지토리는 MacGyvBot의 “음성 명령 기반 공구 서랍 관리 로봇팔 어시스턴트” 프로젝트 규칙을 따릅니다.

## 🌿 브랜치 네이밍

```text
feature/#12-tool-detection-node
bugfix/#18-camera-index-error
docs/#5-update-requirements
experiment/#22-yolo-training
safety/#31-limit-handover-speed
task/#40-update-launch-config
```

권장 prefix:

- `feature/#`: 신규 기능
- `bugfix/#`: 버그 수정
- `docs/#`: 문서 수정
- `experiment/#`: 실험 코드 또는 실험 기록
- `safety/#`: 안전 관련 수정
- `refactor/#`: 리팩토링
- `task/#`: 설정, 빌드, 기타 일반 작업

## ✍️ 커밋 메시지

```text
type: 변경 내용 요약
```

type:

- `feat`: 신규 기능 추가
- `fix`: 버그 수정
- `docs`: 문서 수정
- `refactor`: 기능 변화 없는 구조 개선
- `test`: 테스트 추가 또는 수정
- `task`: 설정, 빌드, 기타 일반 작업
- `experiment`: 모델 학습, 로봇 제어, 시뮬레이션 실험
- `safety`: 안전 제한, 비상정지, 전달 동작 관련 수정

## 🔀 PR 규칙

- PR은 가능한 한 하나의 목적만 포함합니다.
- 관련 이슈를 PR 본문에 연결합니다.
- 테스트 결과를 PR 템플릿에 기록합니다.
- 로봇 제어, 그리퍼 동작, 서랍 개폐, 사용자 손 전달 변경은 안전 관점의 리뷰를 요청합니다.
- 대용량 모델 파일, 데이터셋, 영상, ROS bag 파일은 PR에 직접 포함하지 않습니다.

## 🤖 ROS 2 개발 규칙

- 노드 이름은 역할이 명확하게 드러나도록 작성합니다.
- topic, service, action 이름과 메시지 타입을 문서화합니다.
- 새 노드나 파라미터를 추가하면 launch 파일을 함께 업데이트합니다.
- 설정값은 YAML 파일로 분리합니다.
- 실제 로봇 실행 전에 dry-run 또는 시뮬레이션 실행을 권장합니다.

## 🚨 안전 규칙

- 로봇 작업 공간 제한을 설정합니다.
- 로봇팔과 그리퍼 속도 제한을 적용합니다.
- 비상정지 동작을 실제 실행 전에 확인합니다.
- 사용자 손과 로봇팔/공구 사이의 최소 거리를 유지합니다.
- 안전 관련 변경은 `safety` 라벨을 사용하고 우선 리뷰를 요청합니다.
