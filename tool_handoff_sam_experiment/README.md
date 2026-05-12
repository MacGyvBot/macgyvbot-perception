# SAM tool handoff experiment

Webcam-only experiment for the delivery pipeline before ROS integration.

The app initializes a tool ROI with YOLO or a manual box, previews a SAM mask,
and locks the current mask only when `L` is pressed. After locking, hand-grasp
state is evaluated with the trained `.pkl` classifier and checked against the
locked mask contact.

Default local models:

- YOLO: `merge.pt` if present, otherwise `yolo_v11_merge.pt`
- grasp classifier: `hand_grasp_model.pkl`

The `.pkl` loader requires `joblib` and the scikit-learn runtime used by the
trained model. This model was saved with scikit-learn `1.7.2`. If the shared
venv does not have them:

```powershell
..\\hand_grasp_detection\\.venv\\Scripts\\python.exe -m pip install joblib scikit-learn
```

## Run

From the repository root:

```powershell
cd tool_handoff_sam_experiment
..\\hand_grasp_detection\\.venv\\Scripts\\python.exe app.py --sam-enabled
```

The default grasp decision mode is ML-dominant. Stable `.pkl` output `grasp`
drives the decision, but the hand still must be close to or touching the locked
tool mask/ROI. Mask contact is logged as stronger supporting evidence. To
restore the stricter old behavior:

ML `grasp` is always required. Both raw and stable model states must be
`grasp`. If either one says `open`, mask contact or the old heuristic score
cannot produce `human_grasped_tool=True`.

```powershell
..\\hand_grasp_detection\\.venv\\Scripts\\python.exe app.py --sam-enabled --ml-grasp-mode require-contact
```

The default SAM backend is MobileSAM:

```text
--sam-backend mobile_sam
--sam-checkpoint ..\\hand_grasp_detection\\models\\mobile_sam.pt
--sam-model-type vit_t
```

To use the original SAM ViT-B checkpoint:

```powershell
..\\hand_grasp_detection\\.venv\\Scripts\\python.exe app.py --sam-enabled --sam-backend sam --sam-checkpoint ..\\hand_grasp_detection\\models\\sam_vit_b_01ec64.pth --sam-model-type vit_b
```

If YOLO class names differ from the default:

```powershell
..\\hand_grasp_detection\\.venv\\Scripts\\python.exe app.py --sam-enabled --tool-classes drill,hammer,pliers,screwdriver,wrench,tape-measure
```

For the delivery pipeline test, prefer a single requested tool so unrelated
tools are ignored:

```powershell
..\\hand_grasp_detection\\.venv\\Scripts\\python.exe app.py --sam-enabled --requested-tool screwdriver
```

The display can show multiple YOLO detections. All detections are drawn as
boxes, and the active detection used for SAM is marked with `*`. If
`--requested-tool` is set, only that class is shown. To show several classes at
once, omit `--requested-tool` and use `--tool-classes`.

SAM masks are generated for detected candidates too. The default limit is five:

```powershell
..\\hand_grasp_detection\\.venv\\Scripts\\python.exe app.py --sam-enabled --max-sam-candidates 5
```

The active candidate is used for `L` lock and handoff judgment; the other
candidate masks are displayed for comparison.

## SAM precision controls

By default the experiment uses a refined SAM prompt:

- YOLO bbox as the main prompt
- bbox center as a foreground point
- bbox corners as background points
- mask candidate reranking to avoid masks that overfill the bbox
- clipping to a small bbox margin
- close/open morphology and largest-component cleanup

Useful tuning examples:

```powershell
..\\hand_grasp_detection\\.venv\\Scripts\\python.exe app.py --sam-enabled --requested-tool screwdriver --sam-clip-margin 3 --sam-max-bbox-fill 0.65
```

If the mask becomes too small or loses thin parts:

```powershell
..\\hand_grasp_detection\\.venv\\Scripts\\python.exe app.py --sam-enabled --requested-tool screwdriver --sam-clip-margin 16 --sam-open-kernel 1 --sam-max-bbox-fill 0.90
```

For a screwdriver where only the handle is selected, left-click once or twice
on the metal shaft before pressing `L`. Right-click nearby background if the
mask spills onto the desk or mouse pad.

To compare against plain bbox-only SAM prompting:

```powershell
..\\hand_grasp_detection\\.venv\\Scripts\\python.exe app.py --sam-enabled --requested-tool screwdriver --no-sam-point-prompts --no-sam-largest-component
```

To override model paths:

```powershell
..\\hand_grasp_detection\\.venv\\Scripts\\python.exe app.py --sam-enabled --yolo-model .\\merge.pt --grasp-model .\\hand_grasp_model.pkl
```

Manual ROI mode is useful when YOLO misses the tool:

```powershell
..\\hand_grasp_detection\\.venv\\Scripts\\python.exe app.py --detector-source manual --sam-enabled
```

## Keys

- `l`: lock the current SAM mask. This simulates robot grasp success.
- `m`: select a manual tool ROI.
- left mouse click: add a SAM foreground point.
- right mouse click: add a SAM background point.
- `p`: clear SAM prompt points.
- `v`: save VLM-ready images for the current candidate or locked mask.
- `s`: save a screenshot.
- `r`: reset grasp state, locked mask, and manual ROI.
- `c`: clear only the locked mask and grasp state.
- `q`: quit.

## Pipeline mapping

Before `L`, the app represents delivery step 2: tool candidate selection and
SAM segment preview. Pressing `L` represents the point after robot grasp success.
After `L`, the locked mask becomes the reference object for detecting whether
the user has grasped the tool.
