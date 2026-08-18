# SafeFall AI — Elderly Fall Detection System (FA-2)

Working prototype for CareVision HealthTech's SafeFall AI monitoring system,
built with MediaPipe Pose + a Dense Neural Network classifier, deployed as
a Streamlit dashboard.

## How this satisfies the FA-2 steps

| Rubric Step | Where it happens |
|---|---|
| Step 4: Model selection | MediaPipe Pose (pose estimation) + Dense NN + Random Forest baseline (classification) — see `3_train_model.py` |
| Step 5: Model design & training | `1_extract_frames_and_labels.py` → `2_extract_pose_landmarks.py` → `3_train_model.py` |
| Step 6: Evaluation | `4_evaluate_model.py` — accuracy, precision, recall, F1, confusion matrix |
| Step 7: Deployment | `app.py` — Streamlit dashboard, deployed to Streamlit Cloud |
| Step 8: Monitoring & maintenance | See "Future Improvements" section below |

### Important design note (mention this in your project video)
The Le2i/IMVIA annotation files only give ground-truth **timing of the fall
itself** (start/end frame). They don't provide per-frame labels for
walking/sitting/standing. So the system is a **hybrid**:
- A **real trained deep learning model** classifies Fall vs Not-Fall using
  the actual annotated ground truth — this is what your accuracy/precision/
  recall/F1/confusion matrix are measuring.
- A **geometry-based rule layer** (joint angles computed from the same pose
  landmarks) splits "Not-Fall" frames into Walking / Sitting / Standing /
  Normal Activity, satisfying the 5-class requirement.

This is a standard, defensible pattern (ML + rule-based hybrid) — call it
out explicitly rather than letting it look like all 5 classes were
independently trained, since that's more honest and actually shows deeper
understanding of the dataset's limitations (which the rubric rewards under
"deployment challenges").

## Files

```
fall_detection_project/
├── requirements.txt
├── utils.py                          # shared pose/feature/heuristic functions
├── 1_extract_frames_and_labels.py    # video -> labelled frames
├── 2_extract_pose_landmarks.py       # frames -> pose landmark feature table
├── 3_train_model.py                  # trains Dense NN + Random Forest
├── 4_evaluate_model.py               # metrics + confusion matrix
├── app.py                            # Streamlit dashboard
└── README.md                         # this file
```

## Step-by-step: run in Google Colab

**1. Get the dataset into Colab**

```python
# in a Colab cell
!pip install kaggle
from google.colab import files
files.upload()  # upload your kaggle.json API token (from kaggle.com/settings)
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d tuyenldvn/falldataset-imvia -p /content
!unzip -q /content/falldataset-imvia.zip -d /content/falldataset-imvia
```

**2. Upload the project files**

Upload all the `.py` files above into your Colab session (or `git clone` if
you push this to a GitHub repo first — recommended, since you'll need a
repo for Streamlit Cloud anyway).

```python
!pip install -r requirements.txt
```

**3. Extract frames + labels** (Step 4/5 groundwork)

```python
!python 1_extract_frames_and_labels.py \
    --dataset_root /content/falldataset-imvia \
    --out_dir /content/frames \
    --sample_every 5
```
Check the printed fall/not_fall counts — if wildly imbalanced, lower
`--sample_every` to grab more frames, or oversample the fall class later.

**4. Extract pose landmarks**

```python
!python 2_extract_pose_landmarks.py \
    --labels_csv /content/frames/labels.csv \
    --out_csv /content/landmarks.csv
```

**5. Train the model** (Step 5)

```python
!python 3_train_model.py \
    --landmarks_csv /content/landmarks.csv \
    --out_dir /content/model \
    --epochs 60
```
This prints train/val/test sizes, saves `training_history.png` (accuracy +
loss curves — required evidence), and saves the trained model + scaler.

**6. Evaluate** (Step 6)

```python
!python 4_evaluate_model.py --model_dir /content/model
```
This prints accuracy/precision/recall/F1 and saves
`confusion_matrix_nn.png`. **Screenshot this output and the confusion
matrix image** — this is required evidence for your submission.

**7. Sanity-check the dashboard locally in Colab (optional)**

```python
!pip install pyngrok
!streamlit run app.py &
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(public_url)
```
(You'll need a free ngrok auth token — `ngrok.set_auth_token("...")`.)

## Deploying to Streamlit Cloud (Step 7 — this is what's actually required)

1. Create a GitHub repo, push everything in this folder **plus** the
   `model/` folder contents from Colab (`model_nn.keras`, `scaler.joblib`,
   `label_encoder.joblib`, `training_history.png`,
   `confusion_matrix_nn.png`, `nn_classification_report.txt`) into a
   `model/` subfolder in the repo.
   - Keras model files are usually small enough for GitHub; if not, use
     Git LFS.
2. Go to [share.streamlit.io](https://streamlit.io/cloud), sign in with
   GitHub, click "New app", point it at your repo and `app.py`.
3. Streamlit Cloud installs `requirements.txt` automatically. Deploy.
4. Grab the live `.streamlit.app` URL — this is your submission link.

## What to screen-record for your video (matches the evidence checklist)

1. Quick project overview (30 sec) — what SafeFall AI does.
2. Dataset explanation — show the Le2i folder structure, mention scenes.
3. Run through `1_extract...` and `2_extract...` output (or show saved
   frame examples + landmark CSV).
4. Show `training_history.png` and explain the NN architecture briefly.
5. Show the evaluation output — accuracy/precision/recall/F1 +
   `confusion_matrix_nn.png`.
6. Open the deployed Streamlit app, upload a fall-video clip, show the
   alert firing, then upload a normal-activity clip, show it correctly
   classified, and show the analytics panel / activity distribution chart.

## Future Improvements (Step 8 — write this up in your report)

- **More labelled data**: request or manually annotate a subset with true
  per-frame Walking/Sitting/Standing labels to replace the rule-based
  sub-classifier with a trained one.
- **Better pose robustness**: MediaPipe struggles with heavy occlusion
  (furniture) or extreme camera angles common in the Office/Lecture_room
  scenes — a higher-resolution pose model (YOLOv8-Pose) could help.
- **Temporal modelling**: a CNN+LSTM or a sliding-window approach over
  several consecutive frames (instead of single-frame classification) would
  catch the *motion* of falling, not just the end pose, and should reduce
  false alarms from someone sitting down quickly.
- **False alert reduction**: track a short buffer of frames and only alert
  if several consecutive frames agree it's a fall, rather than a single
  frame.
- **Low-light handling**: augment training data with brightness/contrast
  jitter (as planned in FA-1 preprocessing) — currently only geometric
  augmentation is implicitly handled through the normalisation step.
- **Real-time CCTV feeds**: swap the Streamlit file-upload input for an
  RTSP/webcam stream reader (`cv2.VideoCapture(rtsp_url)`) run frame-by-frame
  through the same `predict_frame()` function already in `app.py`.
- **Periodic retraining**: re-run steps 1–2 whenever new annotated videos
  are added, and retrain on the combined dataset on a scheduled basis.
