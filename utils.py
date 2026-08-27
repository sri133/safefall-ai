import base64
import io
import struct
import wave
 
import numpy as np
import cv2
import mediapipe as mp
 
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
 

LM = mp_pose.PoseLandmark
 
FEATURE_LANDMARKS = [
    LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
    LM.LEFT_ELBOW, LM.RIGHT_ELBOW,
    LM.LEFT_WRIST, LM.RIGHT_WRIST,
    LM.LEFT_HIP, LM.RIGHT_HIP,
    LM.LEFT_KNEE, LM.RIGHT_KNEE,
    LM.LEFT_ANKLE, LM.RIGHT_ANKLE,
    LM.NOSE,
]
 
FEATURE_COLUMNS = []
for lm in FEATURE_LANDMARKS:
    name = lm.name.lower()
    FEATURE_COLUMNS += [f"{name}_x", f"{name}_y", f"{name}_vis"]
 
 
def get_pose_landmarks(image_bgr, pose_estimator):
    """
    Run MediaPipe Pose on a single BGR image.
    Returns the raw mediapipe results object (or None if no person detected).
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = pose_estimator.process(image_rgb)
    if results.pose_landmarks is None:
        return None
    return results
 
 
def landmarks_to_dict(results):
    """Convert mediapipe results into {landmark_name: (x, y, z, visibility)}."""
    lm_list = results.pose_landmarks.landmark
    out = {}
    for lm in LM:
        p = lm_list[lm.value]
        out[lm.name.lower()] = (p.x, p.y, p.z, p.visibility)
    return out
 
 
def extract_feature_vector(landmark_dict):
    """
    Build a scale- and position-invariant feature vector from raw landmarks.
 
    We normalise every point relative to the torso: subtract the hip
    midpoint (removes position dependence) and divide by torso length
    (removes scale/distance-from-camera dependence). This is the standard
    trick for pose-based activity recognition.
 
    Returns a 1D numpy array of length len(FEATURE_COLUMNS), or None if the
    torso landmarks themselves are missing/low-confidence.
    """
    try:
        lh = np.array(landmark_dict["left_hip"][:2])
        rh = np.array(landmark_dict["right_hip"][:2])
        ls = np.array(landmark_dict["left_shoulder"][:2])
        rs = np.array(landmark_dict["right_shoulder"][:2])
    except KeyError:
        return None
 
    mid_hip = (lh + rh) / 2.0
    mid_shoulder = (ls + rs) / 2.0
    torso_size = np.linalg.norm(mid_shoulder - mid_hip)
    if torso_size < 1e-6:
        return None
 
    feats = []
    for lm in FEATURE_LANDMARKS:
        name = lm.name.lower()
        x, y, z, vis = landmark_dict[name]
        norm_x = (x - mid_hip[0]) / torso_size
        norm_y = (y - mid_hip[1]) / torso_size
        feats += [norm_x, norm_y, vis]
    return np.array(feats, dtype=np.float32)
 
 
def _angle(a, b, c):
    """Angle (degrees) at point b, formed by points a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return 180.0
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))
 
 
def compute_heuristic_features(landmark_dict):
    """
    Extra geometric features used ONLY for the rule-based
    Walking/Sitting/Standing/Normal sub-classifier (applied to frames the
    trained model already said are NOT a fall).
    """
    lh = np.array(landmark_dict["left_hip"][:2])
    rh = np.array(landmark_dict["right_hip"][:2])
    ls = np.array(landmark_dict["left_shoulder"][:2])
    rs = np.array(landmark_dict["right_shoulder"][:2])
    lk = np.array(landmark_dict["left_knee"][:2])
    rk = np.array(landmark_dict["right_knee"][:2])
    la = np.array(landmark_dict["left_ankle"][:2])
    ra = np.array(landmark_dict["right_ankle"][:2])
 
    mid_hip = (lh + rh) / 2.0
    mid_shoulder = (ls + rs) / 2.0
    torso_size = np.linalg.norm(mid_shoulder - mid_hip) + 1e-6
 
    
    torso_vec = mid_shoulder - mid_hip
    vertical = np.array([0.0, -1.0])  
    cosang = np.clip(
        np.dot(torso_vec, vertical) / (np.linalg.norm(torso_vec) + 1e-6), -1, 1
    )
    torso_angle = float(np.degrees(np.arccos(cosang)))
 
    knee_angle = (_angle(lh, lk, la) + _angle(rh, rk, ra)) / 2.0
 
    
    hip_ankle_vert = ((la[1] - lh[1]) + (ra[1] - rh[1])) / 2.0 / torso_size
 
    return {
        "torso_angle": torso_angle,
        "knee_angle": knee_angle,
        "hip_ankle_vert": hip_ankle_vert,
    }
 
 
def compute_radar_features(landmark_dict):
    """
    A richer set of joint angles used for the live radar/spider chart in
    the dashboard -- gives a visual "body posture fingerprint" for the
    current prediction, on top of the smaller feature set used for the
    walking/sitting/standing heuristic.
    """
    ls = np.array(landmark_dict["left_shoulder"][:2])
    rs = np.array(landmark_dict["right_shoulder"][:2])
    le = np.array(landmark_dict["left_elbow"][:2])
    re = np.array(landmark_dict["right_elbow"][:2])
    lw = np.array(landmark_dict["left_wrist"][:2])
    rw = np.array(landmark_dict["right_wrist"][:2])
    lh = np.array(landmark_dict["left_hip"][:2])
    rh = np.array(landmark_dict["right_hip"][:2])
    lk = np.array(landmark_dict["left_knee"][:2])
    rk = np.array(landmark_dict["right_knee"][:2])
    la = np.array(landmark_dict["left_ankle"][:2])
    ra = np.array(landmark_dict["right_ankle"][:2])
 
    h = compute_heuristic_features(landmark_dict)
 
    elbow_angle = (_angle(ls, le, lw) + _angle(rs, re, rw)) / 2.0
    hip_angle = (_angle(ls, lh, lk) + _angle(rs, rh, rk)) / 2.0
    shoulder_angle = (_angle(le, ls, lh) + _angle(re, rs, rh)) / 2.0
 
    return {
        "Torso Tilt": round(h["torso_angle"], 1),
        "Knee Angle": round(h["knee_angle"], 1),
        "Hip Angle": round(hip_angle, 1),
        "Elbow Angle": round(elbow_angle, 1),
        "Shoulder Angle": round(shoulder_angle, 1),
    }
 
 
def is_lying_down(landmark_dict, angle_threshold=55):
    """
    Geometric safety-net check: a near-horizontal torso strongly indicates
    the person is down on the ground (either mid-fall or lying there
    afterward).
 
    Why this exists: the Le2i annotation only marks the fall EVENT itself
    (the transition), not how long the person remains down afterward. So
    the trained classifier sometimes mislabels "still lying on the ground"
    frames as Standing/Normal, since those frames were trained as
    not_fall. This geometric check catches that failure mode directly from
    body orientation, independent of what the trained model predicts.
    """
    h = compute_heuristic_features(landmark_dict)
    return h["torso_angle"] > angle_threshold
 
 
def sub_classify_activity(landmark_dict, ankle_motion=None):
    """
    Rule-based classifier applied only to frames predicted as NOT a fall.
 
    ankle_motion: optional float, normalised frame-to-frame ankle
    displacement (only computable in video mode where a previous frame
    exists). If None, walking cannot be distinguished from standing and we
    fall back to "Standing".
    """
    h = compute_heuristic_features(landmark_dict)
 
    
    if h["knee_angle"] < 130 and h["hip_ankle_vert"] < 1.3:
        return "Sitting"
 
    
    if ankle_motion is not None and ankle_motion > 0.15:
        return "Walking"
 
    
    if h["knee_angle"] >= 150 and h["torso_angle"] < 30:
        return "Standing"
 
    return "Normal Activity"
 
 
def draw_pose_and_label(image_bgr, results, label, confidence, alert=False):
    """Draw the skeleton + a label banner on the frame for display."""
    annotated = image_bgr.copy()
    mp_drawing.draw_landmarks(
        annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
    )
    text = f"{label} ({confidence*100:.1f}%)"
    color = (0, 0, 255) if alert else (0, 200, 0)
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), color, -1)
    cv2.putText(
        annotated, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (255, 255, 255), 2, cv2.LINE_AA
    )
    return annotated
 
 
def ankle_midpoint(landmark_dict):
    la = np.array(landmark_dict["left_ankle"][:2])
    ra = np.array(landmark_dict["right_ankle"][:2])
    return (la + ra) / 2.0
 
 
def is_fall_alert(label, confidence, threshold=0.6):
    """Alert logic: fire only when model is reasonably confident."""
    return label == "Fall Detected" and confidence >= threshold
 
 
def generate_beep_base64(duration=0.35, freq=880, sample_rate=22050, volume=0.5):
    """
    Generate a short beep tone as a base64-encoded WAV, entirely in memory
    -- no external audio file needed. Used for the audible fall alert.
    A short fade-in/fade-out envelope avoids a harsh click at the edges.
    """
    n_samples = int(sample_rate * duration)
    fade_samples = max(1, int(sample_rate * 0.01))
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            t = i / sample_rate
            envelope = min(1.0, i / fade_samples, (n_samples - i) / fade_samples)
            sample = volume * envelope * np.sin(2 * np.pi * freq * t)
            frames += struct.pack("<h", int(sample * 32767))
        wf.writeframes(bytes(frames))
    return base64.b64encode(buf.getvalue()).decode("utf-8")
 
