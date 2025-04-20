import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import av

# Initialize MediaPipe and model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

@st.cache_resource
def load_fight_model():
    try:
        return load_model("C:\\Inspector\\LSTM-Actions-Recognition\\lstm-fight-detection.h5")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_fight_model()

# Global variables
label = "unknown"
label_history = []
lm_list = []

# WebRTC configuration (use default STUN server)
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class ViolenceProcessor(VideoProcessorBase):
    def __init__(self):
        self.label = "unknown"
        self.label_history = []
        self.lm_list = []

    def make_landmark_timestep(self, results):
        if results.pose_landmarks:
            c_lm = []
            for lm in results.pose_landmarks.landmark:
                c_lm.extend([lm.x, lm.y, lm.z])
            return c_lm
        return None

    def draw_landmark_on_image(self, mp_draw, results, frame):
        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
        return frame

    def detect(self, model, lm_list):
        if len(lm_list) >= 20:
            lm_array = np.array(lm_list[-20:])
            lm_array = np.expand_dims(lm_array, axis=0)
            try:
                result = model.predict(lm_array, verbose=0)[0]
                pred_label = "fight" if result[0] > 0.5 else "normal"
                # Heuristic: Rapid movement in wrists (15, 16) and elbows (13, 14)
                if len(lm_list) > 1 and lm_list[-1] and lm_list[-2]:
                    for idx in [15, 16, 13, 14]:
                        x_diff = lm_list[-1][idx*3] - lm_list[-2][idx*3]
                        y_diff = lm_list[-1][idx*3+1] - lm_list[-2][idx*3+1]
                        if abs(x_diff) > 0.07 or abs(y_diff) > 0.07:
                            pred_label = "fight"
                            break
                # Smoothing
                self.label_history.append(pred_label)
                if len(self.label_history) > 3:
                    self.label_history.pop(0)
                self.label = max(set(self.label_history), key=self.label_history.count) if self.label_history else pred_label
            except Exception as e:
                st.error(f"Prediction error: {e}")
                self.label = "ERROR"
        return self.label

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        lm = self.make_landmark_timestep(results)
        if lm:
            self.lm_list.append(lm)
        img = self.draw_landmark_on_image(mp_draw, results, img)
        self.label = self.detect(model, self.lm_list)
        color = (0, 0, 255) if self.label == "fight" else (0, 255, 0)
        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color, 2)
        cv2.putText(img, self.label.upper(), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("FightSense AI - Live Violence Detection")
st.markdown("""
    Detect single-person violence in real-time using your webcam with MediaPipe Pose and LSTM.
    Labels: **FIGHT** (red) for violent actions, **NORMAL** (green) for non-violent.
""")

# Webcam stream
ctx = webrtc_streamer(
    key="violence-detection",
    video_processor_factory=ViolenceProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# Display prediction
if ctx.video_processor:
    st.write(f"**Prediction**: {ctx.video_processor.label.upper()}")

# Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Cleanup
if not ctx.state.playing:
    pose.close()