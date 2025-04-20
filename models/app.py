# import streamlit as st
# import cv2
# import numpy as np
# from datetime import datetime, timedelta
# import pandas as pd
# import os
# import mediapipe as mp
# import face_recognition
# from tensorflow.keras.models import load_model
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
# import av
# import smtplib
# from email.mime.text import MIMEText
# import sys
# from ultralytics import YOLO
# from collections import Counter, deque
# from auth import login_user, register_user
# from utils import get_logged_user

# # Add the Gender_and_age_Detection folder to Python's path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'Gender_and_age_Detection'))

# # Initialize session state
# if "alerts" not in st.session_state:
#     st.session_state.alerts = []
# if "page" not in st.session_state:
#     st.session_state.page = "Login"

# # Set page config
# st.set_page_config(page_title="AI Surveillance System", layout="wide")

# def show_login_page():
#     st.title("Login")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("Login"):
#             token = login_user(username, password)
#             if token:
#                 st.session_state["token"] = token
#                 st.session_state.page = "Dashboard"
#                 st.success("Logged in successfully!")
#                 st.rerun()
#             else:
#                 st.error("Invalid credentials")
#     with col2:
#         if st.button("Go to Register"):
#             st.session_state.page = "Register"
#             st.rerun()

# def show_register_page():
#     st.title("Register")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     if st.button("Register"):
#         if register_user(username, password):
#             st.success("Registered successfully! Please go to Login.")
#         else:
#             st.error("User already exists!")
#     if st.button("Back to Login"):
#         st.session_state.page = "Login"
#         st.rerun()

# # Surveillance Dashboard Functions
# # Importing face recognition functions
# from import_face_recognition import load_known_faces, recognize_faces_live
# known_encodings, known_names = load_known_faces()

# # Initialize MediaPipe Pose and Drawing utilities
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# mp_draw = mp.solutions.drawing_utils

# # Initialize MediaPipe Face Detection
# mp_face = mp.solutions.face_detection
# face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# # Load YOLOv8 model globally
# try:
#     yolo_model = YOLO("yolov8n.pt")  # Replace with yolov8.pt or another variant if needed
# except Exception as e:
#     st.error(f"Error loading YOLO model: {e}")
#     st.stop()

# # Load fight detection model
# def load_fight_model():
#     try:
#         return load_model(r'C:\Users\kriya\Desktop\face\lstm-fight-detection.h5')
#     except Exception as e:
#         st.error(f"Error loading fight model: {e}")
#         st.stop()

# fight_model = load_fight_model()

# # WebRTC configuration
# RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# # Violence Processor for Fight Detection
# class ViolenceProcessor(VideoProcessorBase):
#     def __init__(self):
#         self.label = "unknown"
#         self.label_history = []
#         self.lm_list = []
#         self.fight_start_time = None
#         self.alert_sent = False

#     def make_landmark_timestep(self, results):
#         if results.pose_landmarks:
#             c_lm = []
#             for lm in results.pose_landmarks.landmark:
#                 c_lm.extend([lm.x, lm.y, lm.z])
#             return c_lm
#         return None

#     def draw_landmark_on_image(self, mp_draw, results, frame):
#         if results.pose_landmarks:
#             mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#             for id, lm in enumerate(results.pose_landmarks.landmark):
#                 h, w, c = frame.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
#         return frame

#     def detect(self, model, lm_list):
#         if len(lm_list) >= 20:
#             lm_array = np.array(lm_list[-20:])
#             lm_array = np.expand_dims(lm_array, axis=0)
#             try:
#                 result = model.predict(lm_array, verbose=0)[0]
#                 pred_label = "fight" if result[0] > 0.5 else "normal"
#                 for idx in [15, 16, 13, 14]:
#                     x_diff = lm_list[-1][idx*3] - lm_list[-2][idx*3]
#                     y_diff = lm_list[-1][idx*3+1] - lm_list[-2][idx*3+1]
#                     if abs(x_diff) > 0.07 or abs(y_diff) > 0.07:
#                         pred_label = "fight"
#                         break
#                 self.label_history.append(pred_label)
#                 if len(self.label_history) > 3:
#                     self.label_history.pop(0)
#                 self.label = max(set(self.label_history), key=self.label_history.count)
#             except Exception as e:
#                 st.error(f"Prediction error: {e}")
#                 self.label = "ERROR"
#         return self.label

#     def send_alert_email(self):
#         try:
#             sender = "viraj.salunke23@spit.ac.in"
#             password = "lppqwukqzossvidg"
#             receiver = "virajsalunke12@gmail.com"
#             subject = "⚠️ Fight Detected Alert"
#             body = f"⚠️ A violent action was detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} in CCTV camera feed."

#             msg = MIMEText(body)
#             msg["Subject"] = subject
#             msg["From"] = sender
#             msg["To"] = receiver

#             with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
#                 server.login(sender, password)
#                 server.sendmail(sender, receiver, msg.as_string())
#             print("✅ Email sent")
#         except Exception as e:
#             print(f"❌ Email error: {e}")

#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = pose.process(frame_rgb)
#         face_results = face_detection.process(frame_rgb)

#         lm = self.make_landmark_timestep(results)
#         if lm:
#             self.lm_list.append(lm)

#         img = self.draw_landmark_on_image(mp_draw, results, img)

#         if face_results.detections:
#             for detection in face_results.detections:
#                 bbox = detection.location_data.relative_bounding_box
#                 h, w, _ = img.shape
#                 x1 = int(bbox.xmin * w)
#                 y1 = int(bbox.ymin * h)
#                 x2 = x1 + int(bbox.width * w)
#                 y2 = y1 + int(bbox.height * h)
#                 x1, y1 = max(0, x1), max(0, y1)
#                 x2, y2 = min(w, x2), min(h, y2)
#                 face_region = img[y1:y2, x1:x2]
#                 if face_region.size > 0:
#                     blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
#                     img[y1:y2, x1:x2] = blurred

#         self.label = self.detect(fight_model, self.lm_list)
#         color = (0, 0, 255) if self.label == "fight" else (0, 255, 0)
#         cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color, 2)
#         cv2.putText(img, self.label.upper(), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#         if self.label == "fight":
#             if self.fight_start_time is None:
#                 self.fight_start_time = datetime.now()
#             elif datetime.now() - self.fight_start_time > timedelta(seconds=5) and not self.alert_sent:
#                 self.send_alert_email()
#                 self.alert_sent = True
#         else:
#             self.fight_start_time = None
#             self.alert_sent = False

#         return av.VideoFrame.from_ndarray(img, format="bgr24")

# def process_video_with_webrtc():
#     ctx = webrtc_streamer(
#         key="violence-detection",
#         video_processor_factory=ViolenceProcessor,
#         rtc_configuration=RTC_CONFIGURATION,
#         media_stream_constraints={"video": True, "audio": False},
#         async_processing=True
#     )
#     if ctx.video_processor:
#         return ctx.video_processor.label
#     return "unknown"

# # Gender Detection Initialization
# GENDER_LIST = ['Male', 'Female']
# gender_history = deque(maxlen=5)  # Store last 5 gender predictions

# def load_gender_model():
#     """
#     Load the gender detection model.
    
#     Returns:
#         tuple: (face_net, gender_net)
#             - face_net: Face detection network
#             - gender_net: Gender classification network
#     """
#     model_path = os.path.join("Gender_and_age_Detection", "models")
#     faceProto = os.path.join(model_path, "opencv_face_detector.pbtxt")
#     faceModel = os.path.join(model_path, "opencv_face_detector_uint8.pb")
#     genderProto = os.path.join(model_path, "gender_deploy.prototxt")
#     genderModel = os.path.join(model_path, "gender_net.caffemodel")

#     try:
#         face_net = cv2.dnn.readNet(faceModel, faceProto)
#         gender_net = cv2.dnn.readNet(genderModel, genderProto)
#     except Exception as e:
#         st.error(f"Error loading gender model: {e}")
#         st.stop()

#     return face_net, gender_net

# face_net, gender_net = load_gender_model()

# def preprocess_face(face):
#     """
#     Preprocess face image for gender detection.
    
#     Args:
#         face: Cropped face image (numpy array, BGR)
    
#     Returns:
#         blob: Preprocessed blob for model input
#     """
#     try:
#         face_resized = cv2.resize(face, (227, 227), interpolation=cv2.INTER_CUBIC)
#         mean = [78.4263377603, 87.7689143744, 114.895847746]
#         blob = cv2.dnn.blobFromImage(
#             face_resized,
#             scalefactor=1.0,
#             size=(227, 227),
#             mean=mean,
#             swapRB=False,
#             crop=False
#         )
#         return blob
#     except Exception as e:
#         st.warning(f"Face preprocessing error: {e}")
#         return None

# def get_smoothed_prediction(predictions, history, labels):
#     """
#     Smooth predictions using majority voting over recent frames.
    
#     Args:
#         predictions: Model output probabilities
#         history: Deque of recent predictions
#         labels: List of possible labels (GENDER_LIST)
    
#     Returns:
#         label: Smoothed prediction
#     """
#     pred_idx = np.argmax(predictions)
#     history.append(pred_idx)
    
#     if len(history) < history.maxlen:
#         return labels[pred_idx]
    
#     counts = np.bincount(list(history), minlength=len(labels))
#     smoothed_idx = np.argmax(counts)
#     return labels[smoothed_idx]

# def detect_gender(frame):
#     """
#     Detect gender in the frame with improved accuracy.
    
#     Args:
#         frame: Input frame (numpy array, BGR)
    
#     Returns:
#         tuple: (annotated_frame, alert)
#             - annotated_frame: Frame with bounding boxes and gender labels
#             - alert: Alert message if detection occurs, else None
#     """
#     try:
#         alert = None
#         padding = 30
#         min_confidence = 0.85
        
#         blob = cv2.dnn.blobFromImage(
#             frame,
#             scalefactor=1.0,
#             size=(300, 300),
#             mean=[104, 117, 123],
#             swapRB=False,
#             crop=False
#         )
#         face_net.setInput(blob)
#         detections = face_net.forward()

#         h, w = frame.shape[:2]
#         annotated_frame = frame.copy()

#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence > min_confidence:
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 x1, y1, x2, y2 = box.astype("int")
#                 x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
#                 x2, y2 = min(w - 1, x2 + padding), min(h - 1, y2 + padding)
#                 face = frame[y1:y2, x1:x2]
#                 if face.size == 0 or face.shape[0] < 30 or face.shape[1] < 30:
#                     continue
#                 face_blob = preprocess_face(face)
#                 if face_blob is None:
#                     continue
#                 gender_net.setInput(face_blob)
#                 gender_preds = gender_net.forward()
#                 gender = get_smoothed_prediction(gender_preds[0], gender_history, GENDER_LIST)
#                 label = f"{gender}"
#                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                 cv2.putText(
#                     annotated_frame,
#                     label,
#                     (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8,
#                     (0, 255, 0),
#                     2
#                 )
#                 alert = f"{label} detected"
#                 st.session_state.alerts.append([datetime.now(), "Gender Detection", alert])
        
#         return annotated_frame, alert

#     except Exception as e:
#         st.error(f"Gender detection error: {e}")
#         return frame, None

# def detect_guard_attentiveness(frame):
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(frame_rgb)
#     alert = None

#     if results.pose_landmarks:
#         landmarks = results.pose_landmarks.landmark
#         try:
#             nose = landmarks[0]
#             left_eye = landmarks[1]
#             right_eye = landmarks[2]
#             left_shoulder = landmarks[11]
#             right_shoulder = landmarks[12]

#             avg_eye_y = (left_eye.y + right_eye.y) / 2
#             avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

#             if (nose.y - avg_shoulder_y) > -0.05 or (avg_eye_y - avg_shoulder_y) > -0.05:
#                 alert = "Sleepy/drowsy pose detected!"
#                 st.session_state.alerts.append([datetime.now(), "Sleepy Pose", alert])
#                 h, w, _ = frame.shape
#                 cv2.putText(frame, alert, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         except IndexError:
#             pass
#         mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#     return frame, alert

# def detect_face(frame):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
#     results = face_detector.process(rgb_frame)

#     face_names = []
#     face_locations = []

#     if results.detections:
#         for detection in results.detections:
#             bbox = detection.location_data.relative_bounding_box
#             ih, iw, _ = frame.shape
#             x = max(0, int(bbox.xmin * iw))
#             y = max(0, int(bbox.ymin * ih))
#             w = int(bbox.width * iw)
#             h = int(bbox.height * ih)
#             top, right, bottom, left = y, x + w, y + h, x
#             face_locations.append((top, right, bottom, left))

#         encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#         for face_encoding in encodings:
#             name = "Unknown"
#             distances = face_recognition.face_distance(known_encodings, face_encoding)
#             if len(distances) > 0:
#                 best_match_index = np.argmin(distances)
#                 if distances[best_match_index] < 0.45:
#                     name = known_names[best_match_index]
#             face_names.append(name)

#     alert = None
#     if "Unknown" in face_names:
#         alert = "Unknown face detected!"
#         st.session_state.alerts.append([datetime.now(), "Face Detection", alert])

#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
#         cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
#         cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
#         cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
#     return frame, alert

# def detect_objects(frame):
#     try:
#         results = yolo_model(frame, verbose=False, conf=0.5)
#         result = results[0]
#         annotated_frame = result.plot(
#             boxes=True,
#             labels=True,
#             conf=True,
#             font_size=12,
#             line_width=2
#         )
#         class_ids = [int(box.cls) for box in result.boxes]
#         class_counts = Counter(result.names[cls_id] for cls_id in class_ids)
#         detection_summary = ", ".join(f"{count} {name}" for name, count in class_counts.items())
#         if not detection_summary:
#             detection_summary = "No objects detected"
#         alert = None
#         for box in result.boxes:
#             class_name = result.names[int(box.cls)]
#             if class_name == "person" and box.conf > 0.5:
#                 alert = f"{class_name.capitalize()} detected!"
#                 st.session_state.alerts.append([datetime.now(), "Object Detection", alert])
#                 break
#         return annotated_frame, alert, detection_summary
#     except Exception as e:
#         st.error(f"Object detection error: {e}")
#         return frame, None, "Detection failed"

# def detect_crowd_density(frame):
#     results = yolo_model(frame, verbose=False)
#     result = results[0]
#     annotated_frame = result.plot()
#     person_count = sum(1 for box in result.boxes if result.names[int(box.cls)] == "person")
#     alert = "High crowd density detected" if person_count > 5 else None
#     return annotated_frame, alert

# def process_video(model_func, model_name):
#     stframe = st.empty()
#     alert_placeholder = st.empty()
#     summary_placeholder = st.empty()
#     cap = cv2.VideoCapture(1)  # Default webcam

#     if not cap.isOpened():
#         st.error("Failed to open webcam.")
#         return

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Failed to capture video feed.")
#             break

#         if model_func == detect_objects:
#             result_frame, alert, detection_summary = model_func(frame)
#             summary_placeholder.text(f"Detections: {detection_summary}")
#         else:
#             result_frame, alert = model_func(frame)
#             summary_placeholder.text("")

#         frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
#         stframe.image(frame_rgb, channels="RGB", use_column_width=True)

#         if alert:
#             alert_placeholder.error(f"ALERT: {alert} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#         else:
#             alert_placeholder.success(f"No issues detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

#     cap.release()

# def show_dashboard():
#     user = get_logged_user()
#     if not user:
#         st.warning("Please login first")
#         st.session_state.page = "Login"
#         st.rerun()
#         return

#     st.title("AI Surveillance System")
#     st.success(f"Welcome, {user['username']} ({user['role']})")

#     if user["role"] == "admin":
#         st.write("🛡️ Admin Panel: Full access to surveillance features")
#     elif user["role"] == "user":
#         st.write("👤 User Dashboard: Limited access to surveillance features")

#     if st.button("Logout"):
#         st.session_state.pop("token", None)
#         st.session_state.page = "Login"
#         st.session_state.alerts = []
#         st.success("Logged out successfully!")
#         st.rerun()

#     options = ["Face Recognition", "Fight Detection", "Object Detection", "Guard Attentiveness", "Gender Detection", "Crowd Density"]
#     if user["role"] == "user":
#         options = ["Face Recognition", "Object Detection", "Gender Detection"]  # Restrict user access
#     choice = st.sidebar.selectbox("Select Model", options)

#     if choice == "Face Recognition":
#         process_video(detect_face, "face_recognition")
#     elif choice == "Fight Detection":
#         process_video_with_webrtc()
#     elif choice == "Object Detection":
#         process_video(detect_objects, "object_detection")
#     elif choice == "Guard Attentiveness":
#         process_video(detect_guard_attentiveness, "guard_attentiveness")
#     elif choice == "Gender Detection":
#         process_video(detect_gender, "gender_detection")
#     elif choice == "Crowd Density":
#         process_video(detect_crowd_density, "crowd_density")

# def main():
#     if st.session_state.page == "Login":
#         show_login_page()
#     elif st.session_state.page == "Register":
#         show_register_page()
#     elif st.session_state.page == "Dashboard":
#         show_dashboard()

# if __name__ == "__main__":
#     main()







# import streamlit as st
# import cv2
# import numpy as np
# from datetime import datetime, timedelta
# import pandas as pd
# import os
# import mediapipe as mp
# import face_recognition
# from tensorflow.keras.models import load_model
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
# import av
# import smtplib
# from email.mime.text import MIMEText
# import sys
# from ultralytics import YOLO
# from collections import Counter, deque
# from auth import login_user, register_user
# from utils import get_logged_user

# # Add the Gender_and_age_Detection folder to Python's path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'Gender_and_age_Detection'))

# # Initialize session state
# if "alerts" not in st.session_state:
#     st.session_state.alerts = []
# if "page" not in st.session_state:
#     st.session_state.page = "Login"
# if "token" not in st.session_state:
#     st.session_state.token = None

# # Set page config
# st.set_page_config(page_title="AI Surveillance System", layout="wide")

# def show_login_page():
#     st.title("Login")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("Login"):
#             token = login_user(username, password)
#             if token:
#                 st.session_state.token = token
#                 st.session_state.page = "Dashboard"
#                 st.success("Logged in successfully!")
#                 st.rerun()
#             else:
#                 st.error("Invalid credentials")
#     with col2:
#         if st.button("Go to Register"):
#             st.session_state.page = "Register"
#             st.rerun()

# def show_register_page():
#     st.title("Register")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     if st.button("Register"):
#         if register_user(username, password):
#             st.success("Registered successfully! Please go to Login.")
#         else:
#             st.error("User already exists!")
#     if st.button("Back to Login"):
#         st.session_state.page = "Login"
#         st.rerun()

# # Surveillance Dashboard Functions
# # Importing face recognition functions
# from import_face_recognition import load_known_faces, recognize_faces_live
# known_encodings, known_names = load_known_faces()

# # Initialize MediaPipe Pose and Drawing utilities
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# mp_draw = mp.solutions.drawing_utils

# # Initialize MediaPipe Face Detection
# mp_face = mp.solutions.face_detection
# face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# # Load YOLOv8 model globally
# try:
#     yolo_model = YOLO("yolov8n.pt")  # Replace with yolov8.pt or another variant if needed
# except Exception as e:
#     st.error(f"Error loading YOLO model: {e}")
#     st.stop()

# # Load fight detection model
# def load_fight_model():
#     try:
#         return load_model(r'C:\Users\kriya\Desktop\face\lstm-fight-detection.h5')
#     except Exception as e:
#         st.error(f"Error loading fight model: {e}")
#         st.stop()

# fight_model = load_fight_model()

# # WebRTC configuration
# RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# # Function to blur faces based on user role
# def blur_faces(frame, user_role):
#     """
#     Blur faces in the frame if the user is not an admin.
    
#     Args:
#         frame: Input frame (numpy array, BGR)
#         user_role: User's role ('admin' or 'user')
    
#     Returns:
#         frame: Frame with faces blurred (if user_role is 'user') or unchanged (if 'admin')
#     """
#     if user_role == "admin":
#         return frame  # No blurring for admins

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_results = face_detection.process(frame_rgb)
#     if face_results.detections:
#         for detection in face_results.detections:
#             bbox = detection.location_data.relative_bounding_box
#             h, w, _ = frame.shape
#             x1 = int(bbox.xmin * w)
#             y1 = int(bbox.ymin * h)
#             x2 = x1 + int(bbox.width * w)
#             y2 = y1 + int(bbox.height * h)
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(w, x2), min(h, y2)
#             face_region = frame[y1:y2, x1:x2]
#             if face_region.size > 0:
#                 blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
#                 frame[y1:y2, x1:x2] = blurred
#     return frame

# # Violence Processor for Fight Detection
# class ViolenceProcessor(VideoProcessorBase):
#     def __init__(self, user_role):
#         self.label = "unknown"
#         self.label_history = []
#         self.lm_list = []
#         self.fight_start_time = None
#         self.alert_sent = False
#         self.user_role = user_role  # Store user role

#     def make_landmark_timestep(self, results):
#         if results.pose_landmarks:
#             c_lm = []
#             for lm in results.pose_landmarks.landmark:
#                 c_lm.extend([lm.x, lm.y, lm.z])
#             return c_lm
#         return None

#     def draw_landmark_on_image(self, mp_draw, results, frame):
#         if results.pose_landmarks:
#             mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#             for id, lm in enumerate(results.pose_landmarks.landmark):
#                 h, w, c = frame.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
#         return frame

#     def detect(self, model, lm_list):
#         if len(lm_list) >= 20:
#             lm_array = np.array(lm_list[-20:])
#             lm_array = np.expand_dims(lm_array, axis=0)
#             try:
#                 result = model.predict(lm_array, verbose=0)[0]
#                 pred_label = "fight" if result[0] > 0.5 else "normal"
#                 for idx in [15, 16, 13, 14]:
#                     x_diff = lm_list[-1][idx*3] - lm_list[-2][idx*3]
#                     y_diff = lm_list[-1][idx*3+1] - lm_list[-2][idx*3+1]
#                     if abs(x_diff) > 0.07 or abs(y_diff) > 0.07:
#                         pred_label = "fight"
#                         break
#                 self.label_history.append(pred_label)
#                 if len(self.label_history) > 3:
#                     self.label_history.pop(0)
#                 self.label = max(set(self.label_history), key=self.label_history.count)
#             except Exception as e:
#                 st.error(f"Prediction error: {e}")
#                 self.label = "ERROR"
#         return self.label

#     def send_alert_email(self):
#         try:
#             sender = "viraj.salunke23@spit.ac.in"
#             password = "lppqwukqzossvidg"
#             receiver = "virajsalunke12@gmail.com"
#             subject = "⚠️ Fight Detected Alert"
#             body = f"⚠️ A violent action was detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} in CCTV camera feed."

#             msg = MIMEText(body)
#             msg["Subject"] = subject
#             msg["From"] = sender
#             msg["To"] = receiver

#             with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
#                 server.login(sender, password)
#                 server.sendmail(sender, receiver, msg.as_string())
#             print("✅ Email sent")
#         except Exception as e:
#             print(f"❌ Email error: {e}")

#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = pose.process(frame_rgb)

#         lm = self.make_landmark_timestep(results)
#         if lm:
#             self.lm_list.append(lm)

#         img = self.draw_landmark_on_image(mp_draw, results, img)

#         # Apply face blurring based on user role
#         img = blur_faces(img, self.user_role)

#         self.label = self.detect(fight_model, self.lm_list)
#         color = (0, 0, 255) if self.label == "fight" else (0, 255, 0)
#         cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color, 2)
#         cv2.putText(img, self.label.upper(), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#         if self.label == "fight":
#             if self.fight_start_time is None:
#                 self.fight_start_time = datetime.now()
#             elif datetime.now() - self.fight_start_time > timedelta(seconds=5) and not self.alert_sent:
#                 self.send_alert_email()
#                 self.alert_sent = True
#         else:
#             self.fight_start_time = None
#             self.alert_sent = False

#         return av.VideoFrame.from_ndarray(img, format="bgr24")

# def process_video_with_webrtc(user_role):
#     ctx = webrtc_streamer(
#         key="violence-detection",
#         video_processor_factory=lambda: ViolenceProcessor(user_role),
#         rtc_configuration=RTC_CONFIGURATION,
#         media_stream_constraints={"video": True, "audio": False},
#         async_processing=True
#     )
#     if ctx.video_processor:
#         return ctx.video_processor.label
#     return "unknown"

# # Gender Detection Initialization
# GENDER_LIST = ['Male', 'Female']
# gender_history = deque(maxlen=5)  # Store last 5 gender predictions

# def load_gender_model():
#     """
#     Load the gender detection model.
    
#     Returns:
#         tuple: (face_net, gender_net)
#             - face_net: Face detection network
#             - gender_net: Gender classification network
#     """
#     model_path = os.path.join("Gender_and_age_Detection", "models")
#     faceProto = os.path.join(model_path, "opencv_face_detector.pbtxt")
#     faceModel = os.path.join(model_path, "opencv_face_detector_uint8.pb")
#     genderProto = os.path.join(model_path, "gender_deploy.prototxt")
#     genderModel = os.path.join(model_path, "gender_net.caffemodel")

#     try:
#         face_net = cv2.dnn.readNet(faceModel, faceProto)
#         gender_net = cv2.dnn.readNet(genderModel, genderProto)
#     except Exception as e:
#         st.error(f"Error loading gender model: {e}")
#         st.stop()

#     return face_net, gender_net

# face_net, gender_net = load_gender_model()

# def preprocess_face(face):
#     """
#     Preprocess face image for gender detection.
    
#     Args:
#         face: Cropped face image (numpy array, BGR)
    
#     Returns:
#         blob: Preprocessed blob for model input
#     """
#     try:
#         face_resized = cv2.resize(face, (227, 227), interpolation=cv2.INTER_CUBIC)
#         mean = [78.4263377603, 87.7689143744, 114.895847746]
#         blob = cv2.dnn.blobFromImage(
#             face_resized,
#             scalefactor=1.0,
#             size=(227, 227),
#             mean=mean,
#             swapRB=False,
#             crop=False
#         )
#         return blob
#     except Exception as e:
#         st.warning(f"Face preprocessing error: {e}")
#         return None

# def get_smoothed_prediction(predictions, history, labels):
#     """
#     Smooth predictions using majority voting over recent frames.
    
#     Args:
#         predictions: Model output probabilities
#         history: Deque of recent predictions
#         labels: List of possible labels (GENDER_LIST)
    
#     Returns:
#         label: Smoothed prediction
#     """
#     pred_idx = np.argmax(predictions)
#     history.append(pred_idx)
    
#     if len(history) < history.maxlen:
#         return labels[pred_idx]
    
#     counts = np.bincount(list(history), minlength=len(labels))
#     smoothed_idx = np.argmax(counts)
#     return labels[smoothed_idx]

# def detect_gender(frame, user_role):
#     """
#     Detect gender in the frame with improved accuracy.
    
#     Args:
#         frame: Input frame (numpy array, BGR)
#         user_role: User's role ('admin' or 'user')
    
#     Returns:
#         tuple: (annotated_frame, alert)
#             - annotated_frame: Frame with bounding boxes and gender labels
#             - alert: Alert message if detection occurs, else None
#     """
#     try:
#         alert = None
#         padding = 30
#         min_confidence = 0.85
        
#         blob = cv2.dnn.blobFromImage(
#             frame,
#             scalefactor=1.0,
#             size=(300, 300),
#             mean=[104, 117, 123],
#             swapRB=False,
#             crop=False
#         )
#         face_net.setInput(blob)
#         detections = face_net.forward()

#         h, w = frame.shape[:2]
#         annotated_frame = frame.copy()

#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence > min_confidence:
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 x1, y1, x2, y2 = box.astype("int")
#                 x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
#                 x2, y2 = min(w - 1, x2 + padding), min(h - 1, y2 + padding)
#                 face = frame[y1:y2, x1:x2]
#                 if face.size == 0 or face.shape[0] < 30 or face.shape[1] < 30:
#                     continue
#                 face_blob = preprocess_face(face)
#                 if face_blob is None:
#                     continue
#                 gender_net.setInput(face_blob)
#                 gender_preds = gender_net.forward()
#                 gender = get_smoothed_prediction(gender_preds[0], gender_history, GENDER_LIST)
#                 label = f"{gender}"
#                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                 cv2.putText(
#                     annotated_frame,
#                     label,
#                     (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8,
#                     (0, 255, 0),
#                     2
#                 )
#                 alert = f"{label} detected"
#                 st.session_state.alerts.append([datetime.now(), "Gender Detection", alert])
        
#         # Apply face blurring after gender detection
#         annotated_frame = blur_faces(annotated_frame, user_role)
#         return annotated_frame, alert

#     except Exception as e:
#         st.error(f"Gender detection error: {e}")
#         return frame, None

# def detect_guard_attentiveness(frame, user_role):
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(frame_rgb)
#     alert = None

#     if results.pose_landmarks:
#         landmarks = results.pose_landmarks.landmark
#         try:
#             nose = landmarks[0]
#             left_eye = landmarks[1]
#             right_eye = landmarks[2]
#             left_shoulder = landmarks[11]
#             right_shoulder = landmarks[12]

#             avg_eye_y = (left_eye.y + right_eye.y) / 2
#             avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

#             if (nose.y - avg_shoulder_y) > -0.05 or (avg_eye_y - avg_shoulder_y) > -0.05:
#                 alert = "Sleepy/drowsy pose detected!"
#                 st.session_state.alerts.append([datetime.now(), "Sleepy Pose", alert])
#                 h, w, _ = frame.shape
#                 cv2.putText(frame, alert, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         except IndexError:
#             pass
#         mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
#     # Apply face blurring
#     frame = blur_faces(frame, user_role)
#     return frame, alert

# def detect_face(frame, user_role):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
#     results = face_detector.process(rgb_frame)

#     face_names = []
#     face_locations = []

#     if results.detections:
#         for detection in results.detections:
#             bbox = detection.location_data.relative_bounding_box
#             ih, iw, _ = frame.shape
#             x = max(0, int(bbox.xmin * iw))
#             y = max(0, int(bbox.ymin * ih))
#             w = int(bbox.width * iw)
#             h = int(bbox.height * ih)
#             top, right, bottom, left = y, x + w, y + h, x
#             face_locations.append((top, right, bottom, left))

#         encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#         for face_encoding in encodings:
#             name = "Unknown"
#             distances = face_recognition.face_distance(known_encodings, face_encoding)
#             if len(distances) > 0:
#                 best_match_index = np.argmin(distances)
#                 if distances[best_match_index] < 0.45:
#                     name = known_names[best_match_index]
#             face_names.append(name)

#     alert = None
#     if "Unknown" in face_names:
#         alert = "Unknown face detected!"
#         st.session_state.alerts.append([datetime.now(), "Face Detection", alert])

#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
#         cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
#         cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
#         cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
#     # Apply face blurring
#     frame = blur_faces(frame, user_role)
#     return frame, alert

# def detect_objects(frame, user_role):
#     try:
#         results = yolo_model(frame, verbose=False, conf=0.5)
#         result = results[0]
#         annotated_frame = result.plot(
#             boxes=True,
#             labels=True,
#             conf=True,
#             font_size=12,
#             line_width=2
#         )
#         class_ids = [int(box.cls) for box in result.boxes]
#         class_counts = Counter(result.names[cls_id] for cls_id in class_ids)
#         detection_summary = ", ".join(f"{count} {name}" for name, count in class_counts.items())
#         if not detection_summary:
#             detection_summary = "No objects detected"
#         alert = None
#         for box in result.boxes:
#             class_name = result.names[int(box.cls)]
#             if class_name == "person" and box.conf > 0.5:
#                 alert = f"{class_name.capitalize()} detected!"
#                 st.session_state.alerts.append([datetime.now(), "Object Detection", alert])
#                 break
        
#         # Apply face blurring
#         annotated_frame = blur_faces(annotated_frame, user_role)
#         return annotated_frame, alert, detection_summary
#     except Exception as e:
#         st.error(f"Object detection error: {e}")
#         return frame, None, "Detection failed"

# def detect_crowd_density(frame, user_role):
#     results = yolo_model(frame, verbose=False)
#     result = results[0]
#     annotated_frame = result.plot()
#     person_count = sum(1 for box in result.boxes if result.names[int(box.cls)] == "person")
#     alert = "High crowd density detected" if person_count > 5 else None
    
#     # Apply face blurring
#     annotated_frame = blur_faces(annotated_frame, user_role)
#     return annotated_frame, alert

# def process_video(model_func, model_name, user_role):
#     stframe = st.empty()
#     alert_placeholder = st.empty()
#     summary_placeholder = st.empty()
#     cap = cv2.VideoCapture(1)  # Default webcam

#     if not cap.isOpened():
#         st.error("Failed to open webcam.")
#         return

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Failed to capture video feed.")
#             break

#         if model_func == detect_objects:
#             result_frame, alert, detection_summary = model_func(frame, user_role)
#             summary_placeholder.text(f"Detections: {detection_summary}")
#         else:
#             result_frame, alert = model_func(frame, user_role)
#             summary_placeholder.text("")

#         frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
#         stframe.image(frame_rgb, channels="RGB", use_column_width=True)

#         if alert:
#             alert_placeholder.error(f"ALERT: {alert} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#         else:
#             alert_placeholder.success(f"No issues detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

#     cap.release()

# def show_dashboard():
#     user = get_logged_user()
#     if not user or not st.session_state.token:
#         st.warning("Please login first")
#         st.session_state.page = "Login"
#         st.session_state.token = None
#         st.rerun()
#         return

#     st.title("AI Surveillance System")
#     st.success(f"Welcome, {user['username']} ({user['role']})")

#     if user["role"] == "admin":
#         st.write("🛡️ Admin Panel: Full access to surveillance features")
#     elif user["role"] == "user":
#         st.write("👤 User Dashboard: Limited access to surveillance features")

#     if st.button("Logout"):
#         st.session_state.pop("token", None)
#         st.session_state.page = "Login"
#         st.session_state.alerts = []
#         st.success("Logged out successfully!")
#         st.rerun()

#     options = ["Face Recognition", "Fight Detection", "Object Detection", "Guard Attentiveness", "Gender Detection", "Crowd Density"]
#     if user["role"] == "user":
#         options = ["Face Recognition", "Object Detection", "Gender Detection"]  # Restrict user access
#     choice = st.sidebar.selectbox("Select Model", options)

#     if choice == "Face Recognition":
#         process_video(detect_face, "face_recognition", user["role"])
#     elif choice == "Fight Detection":
#         process_video_with_webrtc(user["role"])
#     elif choice == "Object Detection":
#         process_video(detect_objects, "object_detection", user["role"])
#     elif choice == "Guard Attentiveness":
#         process_video(detect_guard_attentiveness, "guard_attentiveness", user["role"])
#     elif choice == "Gender Detection":
#         process_video(detect_gender, "gender_detection", user["role"])
#     elif choice == "Crowd Density":
#         process_video(detect_crowd_density, "crowd_density", user["role"])

# def main():
#     if st.session_state.page == "Login":
#         show_login_page()
#     elif st.session_state.page == "Register":
#         show_register_page()
#     elif st.session_state.page == "Dashboard":
#         show_dashboard()

# if __name__ == "__main__":
#     main()







# import streamlit as st
# import cv2
# import numpy as np
# from datetime import datetime, timedelta
# import pandas as pd
# import os
# import mediapipe as mp
# import face_recognition
# from tensorflow.keras.models import load_model
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
# import av
# import smtplib
# from email.mime.text import MIMEText
# import sys
# from ultralytics import YOLO
# from collections import Counter, deque
# from auth import login_user, register_user
# from utils import get_logged_user

# # Add the Gender_and_age_Detection folder to Python's path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'Gender_and_age_Detection'))

# # Initialize session state
# if "alerts" not in st.session_state:
#     st.session_state.alerts = []
# if "page" not in st.session_state:
#     st.session_state.page = "Login"
# if "token" not in st.session_state:
#     st.session_state.token = None

# # Set page config
# st.set_page_config(page_title="AI Surveillance System", layout="wide")

# def show_login_page():
#     st.title("Login")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("Login"):
#             token = login_user(username, password)
#             if token:
#                 st.session_state.token = token
#                 st.session_state.page = "Dashboard"
#                 st.success("Logged in successfully!")
#                 st.rerun()
#             else:
#                 st.error("Invalid credentials")
#     with col2:
#         if st.button("Go to Register"):
#             st.session_state.page = "Register"
#             st.rerun()

# def show_register_page():
#     st.title("Register")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     if st.button("Register"):
#         if register_user(username, password):
#             st.success("Registered successfully! Please go to Login.")
#         else:
#             st.error("User already exists!")
#     if st.button("Back to Login"):
#         st.session_state.page = "Login"
#         st.rerun()

# # Surveillance Dashboard Functions
# # Importing face recognition functions
# from import_face_recognition import load_known_faces, recognize_faces_live
# known_encodings, known_names = load_known_faces()

# # Initialize MediaPipe Pose and Drawing utilities
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# mp_draw = mp.solutions.drawing_utils

# # Initialize MediaPipe Face Detection
# mp_face = mp.solutions.face_detection
# face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# # Load YOLOv8 model globally
# try:
#     yolo_model = YOLO("yolov8n.pt")  # Replace with yolov8.pt or another variant if needed
# except Exception as e:
#     st.error(f"Error loading YOLO model: {e}")
#     st.stop()

# # Load fight detection model
# def load_fight_model():
#     try:
#         return load_model(r'C:\Users\kriya\Desktop\face\lstm-fight-detection.h5')
#     except Exception as e:
#         st.error(f"Error loading fight model: {e}")
#         st.stop()

# fight_model = load_fight_model()

# # WebRTC configuration
# RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# # Violence Processor for Fight Detection
# class ViolenceProcessor(VideoProcessorBase):
#     def __init__(self):
#         self.label = "unknown"
#         self.label_history = []
#         self.lm_list = []
#         self.fight_start_time = None
#         self.alert_sent = False

#     def make_landmark_timestep(self, results):
#         if results.pose_landmarks:
#             c_lm = []
#             for lm in results.pose_landmarks.landmark:
#                 c_lm.extend([lm.x, lm.y, lm.z])
#             return c_lm
#         return None

#     def draw_landmark_on_image(self, mp_draw, results, frame):
#         if results.pose_landmarks:
#             mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#             for id, lm in enumerate(results.pose_landmarks.landmark):
#                 h, w, c = frame.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
#         return frame

#     def detect(self, model, lm_list):
#         if len(lm_list) >= 20:
#             lm_array = np.array(lm_list[-20:])
#             lm_array = np.expand_dims(lm_array, axis=0)
#             try:
#                 result = model.predict(lm_array, verbose=0)[0]
#                 pred_label = "fight" if result[0] > 0.5 else "normal"
#                 for idx in [15, 16, 13, 14]:
#                     x_diff = lm_list[-1][idx*3] - lm_list[-2][idx*3]
#                     y_diff = lm_list[-1][idx*3+1] - lm_list[-2][idx*3+1]
#                     if abs(x_diff) > 0.07 or abs(y_diff) > 0.07:
#                         pred_label = "fight"
#                         break
#                 self.label_history.append(pred_label)
#                 if len(self.label_history) > 3:
#                     self.label_history.pop(0)
#                 self.label = max(set(self.label_history), key=self.label_history.count)
#             except Exception as e:
#                 st.error(f"Prediction error: {e}")
#                 self.label = "ERROR"
#         return self.label

#     def send_alert_email(self):
#         try:
#             sender = "viraj.salunke23@spit.ac.in"
#             password = "lppqwukqzossvidg"
#             receiver = "virajsalunke12@gmail.com"
#             subject = "⚠️ Fight Detected Alert"
#             body = f"⚠️ A violent action was detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} in CCTV camera feed."

#             msg = MIMEText(body)
#             msg["Subject"] = subject
#             msg["From"] = sender
#             msg["To"] = receiver

#             with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
#                 server.login(sender, password)
#                 server.sendmail(sender, receiver, msg.as_string())
#             print("✅ Email sent")
#         except Exception as e:
#             print(f"❌ Email error: {e}")

#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = pose.process(frame_rgb)
#         face_results = face_detection.process(frame_rgb)

#         lm = self.make_landmark_timestep(results)
#         if lm:
#             self.lm_list.append(lm)

#         img = self.draw_landmark_on_image(mp_draw, results, img)

#         # Check user role for face blurring
#         user = get_logged_user()
#         is_admin = user and user.get('role') == 'admin'

#         if face_results.detections and not is_admin:  # Blur faces for non-admins
#             for detection in face_results.detections:
#                 bbox = detection.location_data.relative_bounding_box
#                 h, w, _ = img.shape
#                 x1 = int(bbox.xmin * w)
#                 y1 = int(bbox.ymin * h)
#                 x2 = x1 + int(bbox.width * w)
#                 y2 = y1 + int(bbox.height * h)
#                 x1, y1 = max(0, x1), max(0, y1)
#                 x2, y2 = min(w, x2), min(h, y2)
#                 face_region = img[y1:y2, x1:x2]
#                 if face_region.size > 0:
#                     blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
#                     img[y1:y2, x1:x2] = blurred

#         self.label = self.detect(fight_model, self.lm_list)
#         color = (0, 0, 255) if self.label == "fight" else (0, 255, 0)
#         cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color, 2)
#         cv2.putText(img, self.label.upper(), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#         if self.label == "fight":
#             if self.fight_start_time is None:
#                 self.fight_start_time = datetime.now()
#             elif datetime.now() - self.fight_start_time > timedelta(seconds=5) and not self.alert_sent:
#                 self.send_alert_email()
#                 self.alert_sent = True
#         else:
#             self.fight_start_time = None
#             self.alert_sent = False

#         return av.VideoFrame.from_ndarray(img, format="bgr24")

# def process_video_with_webrtc():
#     ctx = webrtc_streamer(
#         key="violence-detection",
#         video_processor_factory=ViolenceProcessor,
#         rtc_configuration=RTC_CONFIGURATION,
#         media_stream_constraints={"video": True, "audio": False},
#         async_processing=True
#     )
#     if ctx.video_processor:
#         return ctx.video_processor.label
#     return "unknown"

# # Gender Detection Initialization
# GENDER_LIST = ['Male', 'Female']
# gender_history = deque(maxlen=5)  # Store last 5 gender predictions

# def load_gender_model():
#     """
#     Load the gender detection model.
    
#     Returns:
#         tuple: (face_net, gender_net)
#             - face_net: Face detection network
#             - gender_net: Gender classification network
#     """
#     model_path = os.path.join("Gender_and_age_Detection", "models")
#     faceProto = os.path.join(model_path, "opencv_face_detector.pbtxt")
#     faceModel = os.path.join(model_path, "opencv_face_detector_uint8.pb")
#     genderProto = os.path.join(model_path, "gender_deploy.prototxt")
#     genderModel = os.path.join(model_path, "gender_net.caffemodel")

#     try:
#         face_net = cv2.dnn.readNet(faceModel, faceProto)
#         gender_net = cv2.dnn.readNet(genderModel, genderProto)
#     except Exception as e:
#         st.error(f"Error loading gender model: {e}")
#         st.stop()

#     return face_net, gender_net

# face_net, gender_net = load_gender_model()

# def preprocess_face(face):
#     """
#     Preprocess face image for gender detection.
    
#     Args:
#         face: Cropped face image (numpy array, BGR)
    
#     Returns:
#         blob: Preprocessed blob for model input
#     """
#     try:
#         face_resized = cv2.resize(face, (227, 227), interpolation=cv2.INTER_CUBIC)
#         mean = [78.4263377603, 87.7689143744, 114.895847746]
#         blob = cv2.dnn.blobFromImage(
#             face_resized,
#             scalefactor=1.0,
#             size=(227, 227),
#             mean=mean,
#             swapRB=False,
#             crop=False
#         )
#         return blob
#     except Exception as e:
#         st.warning(f"Face preprocessing error: {e}")
#         return None

# def get_smoothed_prediction(predictions, history, labels):
#     """
#     Smooth predictions using majority voting over recent frames.
    
#     Args:
#         predictions: Model output probabilities
#         history: Deque of recent predictions
#         labels: List of possible labels (GENDER_LIST)
    
#     Returns:
#         label: Smoothed prediction
#     """
#     pred_idx = np.argmax(predictions)
#     history.append(pred_idx)
    
#     if len(history) < history.maxlen:
#         return labels[pred_idx]
    
#     counts = np.bincount(list(history), minlength=len(labels))
#     smoothed_idx = np.argmax(counts)
#     return labels[smoothed_idx]

# def blur_faces(frame, is_admin):
#     """
#     Blur faces in the frame if the user is not an admin.
    
#     Args:
#         frame: Input frame (numpy array, BGR)
#         is_admin: Boolean indicating if the user is an admin
    
#     Returns:
#         frame: Frame with blurred faces (if not admin) or original frame
#     """
#     if is_admin:
#         return frame
    
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_results = face_detection.process(frame_rgb)
#     if face_results.detections:
#         for detection in face_results.detections:
#             bbox = detection.location_data.relative_bounding_box
#             h, w, _ = frame.shape
#             x1 = int(bbox.xmin * w)
#             y1 = int(bbox.ymin * h)
#             x2 = x1 + int(bbox.width * w)
#             y2 = y1 + int(bbox.height * h)
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(w, x2), min(h, y2)
#             face_region = frame[y1:y2, x1:x2]
#             if face_region.size > 0:
#                 blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
#                 frame[y1:y2, x1:x2] = blurred
#     return frame

# def detect_gender(frame):
#     """
#     Detect gender in the frame with improved accuracy.
    
#     Args:
#         frame: Input frame (numpy array, BGR)
    
#     Returns:
#         tuple: (annotated_frame, alert)
#             - annotated_frame: Frame with bounding boxes and gender labels
#             - alert: Alert message if detection occurs, else None
#     """
#     try:
#         alert = None
#         padding = 30
#         min_confidence = 0.85
        
#         blob = cv2.dnn.blobFromImage(
#             frame,
#             scalefactor=1.0,
#             size=(300, 300),
#             mean=[104, 117, 123],
#             swapRB=False,
#             crop=False
#         )
#         face_net.setInput(blob)
#         detections = face_net.forward()

#         h, w = frame.shape[:2]
#         annotated_frame = frame.copy()

#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence > min_confidence:
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 x1, y1, x2, y2 = box.astype("int")
#                 x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
#                 x2, y2 = min(w - 1, x2 + padding), min(h - 1, y2 + padding)
#                 face = frame[y1:y2, x1:x2]
#                 if face.size == 0 or face.shape[0] < 30 or face.shape[1] < 30:
#                     continue
#                 face_blob = preprocess_face(face)
#                 if face_blob is None:
#                     continue
#                 gender_net.setInput(face_blob)
#                 gender_preds = gender_net.forward()
#                 gender = get_smoothed_prediction(gender_preds[0], gender_history, GENDER_LIST)
#                 label = f"{gender}"
#                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                 cv2.putText(
#                     annotated_frame,
#                     label,
#                     (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8,
#                     (0, 255, 0),
#                     2
#                 )
#                 alert = f"{label} detected"
#                 st.session_state.alerts.append([datetime.now(), "Gender Detection", alert])
        
#         # Apply face blurring for non-admins after gender detection
#         user = get_logged_user()
#         is_admin = user and user.get('role') == 'admin'
#         annotated_frame = blur_faces(annotated_frame, is_admin)
        
#         return annotated_frame, alert

#     except Exception as e:
#         st.error(f"Gender detection error: {e}")
#         return frame, None

# def detect_guard_attentiveness(frame):
#     user = get_logged_user()
#     is_admin = user and user.get('role') == 'admin'
    
#     # Apply face blurring for non-admins
#     frame = blur_faces(frame, is_admin)
    
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(frame_rgb)
#     alert = None

#     if results.pose_landmarks:
#         landmarks = results.pose_landmarks.landmark
#         try:
#             nose = landmarks[0]
#             left_eye = landmarks[1]
#             right_eye = landmarks[2]
#             left_shoulder = landmarks[11]
#             right_shoulder = landmarks[12]

#             avg_eye_y = (left_eye.y + right_eye.y) / 2
#             avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

#             if (nose.y - avg_shoulder_y) > -0.05 or (avg_eye_y - avg_shoulder_y) > -0.05:
#                 alert = "Sleepy/drowsy pose detected!"
#                 st.session_state.alerts.append([datetime.now(), "Sleepy Pose", alert])
#                 h, w, _ = frame.shape
#                 cv2.putText(frame, alert, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         except IndexError:
#             pass
#         mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#     return frame, alert

# def detect_face(frame):
#     user = get_logged_user()
    
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
#     results = face_detector.process(rgb_frame)

#     face_names = []
#     face_locations = []

#     if results.detections:
#         for detection in results.detections:
#             bbox = detection.location_data.relative_bounding_box
#             ih, iw, _ = frame.shape
#             x = max(0, int(bbox.xmin * iw))
#             y = max(0, int(bbox.ymin * ih))
#             w = int(bbox.width * iw)
#             h = int(bbox.height * ih)
#             top, right, bottom, left = y, x + w, y + h, x
#             face_locations.append((top, right, bottom, left))

#         encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#         for face_encoding in encodings:
#             name = "Unknown"
#             distances = face_recognition.face_distance(known_encodings, face_encoding)
#             if len(distances) > 0:
#                 best_match_index = np.argmin(distances)
#                 if distances[best_match_index] < 0.45:
#                     name = known_names[best_match_index]
#             face_names.append(name)

#     alert = None
#     if "Unknown" in face_names:
#         alert = "Unknown face detected!"
#         st.session_state.alerts.append([datetime.now(), "Face Detection", alert])

#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
#         cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
#         cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
#         cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
#     return frame, alert

# def detect_objects(frame):
#     user = get_logged_user()
#     is_admin = user and user.get('role') == 'admin'
    
#     # Apply face blurring for non-admins
#     frame = blur_faces(frame, is_admin)
    
#     try:
#         results = yolo_model(frame, verbose=False, conf=0.5)
#         result = results[0]
#         annotated_frame = result.plot(
#             boxes=True,
#             labels=True,
#             conf=True,
#             font_size=12,
#             line_width=2
#         )
#         class_ids = [int(box.cls) for box in result.boxes]
#         class_counts = Counter(result.names[cls_id] for cls_id in class_ids)
#         detection_summary = ", ".join(f"{count} {name}" for name, count in class_counts.items())
#         if not detection_summary:
#             detection_summary = "No objects detected"
#         alert = None
#         for box in result.boxes:
#             class_name = result.names[int(box.cls)]
#             if class_name == "person" and box.conf > 0.5:
#                 alert = f"{class_name.capitalize()} detected!"
#                 st.session_state.alerts.append([datetime.now(), "Object Detection", alert])
#                 break
#         return annotated_frame, alert, detection_summary
#     except Exception as e:
#         st.error(f"Object detection error: {e}")
#         return frame, None, "Detection failed"

# def detect_crowd_density(frame):
#     user = get_logged_user()
#     is_admin = user and user.get('role') == 'admin'
    
#     # Apply face blurring for non-admins
#     frame = blur_faces(frame, is_admin)
    
#     results = yolo_model(frame, verbose=False)
#     result = results[0]
#     annotated_frame = result.plot()
#     person_count = sum(1 for box in result.boxes if result.names[int(box.cls)] == "person")
#     alert = "High crowd density detected" if person_count > 5 else None
#     return annotated_frame, alert

# def process_video(model_func, model_name):
#     stframe = st.empty()
#     alert_placeholder = st.empty()
#     summary_placeholder = st.empty()
#     cap = cv2.VideoCapture(1)  # Default webcam

#     if not cap.isOpened():
#         st.error("Failed to open webcam.")
#         return

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Failed to capture video feed.")
#             break

#         if model_func == detect_objects:
#             result_frame, alert, detection_summary = model_func(frame)
#             summary_placeholder.text(f"Detections: {detection_summary}")
#         else:
#             result_frame, alert = model_func(frame)
#             summary_placeholder.text("")

#         frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
#         stframe.image(frame_rgb, channels="RGB", use_column_width=True)

#         if alert:
#             alert_placeholder.error(f"ALERT: {alert} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#         else:
#             alert_placeholder.success(f"No issues detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

#     cap.release()

# def show_dashboard():
#     user = get_logged_user()
#     if not user or not st.session_state.token:
#         st.warning("Please login first")
#         st.session_state.page = "Login"
#         st.session_state.token = None
#         st.rerun()
#         return

#     st.title("AI Surveillance System")
#     st.success(f"Welcome, {user['username']} ({user['role']})")

#     if user["role"] == "admin":
#         st.write("🛡️ Admin Panel: Full access to surveillance features")
#     elif user["role"] == "user":
#         st.write("👤 User Dashboard: Limited access to surveillance features")

#     if st.button("Logout"):
#         st.session_state.pop("token", None)
#         st.session_state.page = "Login"
#         st.session_state.alerts = []
#         st.success("Logged out successfully!")
#         st.rerun()

#     options = ["Face Recognition", "Fight Detection", "Object Detection", "Guard Attentiveness", "Gender Detection"]
#     if user["role"] == "user":
#         options = ["Face Recognition", "Object Detection", "Gender Detection"]  # Restrict user access
#     choice = st.sidebar.selectbox("Select Model", options)

#     if choice == "Face Recognition":
#         process_video(detect_face, "face_recognition")
#     elif choice == "Fight Detection":
#         process_video_with_webrtc()
#     elif choice == "Object Detection":
#         process_video(detect_objects, "object_detection")
#     elif choice == "Guard Attentiveness":
#         process_video(detect_guard_attentiveness, "guard_attentiveness")
#     elif choice == "Gender Detection":
#         process_video(detect_gender, "gender_detection")
#     elif choice == "Crowd Density":
#         process_video(detect_crowd_density, "crowd_density")

# def main():
#     if st.session_state.page == "Login":
#         show_login_page()
#     elif st.session_state.page == "Register":
#         show_register_page()
#     elif st.session_state.page == "Dashboard":
#         show_dashboard()

# if __name__ == "__main__":
#     main()



# import streamlit as st
# import cv2
# import numpy as np
# from datetime import datetime, timedelta
# import pandas as pd
# import os
# import mediapipe as mp
# import face_recognition
# from tensorflow.keras.models import load_model
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
# import av
# import smtplib
# from email.mime.text import MIMEText
# import sys
# from ultralytics import YOLO
# from collections import Counter, deque
# from auth import login_user, register_user
# from utils import get_logged_user

# # Add the Gender_and_age_Detection folder to Python's path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'Gender_and_age_Detection'))

# # Initialize session state
# if "alerts" not in st.session_state:
#     st.session_state.alerts = []
# if "page" not in st.session_state:
#     st.session_state.page = "Login"
# if "token" not in st.session_state:
#     st.session_state.token = None
# if "detection_states" not in st.session_state:
#     st.session_state.detection_states = {}

# # Set page config
# st.set_page_config(page_title="AI Surveillance System", layout="wide")

# def show_login_page():
#     st.title("Login")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("Login"):
#             token = login_user(username, password)
#             if token:
#                 st.session_state.token = token
#                 st.session_state.page = "Dashboard"
#                 st.success("Logged in successfully!")
#                 st.rerun()
#             else:
#                 st.error("Invalid credentials")
#     with col2:
#         if st.button("Go to Register"):
#             st.session_state.page = "Register"
#             st.rerun()

# def show_register_page():
#     st.title("Register")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     if st.button("Register"):
#         if register_user(username, password):
#             st.success("Registered successfully! Please go to Login.")
#         else:
#             st.error("User already exists!")
#     if st.button("Back to Login"):
#         st.session_state.page = "Login"
#         st.rerun()

# # Utility function for sending alerts
# def send_alert_email(subject, body):
#     try:
#         sender = "viraj.salunke23@spit.ac.in"
#         password = "lppqwukqzossvidg"
#         receiver = "virajsalunke12@gmail.com"
#         msg = MIMEText(body)
#         msg["Subject"] = subject
#         msg["From"] = sender
#         msg["To"] = receiver
#         with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
#             server.login(sender, password)
#             server.sendmail(sender, receiver, msg.as_string())
#         print("✅ Email sent")
#         return True
#     except Exception as e:
#         print(f"❌ Email error: {e}")
#         return False

# # Surveillance Dashboard Functions
# from import_face_recognition import load_known_faces, recognize_faces_live
# known_encodings, known_names = load_known_faces()

# # Initialize MediaPipe Pose and Drawing utilities
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# mp_draw = mp.solutions.drawing_utils

# # Initialize MediaPipe Face Detection
# mp_face = mp.solutions.face_detection
# face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# # Load YOLOv8 model globally
# try:
#     yolo_model = YOLO("yolov8n.pt")
# except Exception as e:
#     st.error(f"Error loading YOLO model: {e}")
#     st.stop()

# def load_fight_model():
#   try:
#       # Fixing the file path
#       return load_model(r'C:\Users\kriya\Desktop\face\lstm-fight-detection.h5')
#   except Exception as e:
#       st.error(f"Error loading model: {e}")
#       st.stop()

# model = load_fight_model()

# # Global variables for fight detection
# label = "unknown"
# label_history = []
# lm_list = []

# # WebRTC configuration (use default STUN server)
# RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


# mp_face = mp.solutions.face_detection
# face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# class ViolenceProcessor(VideoProcessorBase):
#   def __init__(self):
#       self.label = "unknown"
#       self.label_history = []
#       self.lm_list = []
#       self.fight_start_time = None
#       self.alert_sent = False

#   def make_landmark_timestep(self, results):
#       if results.pose_landmarks:
#           c_lm = []
#           for lm in results.pose_landmarks.landmark:
#               c_lm.extend([lm.x, lm.y, lm.z])
#           return c_lm
#       return None

#   def draw_landmark_on_image(self, mp_draw, results, frame):
#       if results.pose_landmarks:
#           mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#           for id, lm in enumerate(results.pose_landmarks.landmark):
#               h, w, c = frame.shape
#               cx, cy = int(lm.x * w), int(lm.y * h)
#               cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
#       return frame

#   def detect(self, model, lm_list):
#       if len(lm_list) >= 20:
#           lm_array = np.array(lm_list[-20:])
#           lm_array = np.expand_dims(lm_array, axis=0)
#           try:
#               result = model.predict(lm_array, verbose=0)[0]
#               pred_label = "fight" if result[0] > 0.5 else "normal"
#               for idx in [15, 16, 13, 14]:
#                   x_diff = lm_list[-1][idx*3] - lm_list[-2][idx*3]
#                   y_diff = lm_list[-1][idx*3+1] - lm_list[-2][idx*3+1]
#                   if abs(x_diff) > 0.07 or abs(y_diff) > 0.07:
#                       pred_label = "fight"
#                       break
#               self.label_history.append(pred_label)
#               if len(self.label_history) > 3:
#                   self.label_history.pop(0)
#               self.label = max(set(self.label_history), key=self.label_history.count)
#           except Exception as e:
#               st.error(f"Prediction error: {e}")
#               self.label = "ERROR"
#       return self.label

#     def send_alert_email(self):
#         try:
#             sender = "viraj.salunke23@spit.ac.in"
#             password = "lppqwukqzossvidg"
#             receiver = "virajsalunke12@gmail.com"
#             subject = "⚠️ Fight Detected Alert"
#             body = f"⚠️ A violent action was detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} in CCTV camera feed."

#             msg = MIMEText(body)
#             msg["Subject"] = subject
#             msg["From"] = sender
#             msg["To"] = receiver

#             with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
#                 server.login(sender, password)
#                 server.sendmail(sender, receiver, msg.as_string())
#             print("✅ Email sent")
#         except Exception as e:
#             print(f"❌ Email error: {e}")

#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = pose.process(frame_rgb)
#         face_results = face_detection.process(frame_rgb)

#         lm = self.make_landmark_timestep(results)
#         if lm:
#             self.lm_list.append(lm)

#         img = self.draw_landmark_on_image(mp_draw, results, img)

#         # Check user role for face blurring
#         user = get_logged_user()
#         is_admin = user and user.get('role') == 'admin'
#         if face_results.detections and not is_admin:
#             for detection in face_results.detections:
#                 bbox = detection.location_data.relative_bounding_box
#                 h, w, _ = img.shape
#                 x1 = int(bbox.xmin * w)
#                 y1 = int(bbox.ymin * h)
#                 x2 = x1 + int(bbox.width * w)
#                 y2 = y1 + int(bbox.height * h)
#                 x1, y1 = max(0, x1), max(0, y1)
#                 x2, y2 = min(w, x2), min(h, y2)
#                 face_region = img[y1:y2, x1:x2]
#                 if face_region.size > 0:
#                     blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
#                     img[y1:y2, x1:x2] = blurred

#         self.label = self.detect(fight_model, self.lm_list)
#         color = (0, 0, 255) if self.label == "fight" else (0, 255, 0)
#         cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color, 2)
#         cv2.putText(img, self.label.upper(), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#         # 5-second alert logic
#         detection_key = "fight_detection"
#         if self.label == "fight":
#             if detection_key not in st.session_state.detection_states:
#                 st.session_state.detection_states[detection_key] = {
#                     "start_time": datetime.now(),
#                     "alert_sent": False
#                 }
#             elif (datetime.now() - st.session_state.detection_states[detection_key]["start_time"] > timedelta(seconds=5)
#                   and not st.session_state.detection_states[detection_key]["alert_sent"]):
#                 alert_message = f"Fight detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
#                 if send_alert_email("⚠️ Fight Detected Alert", alert_message):
#                     st.session_state.alerts.append([datetime.now(), "Fight Detection", alert_message])
#                     st.session_state.detection_states[detection_key]["alert_sent"] = True
#         else:
#             if detection_key in st.session_state.detection_states:
#                 del st.session_state.detection_states[detection_key]

#         return av.VideoFrame.from_ndarray(img, format="bgr24")

# def process_video_with_webrtc():
#     st.info("Initializing Fight Detection video feed...")
#     try:
#         ctx = webrtc_streamer(
#             key="violence-detection",
#             video_processor_factory=ViolenceProcessor,
#             rtc_configuration=RTC_CONFIGURATION,
#             media_stream_constraints={"video": True, "audio": False},
#             async_processing=True
#         )
#         if ctx.video_processor:
#             st.success(f"Fight Detection status: {ctx.video_processor.label.upper()}")
#             return ctx.video_processor.label
#         else:
#             st.error("WebRTC video processor not initialized. Please ensure webcam access is granted.")
#             return "unknown"
#     except Exception as e:
#         st.error(f"WebRTC error: {e}")
#         return "error"

# # Gender and Age Detection Initialization
# GENDER_LIST = ['Male', 'Female']
# AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# gender_history = deque(maxlen=5)
# age_history = deque(maxlen=5)

# def load_gender_age_model():
#     model_path = os.path.join("Gender_and_age_Detection", "models")
#     faceProto = os.path.join(model_path, "opencv_face_detector.pbtxt")
#     faceModel = os.path.join(model_path, "opencv_face_detector_uint8.pb")
#     genderProto = os.path.join(model_path, "gender_deploy.prototxt")
#     genderModel = os.path.join(model_path, "gender_net.caffemodel")
#     ageProto = os.path.join(model_path, "age_deploy.prototxt")
#     ageModel = os.path.join(model_path, "age_net.caffemodel")

#     try:
#         face_net = cv2.dnn.readNet(faceModel, faceProto)
#         gender_net = cv2.dnn.readNet(genderModel, genderProto)
#         age_net = cv2.dnn.readNet(ageModel, ageProto)
#     except Exception as e:
#         st.error(f"Error loading gender/age model: {e}")
#         st.stop()

#     return face_net, gender_net, age_net

# face_net, gender_net, age_net = load_gender_age_model()

# def preprocess_face(face):
#     try:
#         face_resized = cv2.resize(face, (227, 227), interpolation=cv2.INTER_CUBIC)
#         mean = [78.4263377603, 87.7689143744, 114.895847746]
#         blob = cv2.dnn.blobFromImage(
#             face_resized,
#             scalefactor=1.0,
#             size=(227, 227),
#             mean=mean,
#             swapRB=False,
#             crop=False
#         )
#         return blob
#     except Exception as e:
#         st.warning(f"Face preprocessing error: {e}")
#         return None

# def get_smoothed_prediction(predictions, history, labels):
#     pred_idx = np.argmax(predictions)
#     history.append(pred_idx)
#     if len(history) < history.maxlen:
#         return labels[pred_idx]
#     counts = np.bincount(list(history), minlength=len(labels))
#     smoothed_idx = np.argmax(counts)
#     return labels[smoothed_idx]

# def blur_faces(frame, is_admin):
#     if is_admin:
#         return frame
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_results = face_detection.process(frame_rgb)
#     if face_results.detections:
#         for detection in face_results.detections:
#             bbox = detection.location_data.relative_bounding_box
#             h, w, _ = frame.shape
#             x1 = int(bbox.xmin * w)
#             y1 = int(bbox.ymin * h)
#             x2 = x1 + int(bbox.width * w)
#             y2 = y1 + int(bbox.height * h)
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(w, x2), min(h, y2)
#             face_region = frame[y1:y2, x1:x2]
#             if face_region.size > 0:
#                 blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
#                 frame[y1:y2, x1:x2] = blurred
#     return frame

# def detect_gender_age(frame):
#     try:
#         alert = None
#         padding = 30
#         min_confidence = 0.85
        
#         blob = cv2.dnn.blobFromImage(
#             frame,
#             scalefactor=1.0,
#             size=(300, 300),
#             mean=[104, 117, 123],
#             swapRB=False,
#             crop=False
#         )
#         face_net.setInput(blob)
#         detections = face_net.forward()

#         h, w = frame.shape[:2]
#         annotated_frame = frame.copy()

#         gender_age_detected = False
#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence > min_confidence:
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 x1, y1, x2, y2 = box.astype("int")
#                 x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
#                 x2, y2 = min(w - 1, x2 + padding), min(h - 1, y2 + padding)
#                 face = frame[y1:y2, x1:x2]
#                 if face.size == 0 or face.shape[0] < 30 or face.shape[1] < 30:
#                     continue
#                 face_blob = preprocess_face(face)
#                 if face_blob is None:
#                     continue
#                 # Gender prediction
#                 gender_net.setInput(face_blob)
#                 gender_preds = gender_net.forward()
#                 gender = get_smoothed_prediction(gender_preds[0], gender_history, GENDER_LIST)
#                 # Age prediction
#                 age_net.setInput(face_blob)
#                 age_preds = age_net.forward()
#                 age = get_smoothed_prediction(age_preds[0], age_history, AGE_LIST)
#                 label = f"{gender}, {age}"
#                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                 cv2.putText(
#                     annotated_frame,
#                     label,
#                     (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8,
#                     (0, 255, 0),
#                     2
#                 )
#                 alert = f"{gender}, {age} detected"
#                 gender_age_detected = True

#         # 5-second alert logic
#         detection_key = "gender_age_detection"
#         if gender_age_detected:
#             if detection_key not in st.session_state.detection_states:
#                 st.session_state.detection_states[detection_key] = {
#                     "start_time": datetime.now(),
#                     "alert_sent": False
#                 }
#             elif (datetime.now() - st.session_state.detection_states[detection_key]["start_time"] > timedelta(seconds=5)
#                   and not st.session_state.detection_states[detection_key]["alert_sent"]):
#                 alert_message = f"Gender/Age ({alert}) detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
#                 if send_alert_email("⚠️ Gender/Age Detection Alert", alert_message):
#                     st.session_state.alerts.append([datetime.now(), "Gender/Age Detection", alert_message])
#                     st.session_state.detection_states[detection_key]["alert_sent"] = True
#         else:
#             if detection_key in st.session_state.detection_states:
#                 del st.session_state.detection_states[detection_key]

#         user = get_logged_user()
#         is_admin = user and user.get('role') == 'admin'
#         annotated_frame = blur_faces(annotated_frame, is_admin)
        
#         return annotated_frame, alert

#     except Exception as e:
#         st.error(f"Gender/Age detection error: {e}")
#         return frame, None

# def detect_guard_attentiveness(frame):
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(frame_rgb)
#     alert = None

#     if results.pose_landmarks:
#         landmarks = results.pose_landmarks.landmark
#         try:
#             nose = landmarks[0]
#             left_eye = landmarks[1]
#             right_eye = landmarks[2]
#             left_shoulder = landmarks[11]
#             right_shoulder = landmarks[12]

#             avg_eye_y = (left_eye.y + right_eye.y) / 2
#             avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

#             if (nose.y - avg_shoulder_y) > -0.05 or (avg_eye_y - avg_shoulder_y) > -0.05:
#                 alert = "Sleepy/drowsy pose detected!"
#         except IndexError:
#             pass
#         mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#     # 5-second alert logic
#     detection_key = "guard_attentiveness"
#     if alert:
#         if detection_key not in st.session_state.detection_states:
#             st.session_state.detection_states[detection_key] = {
#                 "start_time": datetime.now(),
#                 "alert_sent": False
#             }
#         elif (datetime.now() - st.session_state.detection_states[detection_key]["start_time"] > timedelta(seconds=5)
#               and not st.session_state.detection_states[detection_key]["alert_sent"]):
#             alert_message = f"Sleepy/drowsy pose detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
#             if send_alert_email("⚠️ Guard Attentiveness Alert", alert_message):
#                 st.session_state.alerts.append([ hemoglobin.now(), "Guard Attentiveness", alert_message])
#                 st.session_state.detection_states[detection_key]["alert_sent"] = True
#     else:
#         if detection_key in st.session_state.detection_states:
#             del st.session_state.detection_states[detection_key]

#     if alert:
#         h, w, _ = frame.shape
#         cv2.putText(frame, alert, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         st.session_state.alerts.append([datetime.now(), "Guard Attentiveness", alert])

#     return frame, alert

# def detect_face(frame):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
#     results = face_detector.process(rgb_frame)

#     face_names = []
#     face_locations = []

#     if results.detections:
#         for detection in results.detections:
#             bbox = detection.location_data.relative_bounding_box
#             ih, iw, _ = frame.shape
#             x = max(0, int(bbox.xmin * iw))
#             y = max(0, int(bbox.ymin * ih))
#             w = int(bbox.width * iw)
#             h = int(bbox.height * ih)
#             top, right, bottom, left = y, x + w, y + h, x
#             face_locations.append((top, right, bottom, left))

#         encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#         for face_encoding in encodings:
#             name = "Unknown"
#             distances = face_recognition.face_distance(known_encodings, face_encoding)
#             if len(distances) > 0:
#                 best_match_index = np.argmin(distances)
#                 if distances[best_match_index] < 0.45:
#                     name = known_names[best_match_index]
#             face_names.append(name)

#     alert = None
#     if "Unknown" in face_names:
#         alert = "Unknown face detected!"

#     # 5-second alert logic
#     detection_key = "face_detection"
#     if alert:
#         if detection_key not in st.session_state.detection_states:
#             st.session_state.detection_states[detection_key] = {
#                 "start_time": datetime.now(),
#                 "alert_sent": False
#             }
#         elif (datetime.now() - st.session_state.detection_states[detection_key]["start_time"] > timedelta(seconds=5)
#               and not st.session_state.detection_states[detection_key]["alert_sent"]):
#             alert_message = f"Unknown face detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
#             if send_alert_email("⚠️ Face Detection Alert", alert_message):
#                 st.session_state.alerts.append([datetime.now(), "Face Detection", alert_message])
#                 st.session_state.detection_states[detection_key]["alert_sent"] = True
#     else:
#         if detection_key in st.session_state.detection_states:
#             del st.session_state.detection_states[detection_key]

#     if alert:
#         st.session_state.alerts.append([datetime.now(), "Face Detection", alert])

#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
#         cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
#         cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
#         cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
#     return frame, alert

# def detect_objects(frame):
#     user = get_logged_user()
#     is_admin = user and user.get('role') == 'admin'
    
#     frame = blur_faces(frame, is_admin)
    
#     try:
#         results = yolo_model(frame, verbose=False, conf=0.5)
#         result = results[0]
#         annotated_frame = result.plot(
#             boxes=True,
#             labels=True,
#             conf=True,
#             font_size=12,
#             line_width=2
#         )
#         class_ids = [int(box.cls) for box in result.boxes]
#         class_counts = Counter(result.names[cls_id] for cls_id in class_ids)
#         detection_summary = ", ".join(f"{count} {name}" for name, count in class_counts.items())
#         if not detection_summary:
#             detection_summary = "No objects detected"
        
#         alerts = []
#         dangerous_items = ["cell phone", "bottle"]  # "marker" not in COCO; use custom model if needed
#         if not any(name in result.names.values() for name in ["pen", "marker"]):
#             st.warning("Note: 'marker' not in YOLOv8 COCO classes. Consider custom training for marker detection.")

#         for box in result.boxes:
#             class_name = result.names[int(box.cls)]
#             if class_name == "person" and box.conf > 0.5:
#                 alerts.append(f"{class_name.capitalize()} detected!")
#             elif class_name in dangerous_items and box.conf > 0.5:
#                 alerts.append(f"Dangerous item ({class_name}) detected!")

#         # 5-second alert logic for each detected item
#         for alert in alerts:
#             detection_key = f"object_detection_{alert}"
#             if detection_key not in st.session_state.detection_states:
#                 st.session_state.detection_states[detection_key] = {
#                     "start_time": datetime.now(),
#                     "alert_sent": False
#                 }
#             elif (datetime.now() - st.session_state.detection_states[detection_key]["start_time"] > timedelta(seconds=5)
#                   and not st.session_state.detection_states[detection_key]["alert_sent"]):
#                 alert_message = f"{alert} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
#                 if send_alert_email(f"⚠️ Object Detection Alert: {alert.split()[0]}", alert_message):
#                     st.session_state.alerts.append([datetime.now(), "Object Detection", alert_message])
#                     st.session_state.detection_states[detection_key]["alert_sent"] = True

#         # Clean up detection states for non-detected items
#         for key in list(st.session_state.detection_states.keys()):
#             if key.startswith("object_detection_") and not any(key.endswith(alert) for alert in alerts):
#                 del st.session_state.detection_states[key]

#         for alert in alerts:
#             st.session_state.alerts.append([datetime.now(), "Object Detection", alert])
        
#         return annotated_frame, "; ".join(alerts) if alerts else None, detection_summary
#     except Exception as e:
#         st.error(f"Object detection error: {e}")
#         return frame, None, "Detection failed"

# def process_video(model_func, model_name):
#     stframe = st.empty()
#     alert_placeholder = st.empty()
#     summary_placeholder = st.empty()
#     cap = cv2.VideoCapture(1)

#     if not cap.isOpened():
#         st.error("Failed to open webcam.")
#         return

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Failed to capture video feed.")
#             break

#         if model_func == detect_objects:
#             result_frame, alert, detection_summary = model_func(frame)
#             summary_placeholder.text(f"Detections: {detection_summary}")
#         else:
#             result_frame, alert = model_func(frame)
#             summary_placeholder.text("")

#         frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
#         stframe.image(frame_rgb, channels="RGB", use_column_width=True)

#         if alert:
#             alert_placeholder.error(f"ALERT: {alert} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#         else:
#             alert_placeholder.success(f"No issues detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

#     cap.release()

# def show_dashboard():
#     user = get_logged_user()
#     if not user or not st.session_state.token:
#         st.warning("Please login first")
#         st.session_state.page = "Login"
#         st.session_state.token = None
#         st.rerun()
#         return

#     st.title("AI Surveillance System")
#     st.success(f"Welcome, {user['username']} ({user['role']})")

#     if user["role"] == "admin":
#         st.write("🛡️ Admin Panel: Full access to surveillance features")
#     else:
#         st.write("👤 User Dashboard: Limited access to surveillance features")

#     if st.button("Logout"):
#         st.session_state.pop("token", None)
#         st.session_state.page = "Login"
#         st.session_state.alerts = []
#         st.session_state.detection_states = {}
#         st.success("Logged out successfully!")
#         st.rerun()

#     options = ["Face Recognition", "Fight Detection", "Object Detection", "Guard Attentiveness", "Gender/Age Detection"]
#     choice = st.sidebar.selectbox("Select Model", options)

#     if choice == "Face Recognition":
#         process_video(detect_face, "face_recognition")
#     elif choice == "Fight Detection":
#         process_video_with_webrtc()
#     elif choice == "Object Detection":
#         process_video(detect_objects, "object_detection")
#     elif choice == "Guard Attentiveness":
#         process_video(detect_guard_attentiveness, "guard_attentiveness")
#     elif choice == "Gender/Age Detection":
#         process_video(detect_gender_age, "gender_age_detection")

# def main():
#     if st.session_state.page == "Login":
#         show_login_page()
#     elif st.session_state.page == "Register":
#         show_register_page()
#     elif st.session_state.page == "Dashboard":
#         show_dashboard()

# if __name__ == "__main__":
#     main()


import streamlit as st
import cv2
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import os
import mediapipe as mp
import face_recognition
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import smtplib
from email.mime.text import MIMEText
import sys
from ultralytics import YOLO
from collections import Counter, deque
from auth import login_user, register_user
from utils import get_logged_user

# Add the Gender_and_age_Detection folder to Python's path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Gender_and_age_Detection'))

# Initialize session state
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "page" not in st.session_state:
    st.session_state.page = "Login"
if "token" not in st.session_state:
    st.session_state.token = None
if "detection_states" not in st.session_state:
    st.session_state.detection_states = {}

# Set page config
st.set_page_config(page_title="AI Surveillance System", layout="wide")

def show_login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            token = login_user(username, password)
            if token:
                st.session_state.token = token
                st.session_state.page = "Dashboard"
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    with col2:
        if st.button("Go to Register"):
            st.session_state.page = "Register"
            st.rerun()

def show_register_page():
    st.title("Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        if register_user(username, password):
            st.success("Registered successfully! Please go to Login.")
        else:
            st.error("User already exists!")
    if st.button("Back to Login"):
        st.session_state.page = "Login"
        st.rerun()

# Utility function for sending alerts
def send_alert_email(subject, body):
    try:
        sender = "viraj.salunke23@spit.ac.in"
        password = "lppqwukqzossvidg"
        receiver = "virajsalunke12@gmail.com"
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = receiver
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print("✅ Email sent")
        return True
    except Exception as e:
        print(f"❌ Email error: {e}")
        return False

# Surveillance Dashboard Functions
from import_face_recognition import load_known_faces, recognize_faces_live
known_encodings, known_names = load_known_faces()

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize MediaPipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Load YOLOv8 model globally
try:
    yolo_model = YOLO("yolov8n.pt")
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop()

# Load fight detection model
def load_fight_model():
    try:
        return load_model(r'C:\Users\kriya\Desktop\face\lstm-fight-detection.h5')
    except Exception as e:
        st.error(f"Error loading fight model: {e}")
        st.stop()

fight_model = load_fight_model()

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Violence Processor for Fight Detection
class ViolenceProcessor(VideoProcessorBase):
    def __init__(self):
        self.label = "unknown"
        self.label_history = []
        self.lm_list = []
        self.fight_start_time = None
        self.alert_sent = False

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
                for idx in [15, 16, 13, 14]:
                    x_diff = lm_list[-1][idx*3] - lm_list[-2][idx*3]
                    y_diff = lm_list[-1][idx*3+1] - lm_list[-2][idx*3+1]
                    if abs(x_diff) > 0.07 or abs(y_diff) > 0.07:
                        pred_label = "fight"
                        break
                self.label_history.append(pred_label)
                if len(self.label_history) > 3:
                    self.label_history.pop(0)
                self.label = max(set(self.label_history), key=self.label_history.count)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                self.label = "ERROR"
        return self.label

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        face_results = face_detection.process(frame_rgb)

        lm = self.make_landmark_timestep(results)
        if lm:
            self.lm_list.append(lm)

        img = self.draw_landmark_on_image(mp_draw, results, img)

        # Check user role for face blurring
        user = get_logged_user()
        is_admin = user and user.get('role') == 'admin'
        if face_results.detections and not is_admin:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = x1 + int(bbox.width * w)
                y2 = y1 + int(bbox.height * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face_region = img[y1:y2, x1:x2]
                if face_region.size > 0:
                    blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
                    img[y1:y2, x1:x2] = blurred

        self.label = self.detect(fight_model, self.lm_list)
        color = (0, 0, 255) if self.label == "fight" else (0, 255, 0)
        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color, 2)
        cv2.putText(img, self.label.upper(), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 5-second alert logic
        detection_key = "fight_detection"
        if self.label == "fight":
            if detection_key not in st.session_state.detection_states:
                st.session_state.detection_states[detection_key] = {
                    "start_time": datetime.now(),
                    "alert_sent": False
                }
            elif (datetime.now() - st.session_state.detection_states[detection_key]["start_time"] > timedelta(seconds=5)
                  and not st.session_state.detection_states[detection_key]["alert_sent"]):
                alert_message = f"Fight detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                if send_alert_email("⚠️ Fight Detected Alert", alert_message):
                    st.session_state.alerts.append([datetime.now(), "Fight Detection", alert_message])
                    st.session_state.detection_states[detection_key]["alert_sent"] = True
        else:
            if detection_key in st.session_state.detection_states:
                del st.session_state.detection_states[detection_key]

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def process_video_with_webrtc():
    st.info("Initializing Fight Detection video feed...")
    try:
        ctx = webrtc_streamer(
            key="violence-detection",
            video_processor_factory=ViolenceProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
        if ctx.video_processor:
            st.success(f"Fight Detection status: {ctx.video_processor.label.upper()}")
            return ctx.video_processor.label
        else:
            st.error("WebRTC video processor not initialized. Please ensure webcam access is granted.")
            return "unknown"
    except Exception as e:
        st.error(f"WebRTC error: {e}")
        return "error"

# Gender and Age Detection Initialization
GENDER_LIST = ['Male', 'Female']
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_history = deque(maxlen=5)
age_history = deque(maxlen=5)

def load_gender_age_model():
    model_path = os.path.join("Gender_and_age_Detection", "models")
    faceProto = os.path.join(model_path, "opencv_face_detector.pbtxt")
    faceModel = os.path.join(model_path, "opencv_face_detector_uint8.pb")
    genderProto = os.path.join(model_path, "gender_deploy.prototxt")
    genderModel = os.path.join(model_path, "gender_net.caffemodel")
    ageProto = os.path.join(model_path, "age_deploy.prototxt")
    ageModel = os.path.join(model_path, "age_net.caffemodel")

    try:
        face_net = cv2.dnn.readNet(faceModel, faceProto)
        gender_net = cv2.dnn.readNet(genderModel, genderProto)
        age_net = cv2.dnn.readNet(ageModel, ageProto)
    except Exception as e:
        st.error(f"Error loading gender/age model: {e}")
        st.stop()

    return face_net, gender_net, age_net

face_net, gender_net, age_net = load_gender_age_model()

def preprocess_face(face):
    try:
        face_resized = cv2.resize(face, (227, 227), interpolation=cv2.INTER_CUBIC)
        mean = [78.4263377603, 87.7689143744, 114.895847746]
        blob = cv2.dnn.blobFromImage(
            face_resized,
            scalefactor=1.0,
            size=(227, 227),
            mean=mean,
            swapRB=False,
            crop=False
        )
        return blob
    except Exception as e:
        st.warning(f"Face preprocessing error: {e}")
        return None

def get_smoothed_prediction(predictions, history, labels):
    pred_idx = np.argmax(predictions)
    history.append(pred_idx)
    if len(history) < history.maxlen:
        return labels[pred_idx]
    counts = np.bincount(list(history), minlength=len(labels))
    smoothed_idx = np.argmax(counts)
    return labels[smoothed_idx]

def blur_faces(frame, is_admin):
    if is_admin:
        return frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_detection.process(frame_rgb)
    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = x1 + int(bbox.width * w)
            y2 = y1 + int(bbox.height * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face_region = frame[y1:y2, x1:x2]
            if face_region.size > 0:
                blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
                frame[y1:y2, x1:x2] = blurred
    return frame

def detect_gender_age(frame):
    try:
        alert = None
        padding = 30
        min_confidence = 0.85
        
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0,
            size=(300, 300),
            mean=[104, 117, 123],
            swapRB=False,
            crop=False
        )
        face_net.setInput(blob)
        detections = face_net.forward()

        h, w = frame.shape[:2]
        annotated_frame = frame.copy()

        gender_age_detected = False
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > min_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                x2, y2 = min(w - 1, x2 + padding), min(h - 1, y2 + padding)
                face = frame[y1:y2, x1:x2]
                if face.size == 0 or face.shape[0] < 30 or face.shape[1] < 30:
                    continue
                face_blob = preprocess_face(face)
                if face_blob is None:
                    continue
                # Gender prediction
                gender_net.setInput(face_blob)
                gender_preds = gender_net.forward()
                gender = get_smoothed_prediction(gender_preds[0], gender_history, GENDER_LIST)
                # Age prediction
                age_net.setInput(face_blob)
                age_preds = age_net.forward()
                age = get_smoothed_prediction(age_preds[0], age_history, AGE_LIST)
                label = f"{gender}, {age}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                alert = f"{gender}, {age} detected"
                gender_age_detected = True

        # 5-second alert logic
        detection_key = "gender_age_detection"
        if gender_age_detected:
            if detection_key not in st.session_state.detection_states:
                st.session_state.detection_states[detection_key] = {
                    "start_time": datetime.now(),
                    "alert_sent": False
                }
            elif (datetime.now() - st.session_state.detection_states[detection_key]["start_time"] > timedelta(seconds=5)
                  and not st.session_state.detection_states[detection_key]["alert_sent"]):
                alert_message = f"Gender/Age ({alert}) detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                if send_alert_email("⚠️ Gender/Age Detection Alert", alert_message):
                    st.session_state.alerts.append([datetime.now(), "Gender/Age Detection", alert_message])
                    st.session_state.detection_states[detection_key]["alert_sent"] = True
        else:
            if detection_key in st.session_state.detection_states:
                del st.session_state.detection_states[detection_key]

        user = get_logged_user()
        is_admin = user and user.get('role') == 'admin'
        annotated_frame = blur_faces(annotated_frame, is_admin)
        
        return annotated_frame, alert

    except Exception as e:
        st.error(f"Gender/Age detection error: {e}")
        return frame, None

def detect_guard_attentiveness(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    alert = None

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        try:
            nose = landmarks[0]
            left_eye = landmarks[1]
            right_eye = landmarks[2]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]

            avg_eye_y = (left_eye.y + right_eye.y) / 2
            avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

            if (nose.y - avg_shoulder_y) > -0.05 or (avg_eye_y - avg_shoulder_y) > -0.05:
                alert = "Sleepy/drowsy pose detected!"
        except IndexError:
            pass
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 5-second alert logic
    detection_key = "guard_attentiveness"
    if alert:
        if detection_key not in st.session_state.detection_states:
            st.session_state.detection_states[detection_key] = {
                "start_time": datetime.now(),
                "alert_sent": False
            }
        elif (datetime.now() - st.session_state.detection_states[detection_key]["start_time"] > timedelta(seconds=5)
              and not st.session_state.detection_states[detection_key]["alert_sent"]):
            alert_message = f"Sleepy/drowsy pose detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            if send_alert_email("⚠️ Guard Attentiveness Alert", alert_message):
                st.session_state.alerts.append([datetime.now(), "Guard Attentiveness", alert_message])
                st.session_state.detection_states[detection_key]["alert_sent"] = True
    else:
        if detection_key in st.session_state.detection_states:
            del st.session_state.detection_states[detection_key]

    if alert:
        h, w, _ = frame.shape
        cv2.putText(frame, alert, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        st.session_state.alerts.append([datetime.now(), "Guard Attentiveness", alert])

    return frame, alert

def detect_face(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    results = face_detector.process(rgb_frame)

    face_names = []
    face_locations = []

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = max(0, int(bbox.xmin * iw))
            y = max(0, int(bbox.ymin * ih))
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)
            top, right, bottom, left = y, x + w, y + h, x
            face_locations.append((top, right, bottom, left))

        encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for face_encoding in encodings:
            name = "Unknown"
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                if distances[best_match_index] < 0.45:
                    name = known_names[best_match_index]
            face_names.append(name)

    alert = None
    if "Unknown" in face_names:
        alert = "Unknown face detected!"

    # 5-second alert logic
    detection_key = "face_detection"
    if alert:
        if detection_key not in st.session_state.detection_states:
            st.session_state.detection_states[detection_key] = {
                "start_time": datetime.now(),
                "alert_sent": False
            }
        elif (datetime.now() - st.session_state.detection_states[detection_key]["start_time"] > timedelta(seconds=5)
              and not st.session_state.detection_states[detection_key]["alert_sent"]):
            alert_message = f"Unknown face detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            if send_alert_email("⚠️ Face Detection Alert", alert_message):
                st.session_state.alerts.append([datetime.now(), "Face Detection", alert_message])
                st.session_state.detection_states[detection_key]["alert_sent"] = True
    else:
        if detection_key in st.session_state.detection_states:
            del st.session_state.detection_states[detection_key]

    if alert:
        st.session_state.alerts.append([datetime.now(), "Face Detection", alert])

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame, alert

def detect_objects(frame):
    user = get_logged_user()
    is_admin = user and user.get('role') == 'admin'
    
    frame = blur_faces(frame, is_admin)
    
    try:
        results = yolo_model(frame, verbose=False, conf=0.5)
        result = results[0]
        annotated_frame = result.plot(
            boxes=True,
            labels=True,
            conf=True,
            font_size=12,
            line_width=2
        )
        class_ids = [int(box.cls) for box in result.boxes]
        class_counts = Counter(result.names[cls_id] for cls_id in class_ids)
        detection_summary = ", ".join(f"{count} {name}" for name, count in class_counts.items())
        if not detection_summary:
            detection_summary = "No objects detected"
        
        alerts = []
        dangerous_items = ["cell phone", "bottle"]  # "marker" not in COCO; use custom model if needed
        if not any(name in result.names.values() for name in ["pen", "marker"]):
            st.warning("Note: 'marker' not in YOLOv8 COCO classes. Consider custom training for marker detection.")

        for box in result.boxes:
            class_name = result.names[int(box.cls)]
            if class_name == "person" and box.conf > 0.5:
                alerts.append(f"{class_name.capitalize()} detected!")
            elif class_name in dangerous_items and box.conf > 0.5:
                alerts.append(f"Dangerous item ({class_name}) detected!")

        # 5-second alert logic for each detected item
        for alert in alerts:
            detection_key = f"object_detection_{alert}"
            if detection_key not in st.session_state.detection_states:
                st.session_state.detection_states[detection_key] = {
                    "start_time": datetime.now(),
                    "alert_sent": False
                }
            elif (datetime.now() - st.session_state.detection_states[detection_key]["start_time"] > timedelta(seconds=5)
                  and not st.session_state.detection_states[detection_key]["alert_sent"]):
                alert_message = f"{alert} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                if send_alert_email(f"⚠️ Object Detection Alert: {alert.split()[0]}", alert_message):
                    st.session_state.alerts.append([datetime.now(), "Object Detection", alert_message])
                    st.session_state.detection_states[detection_key]["alert_sent"] = True

        # Clean up detection states for non-detected items
        for key in list(st.session_state.detection_states.keys()):
            if key.startswith("object_detection_") and not any(key.endswith(alert) for alert in alerts):
                del st.session_state.detection_states[key]

        for alert in alerts:
            st.session_state.alerts.append([datetime.now(), "Object Detection", alert])
        
        return annotated_frame, "; ".join(alerts) if alerts else None, detection_summary
    except Exception as e:
        st.error(f"Object detection error: {e}")
        return frame, None, "Detection failed"

def process_video(model_func, model_name):
    stframe = st.empty()
    alert_placeholder = st.empty()
    summary_placeholder = st.empty()
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        st.error("Failed to open webcam.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video feed.")
            break

        if model_func == detect_objects:
            result_frame, alert, detection_summary = model_func(frame)
            summary_placeholder.text(f"Detections: {detection_summary}")
        else:
            result_frame, alert = model_func(frame)
            summary_placeholder.text("")

        frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        if alert:
            alert_placeholder.error(f"ALERT: {alert} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            alert_placeholder.success(f"No issues detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    cap.release()

def show_dashboard():
    user = get_logged_user()
    if not user or not st.session_state.token:
        st.warning("Please login first")
        st.session_state.page = "Login"
        st.session_state.token = None
        st.rerun()
        return

    st.title("AI Surveillance System")
    st.success(f"Welcome, {user['username']} ({user['role']})")

    if user["role"] == "admin":
        st.write("🛡️ Admin Panel: Full access to surveillance features")
    else:
        st.write("👤 User Dashboard: Limited access to surveillance features")

    if st.button("Logout"):
        st.session_state.pop("token", None)
        st.session_state.page = "Login"
        st.session_state.alerts = []
        st.session_state.detection_states = {}
        st.success("Logged out successfully!")
        st.rerun()

    options = ["Face Recognition", "Fight Detection", "Object Detection", "Guard Attentiveness", "Gender/Age Detection"]
    choice = st.sidebar.selectbox("Select Model", options)

    if choice == "Face Recognition":
        process_video(detect_face, "face_recognition")
    elif choice == "Fight Detection":
        process_video_with_webrtc()
    elif choice == "Object Detection":
        process_video(detect_objects, "object_detection")
    elif choice == "Guard Attentiveness":
        process_video(detect_guard_attentiveness, "guard_attentiveness")
    elif choice == "Gender/Age Detection":
        process_video(detect_gender_age, "gender_age_detection")

def main():
    if st.session_state.page == "Login":
        show_login_page()
    elif st.session_state.page == "Register":
        show_register_page()
    elif st.session_state.page == "Dashboard":
        show_dashboard()

if __name__ == "__main__":
    main()

