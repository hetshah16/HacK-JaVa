# import streamlit as st
# import cv2
# import numpy as np
# import time
# from datetime import datetime
# import pandas as pd
# import pickle
# import os
# import mediapipe as mp
# import face_recognition
# from tensorflow.keras.models import load_model
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
# import av
# import smtplib
# from email.mime.text import MIMEText
# import sys
# from datetime import timedelta
# from ultralytics import YOLO

# # sys.path.append(os.path.join(os.path.dirname(__file__), 'Gender_and_age_Detection'))

# # from Gender_and_age_Detection.detect import GenderAgeDetector


# import sys

# # Add the Gender_and_age_Detection folder to Python's path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'Gender_and_age_Detection'))

# if "alerts" not in st.session_state:
# 	st.session_state.alerts = []

# # Importing the necessary functions from the face detection file
# from import_face_recognition import load_known_faces, recognize_faces_live
# known_encodings, known_names = load_known_faces()

# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# mp_draw = mp.solutions.drawing_utils

# # Fixed model loading function with corrected path
# def load_fight_model():
# 	try:
# 		# Fixing the file path
# 		return load_model(r'C:\Users\kriya\Desktop\face\lstm-fight-detection.h5')
# 	except Exception as e:
# 		st.error(f"Error loading model: {e}")
# 		st.stop()

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
# 	def __init__(self):
# 		self.label = "unknown"
# 		self.label_history = []
# 		self.lm_list = []
# 		self.fight_start_time = None
# 		self.alert_sent = False

# 	def make_landmark_timestep(self, results):
# 		if results.pose_landmarks:
# 			c_lm = []
# 			for lm in results.pose_landmarks.landmark:
# 				c_lm.extend([lm.x, lm.y, lm.z])
# 			return c_lm
# 		return None

# 	def draw_landmark_on_image(self, mp_draw, results, frame):
# 		if results.pose_landmarks:
# 			mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
# 			for id, lm in enumerate(results.pose_landmarks.landmark):
# 				h, w, c = frame.shape
# 				cx, cy = int(lm.x * w), int(lm.y * h)
# 				cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
# 		return frame

# 	def detect(self, model, lm_list):
# 		if len(lm_list) >= 20:
# 			lm_array = np.array(lm_list[-20:])
# 			lm_array = np.expand_dims(lm_array, axis=0)
# 			try:
# 				result = model.predict(lm_array, verbose=0)[0]
# 				pred_label = "fight" if result[0] > 0.5 else "normal"
# 				for idx in [15, 16, 13, 14]:
# 					x_diff = lm_list[-1][idx*3] - lm_list[-2][idx*3]
# 					y_diff = lm_list[-1][idx*3+1] - lm_list[-2][idx*3+1]
# 					if abs(x_diff) > 0.07 or abs(y_diff) > 0.07:
# 						pred_label = "fight"
# 						break
# 				self.label_history.append(pred_label)
# 				if len(self.label_history) > 3:
# 					self.label_history.pop(0)
# 				self.label = max(set(self.label_history), key=self.label_history.count)
# 			except Exception as e:
# 				st.error(f"Prediction error: {e}")
# 				self.label = "ERROR"
# 		return self.label

# 	def send_alert_email(self):
# 		try:
# 			sender = "viraj.salunke23@spit.ac.in"
# 			password = "lppqwukqzossvidg"
# 			receiver = "virajsalunke12@gmail.com"
# 			subject = "⚠️ Fight Detected Alert"
# 			body = f"⚠️ A violent action was detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} in CCTV camera feed."

# 			msg = MIMEText(body)
# 			msg["Subject"] = subject
# 			msg["From"] = sender
# 			msg["To"] = receiver

# 			with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
# 				server.login(sender, password)
# 				server.sendmail(sender, receiver, msg.as_string())
# 			print("✅ Email sent")
# 		except Exception as e:
# 			print(f"❌ Email error: {e}")

# 	def recv(self, frame):
# 		img = frame.to_ndarray(format="bgr24")
# 		frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 		results = pose.process(frame_rgb)
# 		face_results = face_detection.process(frame_rgb)

# 		lm = self.make_landmark_timestep(results)
# 		if lm:
# 			self.lm_list.append(lm)

# 		img = self.draw_landmark_on_image(mp_draw, results, img)

# 		# Face blurring
# 		if face_results.detections:
# 			for detection in face_results.detections:
# 				bbox = detection.location_data.relative_bounding_box
# 				h, w, _ = img.shape
# 				x1 = int(bbox.xmin * w)
# 				y1 = int(bbox.ymin * h)
# 				x2 = x1 + int(bbox.width * w)
# 				y2 = y1 + int(bbox.height * h)
# 				x1, y1 = max(0, x1), max(0, y1)
# 				x2, y2 = min(w, x2), min(h, y2)
# 				face_region = img[y1:y2, x1:x2]
# 				if face_region.size > 0:
# 					blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
# 					img[y1:y2, x1:x2] = blurred

# 		self.label = self.detect(model, self.lm_list)
# 		color = (0, 0, 255) if self.label == "fight" else (0, 255, 0)
# 		cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color, 2)
# 		cv2.putText(img, self.label.upper(), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# 		if self.label == "fight":
# 			if self.fight_start_time is None:
# 				self.fight_start_time = datetime.now()
# 			elif datetime.now() - self.fight_start_time > timedelta(seconds=5) and not self.alert_sent:
# 				self.send_alert_email()
# 				self.alert_sent = True
# 		else:
# 			self.fight_start_time = None
# 			self.alert_sent = False

# 		return av.VideoFrame.from_ndarray(img, format="bgr24")

# def process_video_with_webrtc():
# 	ctx = webrtc_streamer(
# 		key="violence-detection",
# 		video_processor_factory=ViolenceProcessor,
# 		rtc_configuration=RTC_CONFIGURATION,
# 		media_stream_constraints={"video": True, "audio": False},
# 		async_processing=True
# 	)
# 	if ctx.video_processor:
# 		return ctx.video_processor.label
# 	return "unknown"

# # def load_gender_model():
# #     return GenderAgeDetector()

# # gender_model = load_gender_model()

# # import subprocess

# # def detect_gender(frame):
# #     try:
# #         # Run gender detection in a separate subprocess
# #         result = subprocess.run(["python", "Gender_and_Age_Detection/detect.py"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		
# #         # Capture output or errors from the subprocess if needed
# #         if result.returncode == 0:
# #             return None, "Gender & Age Detection started successfully in a separate window."
# #         else:
# #             return None, f"Error running gender detection: {result.stderr.decode()}"
# #     except subprocess.CalledProcessError as e:
# #         return None, f"Error executing the gender detection script: {e}"



# # Gender & Age Detection Initialization
# AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# GENDER_LIST = ['Male', 'Female']

# def load_gender_age_models():
# 	model_path = os.path.join("Gender_and_age_Detection", "models")
	
# 	faceProto = os.path.join(model_path, "opencv_face_detector.pbtxt")
# 	faceModel = os.path.join(model_path, "opencv_face_detector_uint8.pb")

# 	ageProto = os.path.join(model_path, "age_deploy.prototxt")
# 	ageModel = os.path.join(model_path, "age_net.caffemodel")

# 	genderProto = os.path.join(model_path, "gender_deploy.prototxt")
# 	genderModel = os.path.join(model_path, "gender_net.caffemodel")

# 	face_net = cv2.dnn.readNet(faceModel, faceProto)
# 	age_net = cv2.dnn.readNet(ageModel, ageProto)
# 	gender_net = cv2.dnn.readNet(genderModel, genderProto)

# 	return face_net, age_net, gender_net

# face_net, age_net, gender_net = load_gender_age_models()


# # def detect_guard_attentiveness(frame):
# # 	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# # 	results = pose.process(frame_rgb)
# # 	alert = None

# # 	if results.pose_landmarks:
# # 		landmarks = results.pose_landmarks.landmark

# # 		# Get coordinates for nose, left eye, right eye, and shoulders
# # 		try:
# # 			nose = landmarks[0]
# # 			left_eye = landmarks[1]
# # 			right_eye = landmarks[2]
# # 			left_shoulder = landmarks[11]
# # 			right_shoulder = landmarks[12]

# # 			avg_eye_y = (left_eye.y + right_eye.y) / 2
# # 			avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

# # 			# Check if nose and eyes are near the shoulder level (head drooping)
# # 			if nose.y > avg_shoulder_y or avg_eye_y > avg_shoulder_y:
# # 				alert = "Sleepy/drowsy pose detected!"
# # 				st.session_state.alerts.append([datetime.now(), "Sleepy Pose", alert])

# # 				# Draw alert on frame
# # 				h, w, _ = frame.shape
# # 				cv2.putText(frame, alert, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# # 		except IndexError:
# # 			pass

# # 		# Draw landmarks
# # 		mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# # 	return frame, alert



# def detect_guard_attentiveness(frame):
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(frame_rgb)
#     alert = None

#     if results.pose_landmarks:
#         st.write("Pose landmarks detected!")  # Debug
#         landmarks = results.pose_landmarks.landmark

#         try:
#             nose = landmarks[0]
#             left_eye = landmarks[1]
#             right_eye = landmarks[2]
#             left_shoulder = landmarks[11]
#             right_shoulder = landmarks[12]

#             avg_eye_y = (left_eye.y + right_eye.y) / 2
#             avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

#             st.write(f"Nose.y: {nose.y:.3f}, Eyes.y: {avg_eye_y:.3f}, Shoulders.y: {avg_shoulder_y:.3f}")

#             # More lenient threshold
#             if (nose.y - avg_shoulder_y) > -0.05 or (avg_eye_y - avg_shoulder_y) > -0.05:
#                 alert = "Sleepy/drowsy pose detected!"
#                 st.session_state.alerts.append([datetime.now(), "Sleepy Pose", alert])
#                 h, w, _ = frame.shape
#                 cv2.putText(frame, alert, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         except IndexError:
#             st.write("Landmark index error")

#         mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#     else:
#         st.write("No landmarks detected")

#     return frame, alert


# def detect_objects(frame):
#     """
#     Detects objects in the frame using YOLOv8 model.
	
#     Args:
#         frame: Input frame from the video feed (numpy array, BGR format)
	
#     Returns:
#         tuple: (annotated_frame, alert)
#             - annotated_frame: Frame with object detection annotations
#             - alert: String alert message if specific conditions are met, else None
#     """
#     # Load the YOLOv8 model (only once, ideally outside the function in production)
#     model = YOLO("yolov8n.pt")  # Replace with yolov8.pt or another variant if needed

#     # Run YOLO inference on the frame
#     results = model(frame)

#     # Access the first result
#     result = results[0]

#     # Plot detection results on the frame
#     annotated_frame = result.plot()  # Frame with bounding boxes and labels

#     # Placeholder for alert logic (customize based on requirements)
#     alert = None
#     # Example: Trigger alert if specific objects (e.g., "person") are detected
#     for box in result.boxes:
#         class_name = result.names[int(box.cls)]
#         if class_name == "person":  # Example condition
#             alert = f"{class_name} detected!"
#             st.session_state.alerts.append([datetime.now(), "Object Detection", alert])
#             break

#     return annotated_frame, alert

# def detect_gender_age(frame):
# 	alert = None
# 	padding = 20
# 	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [
# 								 104, 117, 123], swapRB=False)
# 	face_net.setInput(blob)
# 	detections = face_net.forward()

# 	h, w = frame.shape[:2]
# 	for i in range(detections.shape[2]):
# 		confidence = detections[0, 0, i, 2]
# 		if confidence > 0.7:
# 			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
# 			x1, y1, x2, y2 = box.astype("int")

# 			x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
# 			x2, y2 = min(w - 1, x2 + padding), min(h - 1, y2 + padding)

# 			face = frame[y1:y2, x1:x2]
# 			if face.size == 0:
# 				continue
# 			face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [
# 											  78.4263377603, 87.7689143744, 114.895847746], swapRB=False)
# 			gender_net.setInput(face_blob)
# 			gender_preds = gender_net.forward()
# 			gender = GENDER_LIST[gender_preds[0].argmax()]

# 			age_net.setInput(face_blob)
# 			age_preds = age_net.forward()
# 			age = AGE_LIST[age_preds[0].argmax()]

# 			label = f"{gender}, {age}"
# 			cv2.rectangle(frame, (x1, y1), (x2, y2),
# 						  (255, 0, 0), 2)
# 			cv2.putText(frame, label, (x1, y1 - 10),
# 						cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# 			alert = f"{label} detected"
# 			st.session_state.alerts.append([datetime.now(), "Gender Detection", label])

# 	return frame, alert






# # Load encodings once globally
# known_encodings, known_names = load_known_faces()

# def detect_face(frame):
# 	"""Detects faces in the frame using the custom face recognition model."""
# 	rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 	face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
# 	results = face_detector.process(rgb_frame)

# 	face_names = []
# 	face_locations = []

# 	if results.detections:
# 		for detection in results.detections:
# 			bbox = detection.location_data.relative_bounding_box
# 			ih, iw, _ = frame.shape
# 			x = max(0, int(bbox.xmin * iw))
# 			y = max(0, int(bbox.ymin * ih))
# 			w = int(bbox.width * iw)
# 			h = int(bbox.height * ih)
# 			top, right, bottom, left = y, x + w, y + h, x
# 			face_locations.append((top, right, bottom, left))

# 		encodings = face_recognition.face_encodings(rgb_frame, face_locations)
# 		for face_encoding in encodings:
# 			name = "Unknown"
# 			distances = face_recognition.face_distance(known_encodings, face_encoding)
# 			if len(distances) > 0:
# 				best_match_index = np.argmin(distances)
# 				if distances[best_match_index] < 0.45:
# 					name = known_names[best_match_index]
# 			face_names.append(name)

# 	alert = None
# 	if "Unknown" in face_names:
# 		alert = "Unknown face detected!"
# 		st.session_state.alerts.append([datetime.now(), "Face Detection", alert])

# 	# Draw bounding boxes
# 	for (top, right, bottom, left), name in zip(face_locations, face_names):
# 		color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
# 		cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
# 		cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
# 		cv2.putText(frame, name, (left + 6, bottom - 6),
# 					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# 	return frame, alert


# def detect_crowd_density(frame):
# 	"""Placeholder for crowd density monitoring model (fifth model)."""
# 	density = np.random.randint(0, 100)
# 	alert = "High crowd density detected" if density > 80 else None
# 	return density, alert

# # Function to process video feed (placeholder for real-time video)
# def process_video(model_func, model_name):
# 	stframe = st.empty()
# 	alert_placeholder = st.empty()
# 	cap = cv2.VideoCapture(1)  # Use default camera (0 for webcam)

# 	while cap.isOpened():
# 		ret, frame = cap.read()
# 		if not ret:
# 			st.error("Failed to capture video feed.")
# 			break

# 		# Process frame with the selected model
# 		result, alert = model_func(frame)

# 		# Display video frame
# 		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 		stframe.image(frame_rgb, channels="RGB", use_column_width=True)


# 		# Display alert
# 		if alert:
# 			alert_placeholder.error(f"ALERT: {alert} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# 		else:
# 			alert_placeholder.success(f"No issues detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# # Main function to start the app
# def start_app():
# 	st.title("AI Surveillance System")

# 	# Initialize session state for alerts if not already done
# 	if "alerts" not in st.session_state:
# 		st.session_state.alerts = []

# 	options = ["Face Recognition", "Fight Detection", "Object Detection", "Guard Attentiveness", "Gender Detection"]
# 	choice = st.sidebar.selectbox("Select Model", options)

# 	if choice == "Face Recognition":
# 		process_video(detect_face, "face_recognition")
# 	elif choice == "Fight Detection":
# 		process_video_with_webrtc()
# 	elif choice == "Crowd Density":
# 		process_video(detect_crowd_density, "crowd_density")
# 	elif choice == "Guard Attentiveness":
# 		process_video(detect_guard_attentiveness, "guard_attentiveness")
# 	elif choice == "Gender & Age Detection":
# 		process_video(detect_gender_age, "gender_age_detection")
# 	elif choice == "Object Detection":
# 		process_video(detect_objects, "object_detection")


# if __name__ == "__main__":
# 	start_app()














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
from collections import Counter

# Add the Gender_and_age_Detection folder to Python's path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Gender_and_age_Detection'))

# Initialize session state for alerts
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# Importing face recognition functions
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
    yolo_model = YOLO("yolov8n.pt")  # Replace with yolov8.pt or another variant if needed
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

    def send_alert_email(self):
        try:
            sender = "viraj.salunke23@spit.ac.in"
            password = "lppqwukqzossvidg"
            receiver = "virajsalunke12@gmail.com"
            subject = "⚠️ Fight Detected Alert"
            body = f"⚠️ A violent action was detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} in CCTV camera feed."

            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = sender
            msg["To"] = receiver

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender, password)
                server.sendmail(sender, receiver, msg.as_string())
            print("✅ Email sent")
        except Exception as e:
            print(f"❌ Email error: {e}")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        face_results = face_detection.process(frame_rgb)

        lm = self.make_landmark_timestep(results)
        if lm:
            self.lm_list.append(lm)

        img = self.draw_landmark_on_image(mp_draw, results, img)

        if face_results.detections:
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

        if self.label == "fight":
            if self.fight_start_time is None:
                self.fight_start_time = datetime.now()
            elif datetime.now() - self.fight_start_time > timedelta(seconds=5) and not self.alert_sent:
                self.send_alert_email()
                self.alert_sent = True
        else:
            self.fight_start_time = None
            self.alert_sent = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def process_video_with_webrtc():
    ctx = webrtc_streamer(
        key="violence-detection",
        video_processor_factory=ViolenceProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    if ctx.video_processor:
        return ctx.video_processor.label
    return "unknown"

# Gender & Age Detection Initialization
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

def load_gender_age_models():
    model_path = os.path.join("Gender_and_age_Detection", "models")
    faceProto = os.path.join(model_path, "opencv_face_detector.pbtxt")
    faceModel = os.path.join(model_path, "opencv_face_detector_uint8.pb")
    ageProto = os.path.join(model_path, "age_deploy.prototxt")
    ageModel = os.path.join(model_path, "age_net.caffemodel")
    genderProto = os.path.join(model_path, "gender_deploy.prototxt")
    genderModel = os.path.join(model_path, "gender_net.caffemodel")

    face_net = cv2.dnn.readNet(faceModel, faceProto)
    age_net = cv2.dnn.readNet(ageModel, ageProto)
    gender_net = cv2.dnn.readNet(genderModel, genderProto)

    return face_net, age_net, gender_net

face_net, age_net, gender_net = load_gender_age_models()

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
                st.session_state.alerts.append([datetime.now(), "Sleepy Pose", alert])
                h, w, _ = frame.shape
                cv2.putText(frame, alert, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except IndexError:
            pass
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return frame, alert

def detect_gender_age(frame):
    alert = None
    padding = 20
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    h, w = frame.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(w - 1, x2 + padding), min(h - 1, y2 + padding)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [78.4263377603, 87.7689143744, 114.895847746], swapRB=False)
            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age = AGE_LIST[age_preds[0].argmax()]
            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            alert = f"{label} detected"
            st.session_state.alerts.append([datetime.now(), "Gender Detection", label])
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
        st.session_state.alerts.append([datetime.now(), "Face Detection", alert])

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame, alert

def detect_objects(frame):
    """
    Detects objects in the frame using YOLOv8 model and displays bounding boxes/labels.
	
    Args:
        frame: Input frame from the video feed (numpy array, BGR format)
	
    Returns:
        tuple: (annotated_frame, alert, detection_summary)
            - annotated_frame: Frame with bounding boxes and labels
            - alert: String alert message if specific conditions are met, else None
            - detection_summary: String summarizing detections (e.g., "5 persons, 1 bottle")
    """
    try:
        # Run YOLO inference
        results = yolo_model(frame, verbose=False, conf=0.5)  # Confidence threshold
        result = results[0]

        # Generate annotated frame
        annotated_frame = result.plot(
            boxes=True,
            labels=True,
            conf=True,
            font_size=12,
            line_width=2
        )

        # Generate detection summary (e.g., "5 persons, 1 bottle, 1 chair")
        class_ids = [int(box.cls) for box in result.boxes]
        class_counts = Counter(result.names[cls_id] for cls_id in class_ids)
        detection_summary = ", ".join(f"{count} {name}" for name, count in class_counts.items())
        if not detection_summary:
            detection_summary = "No objects detected"

        # Alert logic: Trigger for specific objects (e.g., "person")
        alert = None
        for box in result.boxes:
            class_name = result.names[int(box.cls)]
            if class_name == "person" and box.conf > 0.5:
                alert = f"{class_name.capitalize()} detected!"
                st.session_state.alerts.append([datetime.now(), "Object Detection", alert])
                break

        return annotated_frame, alert, detection_summary

    except Exception as e:
        st.error(f"Object detection error: {e}")
        return frame, None, "Detection failed"

def detect_crowd_density(frame):
    results = yolo_model(frame, verbose=False)
    result = results[0]
    annotated_frame = result.plot()
    person_count = sum(1 for box in result.boxes if result.names[int(box.cls)] == "person")
    alert = "High crowd density detected" if person_count > 5 else None
    return annotated_frame, alert

def process_video(model_func, model_name):
    stframe = st.empty()
    alert_placeholder = st.empty()
    summary_placeholder = st.empty()
    cap = cv2.VideoCapture(1)  # Default webcam

    if not cap.isOpened():
        st.error("Failed to open webcam.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video feed.")
            break

        # Process frame
        if model_func == detect_objects:
            result_frame, alert, detection_summary = model_func(frame)
            summary_placeholder.text(f"Detections: {detection_summary}")
        else:
            result_frame, alert = model_func(frame)
            summary_placeholder.text("")

        # Convert to RGB for Streamlit
        frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        # Display alert
        if alert:
            alert_placeholder.error(f"ALERT: {alert} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            alert_placeholder.success(f"No issues detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Small delay to prevent Streamlit from freezing
        # time.sleep(0.01)

    cap.release()

def start_app():
    st.title("AI Surveillance System")

    if "alerts" not in st.session_state:
        st.session_state.alerts = []

    options = ["Face Recognition", "Fight Detection", "Object Detection", "Guard Attentiveness", "Gender Detection", "Crowd Density"]
    choice = st.sidebar.selectbox("Select Model", options)

    if choice == "Face Recognition":
        process_video(detect_face, "face_recognition")
    elif choice == "Fight Detection":
        process_video_with_webrtc()
    elif choice == "Object Detection":
        process_video(detect_objects, "object_detection")
    elif choice == "Guard Attentiveness":
        process_video(detect_guard_attentiveness, "guard_attentiveness")
    elif choice == "Gender Detection":
        process_video(detect_gender_age, "gender_age_detection")
    elif choice == "Crowd Density":
        process_video(detect_crowd_density, "crowd_density")

if __name__ == "__main__":
    start_app()



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
# from collections import Counter

# # Add the Gender_and_age_Detection folder to Python's path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'Gender_and_age_Detection'))

# # Initialize session state for alerts
# if "alerts" not in st.session_state:
#     st.session_state.alerts = []

# # Importing face recognition functions from import_face_recognition
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
#             elif datetime.now() - self.fight_start_time > timedelta(seconds=3) and not self.alert_sent:
#                 self.send_alert_email()
#                 st.session_state.alerts.append([datetime.now(), "Fight Detection", "Fighting detected"])
#                 st.warning("⚠️ Fight detected!")
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

# # Gender & Age Detection Initialization
# AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# GENDER_LIST = ['Male', 'Female']

# def load_gender_age_models():
#     model_path = os.path.join("Gender_and_age_Detection", "models")
#     faceProto = os.path.join(model_path, "opencv_face_detector.pbtxt")
#     faceModel = os.path.join(model_path, "opencv_face_detector_uint8.pb")
#     ageProto = os.path.join(model_path, "age_deploy.prototxt")
#     ageModel = os.path.join(model_path, "age_net.caffemodel")
#     genderProto = os.path.join(model_path, "gender_deploy.prototxt")
#     genderModel = os.path.join(model_path, "gender_net.caffemodel")
#     face_net = cv2.dnn.readNet(faceModel, faceProto)
#     age_net = cv2.dnn.readNet(ageModel, ageProto)
#     gender_net = cv2.dnn.readNet(genderModel, genderProto)
#     return face_net, age_net, gender_net

# face_net, age_net, gender_net = load_gender_age_models()

# def detect_gender_age(frame):
#     alert = None
#     padding = 20
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
#     face_net.setInput(blob)
#     detections = face_net.forward()
#     h, w = frame.shape[:2]
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.7:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             x1, y1, x2, y2 = box.astype("int")
#             x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
#             x2, y2 = min(w - 1, x2 + padding), min(h - 1, y2 + padding)
#             face = frame[y1:y2, x1:x2]
#             if face.size == 0: continue
#             face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [78.4263377603, 87.7689143744, 114.895847746], swapRB=False)
#             gender_net.setInput(face_blob)
#             gender_preds = gender_net.forward()
#             gender = GENDER_LIST[gender_preds[0].argmax()]
#             age_net.setInput(face_blob)
#             age_preds = age_net.forward()
#             age = AGE_LIST[age_preds[0].argmax()]
#             label = f"{gender}, {age}"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#             alert = f"{label} detected"
#             if gender == "Male":
#                 st.session_state.alerts.append([datetime.now(), "Gender Detection", "Male detected"])
#                 st.warning("⚠️ Male detected!")
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
#     """ Detects objects in the frame using YOLOv8 model and displays bounding boxes/labels. """
#     try:
#         # Run YOLO inference
#         results = yolo_model(frame, verbose=False, conf=0.5)  # Confidence threshold
#         result = results[0]  # Generate annotated frame
#         annotated_frame = result.plot(boxes=True, labels=True, conf=True, font_size=12, line_width=2)
#         detection_summary = ", ".join(f"{count} {name}" for name, count in class_counts.items())
#         alert = None
#         for box in result.boxes:
#             class_name = result.names[int(box.cls)]
#             if class_name in ["phone", "bottle", "marker"]:
#                 alert = f"⚠️ Dangerous item detected: {class_name.capitalize()}"
#                 st.session_state.alerts.append([datetime.now(), "Object Detection", alert])
#                 st.warning(alert)
#                 break
#     except Exception as e:
#         st.error(f"Object detection error: {e}")
#         return frame, None, "Detection failed"

#     return annotated_frame, alert, detection_summary

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
#                 st.session_state.alerts.append([datetime.now(), "Guard Attentiveness", alert])
#                 st.warning(alert)
#             h, w, _ = frame.shape
#             cv2.putText(frame, alert, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         except IndexError:
#             pass
#     mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#     return frame, alert

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
#         result_frame, alert = model_func(frame)
#         summary_placeholder.text("")  # Reset summary text
#         frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
#         stframe.image(frame_rgb, channels="RGB", use_column_width=True)

#         # Display alert if alert
#         if alert:
#             alert_placeholder.warning(f"ALERT: {alert} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#         else:
#             alert_placeholder.success(f"No issues detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     cap.release()

# def start_app():
#     st.title("AI Surveillance System")
#     if "alerts" not in st.session_state:
#         st.session_state.alerts = []
   
#     options = ["Face Recognition", "Fight Detection", "Object Detection", "Guard Attentiveness", "Gender Detection", "Crowd Density"]
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
#         process_video(detect_gender_age, "gender_age_detection")
#     elif choice == "Crowd Density":
#         process_video(detect_crowd_density, "crowd_density")

# if __name__ == "__main__":
#     start_app()




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
# from collections import Counter

# # Add the Gender_and_age_Detection folder to Python's path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'Gender_and_age_Detection'))

# # Initialize session state for alerts
# if "alerts" not in st.session_state:
# 	st.session_state.alerts = []

# # Importing face recognition functions from import_face_recognition
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
# 	yolo_model = YOLO("yolov8n.pt")  # Replace with yolov8.pt or another variant if needed
# except Exception as e:
# 	st.error(f"Error loading YOLO model: {e}")
# 	st.stop()

# # Load fight detection model
# def load_fight_model():
# 	try:
# 		return load_model(r'C:\Users\kriya\Desktop\face\lstm-fight-detection.h5')
# 	except Exception as e:
# 		st.error(f"Error loading fight model: {e}")
# 		st.stop()

# fight_model = load_fight_model()

# # WebRTC configuration
# RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# # Violence Processor for Fight Detection
# class ViolenceProcessor(VideoProcessorBase):
# 	def __init__(self):
# 		self.label = "unknown"
# 		self.label_history = []
# 		self.lm_list = []
# 		self.fight_start_time = None
# 		self.alert_sent = False

# 	def make_landmark_timestep(self, results):
# 		if results.pose_landmarks:
# 			c_lm = []
# 			for lm in results.pose_landmarks.landmark:
# 				c_lm.extend([lm.x, lm.y, lm.z])
# 			return c_lm
# 		return None

# 	def draw_landmark_on_image(self, mp_draw, results, frame):
# 		if results.pose_landmarks:
# 			mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
# 			for id, lm in enumerate(results.pose_landmarks.landmark):
# 				h, w, c = frame.shape
# 				cx, cy = int(lm.x * w), int(lm.y * h)
# 				cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
# 		return frame

# 	def detect(self, model, lm_list):
# 		if len(lm_list) >= 20:
# 			lm_array = np.array(lm_list[-20:])
# 			lm_array = np.expand_dims(lm_array, axis=0)
# 			try:
# 				result = model.predict(lm_array, verbose=0)[0]
# 				pred_label = "fight" if result[0] > 0.5 else "normal"
# 				for idx in [15, 16, 13, 14]:
# 					x_diff = lm_list[-1][idx*3] - lm_list[-2][idx*3]
# 					y_diff = lm_list[-1][idx*3+1] - lm_list[-2][idx*3+1]
# 					if abs(x_diff) > 0.07 or abs(y_diff) > 0.07:
# 						pred_label = "fight"
# 						break
# 				self.label_history.append(pred_label)
# 				if len(self.label_history) > 3:
# 					self.label_history.pop(0)
# 				self.label = max(set(self.label_history), key=self.label_history.count)
# 			except Exception as e:
# 				st.error(f"Prediction error: {e}")
# 				self.label = "ERROR"
# 		return self.label

# 	def send_alert_email(self):
# 		try:
# 			sender = "viraj.salunke23@spit.ac.in"
# 			password = "lppqwukqzossvidg"
# 			receiver = "virajsalunke12@gmail.com"
# 			subject = "⚠️ Fight Detected Alert"
# 			body = f"⚠️ A violent action was detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} in CCTV camera feed."
# 			msg = MIMEText(body)
# 			msg["Subject"] = subject
# 			msg["From"] = sender
# 			msg["To"] = receiver
# 			with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
# 				server.login(sender, password)
# 				server.sendmail(sender, receiver, msg.as_string())
# 			print("✅ Email sent")
# 		except Exception as e:
# 			print(f"❌ Email error: {e}")

# 	def apply_blur_to_face(self, frame, face_locations):
# 		for (top, right, bottom, left) in face_locations:
# 			face_region = frame[top:bottom, left:right]
# 			if face_region.size > 0:
# 				blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
# 				frame[top:bottom, left:right] = blurred
# 		return frame

# 	def recv(self, frame):
# 		img = frame.to_ndarray(format="bgr24")
# 		frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 		results = pose.process(frame_rgb)
# 		face_results = face_detection.process(frame_rgb)
# 		lm = self.make_landmark_timestep(results)
# 		if lm:
# 			self.lm_list.append(lm)
# 		img = self.draw_landmark_on_image(mp_draw, results, img)

# 		if face_results.detections:
# 			face_locations = []
# 			for detection in face_results.detections:
# 				bbox = detection.location_data.relative_bounding_box
# 				h, w, _ = img.shape
# 				x1 = int(bbox.xmin * w)
# 				y1 = int(bbox.ymin * h)
# 				x2 = x1 + int(bbox.width * w)
# 				y2 = y1 + int(bbox.height * h)
# 				x1, y1 = max(0, x1), max(0, y1)
# 				x2, y2 = min(w, x2), min(h, y2)
# 				face_locations.append((y1, x2, y2, x1))  # Adjusted for consistency in bbox order

# 			# Apply blur to detected face
# 			img = self.apply_blur_to_face(img, face_locations)

# 		self.label = self.detect(fight_model, self.lm_list)

# 		color = (0, 0, 255) if self.label == "fight" else (0, 255, 0)
# 		cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color, 2)
# 		cv2.putText(img, self.label.upper(), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# 		if self.label == "fight":
# 			if self.fight_start_time is None:
# 				self.fight_start_time = datetime.now()
# 			elif datetime.now() - self.fight_start_time > timedelta(seconds=3) and not self.alert_sent:
# 				self.send_alert_email()
# 				st.session_state.alerts.append([datetime.now(), "Fight Detection", "Fighting detected"])
# 				st.warning("⚠️ Fight detected!")
# 				self.alert_sent = True
# 		else:
# 			self.fight_start_time = None
# 			self.alert_sent = False

# 		return av.VideoFrame.from_ndarray(img, format="bgr24")

# # Gender & Age Detection Initialization
# AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# GENDER_LIST = ['Male', 'Female']

# def load_gender_age_models():
# 	model_path = os.path.join("Gender_and_age_Detection", "models")
# 	faceProto = os.path.join(model_path, "opencv_face_detector.pbtxt")
# 	faceModel = os.path.join(model_path, "opencv_face_detector_uint8.pb")
# 	ageProto = os.path.join(model_path, "age_deploy.prototxt")
# 	ageModel = os.path.join(model_path, "age_net.caffemodel")
# 	genderProto = os.path.join(model_path, "gender_deploy.prototxt")
# 	genderModel = os.path.join(model_path, "gender_net.caffemodel")
# 	face_net = cv2.dnn.readNet(faceModel, faceProto)
# 	age_net = cv2.dnn.readNet(ageModel, ageProto)
# 	gender_net = cv2.dnn.readNet(genderModel, genderProto)
# 	return face_net, age_net, gender_net

# face_net, age_net, gender_net = load_gender_age_models()

# def apply_blur_to_face(frame, face_locations):
# 	for (top, right, bottom, left) in face_locations:
# 		face_region = frame[top:bottom, left:right]
# 		if face_region.size > 0:
# 			blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
# 			frame[top:bottom, left:right] = blurred
# 	return frame

# def detect_gender_age(frame):
# 	alert = None
# 	padding = 20
# 	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
# 	face_net.setInput(blob)
# 	detections = face_net.forward()
# 	h, w = frame.shape[:2]
# 	face_locations = []
# 	for i in range(detections.shape[2]):
# 		confidence = detections[0, 0, i, 2]
# 		if confidence > 0.7:
# 			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
# 			x1, y1, x2, y2 = box.astype("int")
# 			x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
# 			x2, y2 = min(w - 1, x2 + padding), min(h - 1, y2 + padding)
# 			face = frame[y1:y2, x1:x2]
# 			if face.size == 0: continue
# 			face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [78.4263377603, 87.7689143744, 114.895847746], swapRB=False)
# 			gender_net.setInput(face_blob)
# 			gender_preds = gender_net.forward()
# 			gender = GENDER_LIST[gender_preds[0].argmax()]
# 			age_net.setInput(face_blob)
# 			age_preds = age_net.forward()
# 			age = AGE_LIST[age_preds[0].argmax()]
# 			label = f"{gender}, {age}"
# 			cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
# 			cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# 			alert = f"{label} detected"
# 			face_locations.append((y1, x2, y2, x1))  # Save face locations for blurring
# 			if gender == "Male":
# 				st.session_state.alerts.append([datetime.now(), "Gender Detection", "Male detected"])
# 				st.warning("⚠️ Male detected!")

# 	# Apply blur to faces detected
# 	frame = apply_blur_to_face(frame, face_locations)
# 	return frame, alert

# def detect_face(frame):
# 	rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 	face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
# 	results = face_detector.process(rgb_frame)

# 	face_names = []
# 	face_locations = []

# 	if results.detections:
# 		for detection in results.detections:
# 			bbox = detection.location_data.relative_bounding_box
# 			ih, iw, _ = frame.shape
# 			x = max(0, int(bbox.xmin * iw))
# 			y = max(0, int(bbox.ymin * ih))
# 			w = int(bbox.width * iw)
# 			h = int(bbox.height * ih)
# 			top, right, bottom, left = y, x + w, y + h, x
# 			face_locations.append((top, right, bottom, left))

# 		encodings = face_recognition.face_encodings(rgb_frame, face_locations)
# 		for face_encoding in encodings:
# 			name = "Unknown"
# 			distances = face_recognition.face_distance(known_encodings, face_encoding)
# 			if len(distances) > 0:
# 				best_match_index = np.argmin(distances)
# 				if distances[best_match_index] < 0.45:
# 					name = known_names[best_match_index]
# 			face_names.append(name)

# 	alert = None
# 	if "Unknown" in face_names:
# 		alert = "Unknown face detected!"
# 		st.session_state.alerts.append([datetime.now(), "Face Detection", alert])

# 	for (top, right, bottom, left), name in zip(face_locations, face_names):
# 		color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
# 		cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
# 		cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
# 		cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
# 	return frame, alert

# 	# Apply blur to faces detected
# 	frame = apply_blur_to_face(frame, face_locations)

# 	return frame, alert


# def detect_objects(frame):
# 	""" Detects objects in the frame using YOLOv8 model and displays bounding boxes/labels. """
# 	try:
# 		# Run YOLO inference
# 		results = yolo_model(frame, verbose=False, conf=0.5)  # Confidence threshold
# 		result = results[0]  # Generate annotated frame
# 		annotated_frame = result.plot(boxes=True, labels=True, conf=True, font_size=12, line_width=2)
# 		detection_summary = ", ".join(f"{count} {name}" for name, count in class_counts.items())
# 		alert = None
# 		for box in result.boxes:
# 			class_name = result.names[int(box.cls)]
# 			if class_name in ["phone", "bottle", "marker"]:
# 				alert = f"⚠️ Dangerous item detected: {class_name.capitalize()}"
# 				st.session_state.alerts.append([datetime.now(), "Object Detection", alert])
# 				st.warning(alert)
# 				x1, y1, x2, y2 = box.xyxy[0]
# 				frame = apply_blur_to_face(frame, [(int(y1), int(x2), int(y2), int(x1))])
# 				break
# 	except Exception as e:
# 		st.error(f"Object detection error: {e}")
# 		return frame, None, "Detection failed"

# 	return annotated_frame, alert, detection_summary


# def detect_guard_attentiveness(frame):
# 	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 	results = pose.process(frame_rgb)
# 	alert = None
# 	if results.pose_landmarks:
# 		landmarks = results.pose_landmarks.landmark
# 		try:
# 			nose = landmarks[0]
# 			left_eye = landmarks[1]
# 			right_eye = landmarks[2]
# 			left_shoulder = landmarks[11]
# 			right_shoulder = landmarks[12]
# 			avg_eye_y = (left_eye.y + right_eye.y) / 2
# 			avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
# 			if (nose.y - avg_shoulder_y) > -0.05 or (avg_eye_y - avg_shoulder_y) > -0.05:
# 				alert = "Sleepy/drowsy pose detected!"
# 				st.session_state.alerts.append([datetime.now(), "Guard Attentiveness", alert])
# 				st.warning(alert)
# 			h, w, _ = frame.shape
# 			cv2.putText(frame, alert, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
# 		except IndexError:
# 			pass
# 	mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
# 	return frame, alert

# def process_video(model_func, model_name):
# 	stframe = st.empty()
# 	alert_placeholder = st.empty()
# 	summary_placeholder = st.empty()
# 	cap = cv2.VideoCapture(1)  # Default webcam
# 	if not cap.isOpened():
# 		st.error("Failed to open webcam.")
# 		return
# 	while cap.isOpened():
# 		ret, frame = cap.read()
# 		if not ret:
# 			st.error("Failed to capture video feed.")
# 			break
# 		result_frame, alert = model_func(frame)
# 		summary_placeholder.text("")  # Reset summary text
# 		frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
# 		stframe.image(frame_rgb, channels="RGB", use_column_width=True)

# 		# Display alert if alert
# 		if alert:
# 			alert_placeholder.warning(f"ALERT: {alert} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# 		else:
# 			alert_placeholder.success(f"No issues detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# 	cap.release()

# def start_app():
# 	st.title("AI Surveillance System")
# 	if "alerts" not in st.session_state:
# 		st.session_state.alerts = []
   
# 	options = ["Face Recognition", "Fight Detection", "Object Detection", "Guard Attentiveness", "Gender Detection", "Crowd Density"]
# 	choice = st.sidebar.selectbox("Select Model", options)
   
# 	if choice == "Face Recognition":
# 		process_video(detect_face, "face_recognition")
# 	elif choice == "Fight Detection":
# 		process_video_with_webrtc()
# 	elif choice == "Object Detection":
# 		process_video(detect_objects, "object_detection")
# 	elif choice == "Guard Attentiveness":
# 		process_video(detect_guard_attentiveness, "guard_attentiveness")
# 	elif choice == "Gender Detection":
# 		process_video(detect_gender_age, "gender_age_detection")
# 	elif choice == "Crowd Density":
# 		process_video(detect_crowd_density, "crowd_density")

# if __name__ == "__main__":
# 	start_app()