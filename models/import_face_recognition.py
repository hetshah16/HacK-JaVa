# import cv2
# import face_recognition
# import os
# import numpy as np

# def load_known_faces(folder_path="known_faces"):
#     known_encodings = []
#     known_names = []

#     print("📂 Loading known faces...")
#     for filename in os.listdir(folder_path):
#         if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue

#         path = os.path.join(folder_path, filename)
#         name = os.path.splitext(filename)[0]

#         image = face_recognition.load_image_file(path)
#         face_locations = face_recognition.face_locations(image)
#         face_encodings = face_recognition.face_encodings(image, face_locations)

#         if face_encodings:
#             known_encodings.append(face_encodings[0])
#             known_names.append(name)
#             print(f"✅ Loaded and encoded: {name}")
#         else:
#             print(f"⚠️ No face found in {filename}")

#     return known_encodings, known_names

# def recognize_faces_live(known_encodings, known_names, tolerance=0.45):
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("❌ Camera not accessible.")
#         return

#     print("🎥 Starting real-time recognition... Press 'q' to quit.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("❌ Failed to capture frame.")
#             break

#         small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#         rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

#         face_locations = face_recognition.face_locations(rgb_small_frame)
#         face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#         face_names = []

#         for encoding in face_encodings:
#             matches = face_recognition.compare_faces(known_encodings, encoding, tolerance)
#             name = "Unknown"

#             if True in matches:
#                 distances = face_recognition.face_distance(known_encodings, encoding)
#                 best_match_index = np.argmin(distances)
#                 name = known_names[best_match_index]

#             face_names.append(name)

#         # Draw boxes and labels
#         for (top, right, bottom, left), name in zip(face_locations, face_names):
#             top *= 4
#             right *= 4
#             bottom *= 4
#             left *= 4

#             color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
#             label = f"Matched: {name}" if name != "Unknown" else "Unknown"

#             cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
#             cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
#             cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

#         cv2.imshow("🧠 Real-Time Face Recognition", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("🛑 Quitting...")
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def main():
#     known_encodings, known_names = load_known_faces()
#     if not known_encodings:
#         print("❌ No known faces loaded. Please add images to the known_faces folder.")
#         return

#     recognize_faces_live(known_encodings, known_names)

# if __name__ == "__main__":
#     main()



# import os
# import cv2
# import numpy as np
# import face_recognition
# import mediapipe as mp


# # ------------------ Setup ------------------ #
# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils

# # 🔄 Rotate helper
# def rotate_image(image, angle):
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#     return cv2.warpAffine(image, matrix, (w, h))

# # 📥 Load known faces with rotation
# def load_known_faces(known_dir='known_faces'):
#     known_encodings = []
#     known_names = []
#     face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

#     print("📂 Preprocessing known faces...")
#     for filename in os.listdir(known_dir):
#         if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#             continue

#         path = os.path.join(known_dir, filename)
#         image = cv2.imread(path)
#         name = os.path.splitext(filename)[0]

#         found = False
#         for angle in range(0, 360, 90):
#             rotated = rotate_image(image, angle)
#             rgb_image = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
#             results = face_detector.process(rgb_image)

#             if results.detections:
#                 for detection in results.detections:
#                     bbox = detection.location_data.relative_bounding_box
#                     ih, iw, _ = rgb_image.shape
#                     x = max(0, int(bbox.xmin * iw))
#                     y = max(0, int(bbox.ymin * ih))
#                     w = int(bbox.width * iw)
#                     h = int(bbox.height * ih)
#                     top, right, bottom, left = y, x + w, y + h, x

#                     encodings = face_recognition.face_encodings(rgb_image, [(top, right, bottom, left)])
#                     if encodings:
#                         known_encodings.append(encodings[0])
#                         known_names.append(name)
#                         print(f"✅ {filename} (rotated {angle}°) encoded.")
#                         found = True
#                         break
#             if found:
#                 break

#         if not found:
#             print(f"⚠️ No face found in {filename} after rotation attempts.")

#     face_detector.close()
#     return known_encodings, known_names

# # 🎥 Real-time recognition
# def recognize_faces_live(known_encodings, known_names, tolerance=0.45):
#     video_capture = cv2.VideoCapture(1)
#     detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

#     print("📸 Starting real-time face recognition...")
#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             break

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = detector.process(rgb_frame)

#         face_names = []
#         face_locations = []

#         if results.detections:
#             for detection in results.detections:
#                 bbox = detection.location_data.relative_bounding_box
#                 ih, iw, _ = frame.shape
#                 x = max(0, int(bbox.xmin * iw))
#                 y = max(0, int(bbox.ymin * ih))
#                 w = int(bbox.width * iw)
#                 h = int(bbox.height * ih)
#                 top, right, bottom, left = y, x + w, y + h, x

#                 face_locations.append((top, right, bottom, left))

#             encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#             for face_encoding in encodings:
#                 name = "Unknown"
#                 distances = face_recognition.face_distance(known_encodings, face_encoding)
#                 if len(distances) > 0:
#                     best_match_index = np.argmin(distances)
#                     if distances[best_match_index] < tolerance:
#                         name = known_names[best_match_index]
#                 face_names.append(name)

#         # 🎯 Draw detections
#         for (top, right, bottom, left), name in zip(face_locations, face_names):
#             color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
#             cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
#             cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
#             cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

#         cv2.imshow("🧠 Real-Time Face Recognition", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     video_capture.release()
#     detector.close()
#     cv2.destroyAllWindows()

# # 🚀 Main launcher
# def main():
#     known_encodings, known_names = load_known_faces()
#     if not known_encodings:
#         print("❌ No valid faces loaded.")
#         return
#     recognize_faces_live(known_encodings, known_names)

# if __name__ == "__main__":
#     main()


import os
import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import pickle  # 🔐 Added for saving the model

# ------------------ Setup ------------------ #
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def save_encodings(encodings, names, filename="encodings.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump((encodings, names), f)
    print(f"💾 Saved encodings to {filename}")

# Add this to load saved encodings
def load_saved_encodings(filename="encodings.pkl"):
    if not os.path.exists(filename):
        return None, None
    with open(filename, 'rb') as f:
        encodings, names = pickle.load(f)
    print(f"📦 Loaded encodings from {filename}")
    return encodings, names

# 🔄 Rotate helper
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

# 📥 Load known faces with rotation
def load_known_faces(known_dir='known_faces'):
    known_encodings = []
    known_names = []
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    print("📂 Preprocessing known faces...")
    for filename in os.listdir(known_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        path = os.path.join(known_dir, filename)
        image = cv2.imread(path)
        name = os.path.splitext(filename)[0]

        found = False
        for angle in range(0, 360, 90):
            rotated = rotate_image(image, angle)
            rgb_image = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_image)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    ih, iw, _ = rgb_image.shape
                    x = max(0, int(bbox.xmin * iw))
                    y = max(0, int(bbox.ymin * ih))
                    w = int(bbox.width * iw)
                    h = int(bbox.height * ih)
                    top, right, bottom, left = y, x + w, y + h, x

                    encodings = face_recognition.face_encodings(rgb_image, [(top, right, bottom, left)])
                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(name)
                        print(f"✅ {filename} (rotated {angle}°) encoded.")
                        found = True
                        break
            if found:
                break

        if not found:
            print(f"⚠️ No face found in {filename} after rotation attempts.")

    face_detector.close()
    return known_encodings, known_names

# 🎥 Real-time recognition
def recognize_faces_live(known_encodings, known_names, tolerance=0.45):
    video_capture = cv2.VideoCapture(1)
    detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    print("📸 Starting real-time face recognition...")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb_frame)

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
                    if distances[best_match_index] < tolerance:
                        name = known_names[best_match_index]
                face_names.append(name)

        # 🎯 Draw detections
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("🧠 Real-Time Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    detector.close()
    cv2.destroyAllWindows()

# 🚀 Main launcher
def main():
    known_encodings, known_names = load_known_faces()
    if not known_encodings:
        print("❌ No valid faces loaded.")
        return

    # 💾 Save encodings and names for Streamlit
    with open("face_data.pkl", "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)
    print("📦 Encodings saved to face_data.pkl")

    recognize_faces_live(known_encodings, known_names)

if __name__ == "__main__":
    main()
