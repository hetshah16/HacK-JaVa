# import cv2
# from deepface import DeepFace
# import os

# def capture_image():
#     """Capture an image from the webcam and save it to a file."""
#     cap = cv2.VideoCapture(1)

#     if not cap.isOpened():
#         print("❌ Could not access the camera.")
#         return None

#     print("📸 Press 's' to take a picture")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("❌ Failed to grab frame.")
#             break

#         cv2.imshow("Live Feed - Press 's' to capture", frame)

#         if cv2.waitKey(1) & 0xFF == ord('s'):
#             image_path = "captured_image.jpg"
#             cv2.imwrite(image_path, frame)
#             print(f"✅ Image saved as {image_path}")
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return image_path

# def are_faces_same(image_path1, image_path2, model_name="Facenet512", threshold=0.4):
#     """Compare two faces using DeepFace and return similarity result."""
#     try:
#         result = DeepFace.verify(
#             img1_path=image_path1,
#             img2_path=image_path2,
#             model_name=model_name,
#             enforce_detection=True  # ensures face must be detected
#         )
#         print(f"🔍 Distance: {result['distance']:.4f} | Verified: {result['verified']}")
#         return result['verified'] and result['distance'] < threshold
#     except Exception as e:
#         print(f"⚠️ Error comparing faces: {str(e)}")
#         return False

# def main():
#     known_image = "known_faces/20250419_191129.jpg"  # Update with your image

#     if not os.path.exists(known_image):
#         print(f"❌ Known face image not found at: {known_image}")
#         return

#     captured_image_path = capture_image()
#     if captured_image_path is None:
#         print("❌ Image capture failed.")
#         return

#     match = are_faces_same(captured_image_path, known_image)
#     if match:
#         print("✅ Known: Faces match.")
#     else:
#         print("❌ Unknown: Faces do not match.")

# if __name__ == "__main__":
#     main()




import cv2
from deepface import DeepFace
import os
import time
import threading

def load_known_faces(known_dir="known_faces", model_name="Facenet512"):
    known_faces = []
    known_names = []

    print("📂 Loading known faces...")
    for filename in os.listdir(known_dir):
        filepath = os.path.join(known_dir, filename)
        try:
            DeepFace.verify(img1_path=filepath, img2_path=filepath, model_name=model_name)
            known_faces.append(filepath)
            known_names.append(os.path.splitext(filename)[0])
            print(f"✅ Loaded: {filename}")
        except Exception as e:
            print(f"⚠️ Skipped {filename}: {str(e)}")

    return known_faces, known_names

def identify_face(frame, known_faces, known_names, model_name="Facenet512", threshold=0.4):
    temp_path = "temp_face.jpg"
    cv2.imwrite(temp_path, frame)

    for idx, known_face in enumerate(known_faces):
        try:
            result = DeepFace.verify(
                img1_path=temp_path,
                img2_path=known_face,
                model_name=model_name,
                enforce_detection=True
            )
            if result["verified"] and result["distance"] < threshold:
                return known_names[idx]
        except Exception:
            continue
    return "Unknown"

def main():
    known_faces, known_names = load_known_faces()
    if not known_faces:
        print("❌ No valid known faces loaded.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access the camera.")
        return

    print("🎥 Starting real-time face recognition...")

    label = "Detecting..."
    frame_count = 0
    processing = False

    def recognize_thread(frame_to_process):
        nonlocal label, processing
        try:
            label = identify_face(frame_to_process, known_faces, known_names)
        except Exception as e:
            label = "Unknown"
            print(f"⚠️ Error: {str(e)}")
        processing = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        display_frame = frame.copy()
        display_frame = cv2.putText(
            display_frame, f"{label}", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label != "Unknown" else (0, 0, 255), 2
        )
        cv2.imshow("Real-Time Face Recognition", display_frame)

        frame_count += 1
        if frame_count % 20 == 0 and not processing:
            processing = True
            resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            threading.Thread(target=recognize_thread, args=(resized,)).start()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
