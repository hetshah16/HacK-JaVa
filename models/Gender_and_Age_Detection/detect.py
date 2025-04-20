# # #A Gender and Age Detection program by Mahesh Sawant

# # import cv2
# # import math
# # import argparse

# # def highlightFace(net, frame, conf_threshold=0.7):
# #     frameOpencvDnn=frame.copy()
# #     frameHeight=frameOpencvDnn.shape[0]
# #     frameWidth=frameOpencvDnn.shape[1]
# #     blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

# #     net.setInput(blob)
# #     detections=net.forward()
# #     faceBoxes=[]
# #     for i in range(detections.shape[2]):
# #         confidence=detections[0,0,i,2]
# #         if confidence>conf_threshold:
# #             x1=int(detections[0,0,i,3]*frameWidth)
# #             y1=int(detections[0,0,i,4]*frameHeight)
# #             x2=int(detections[0,0,i,5]*frameWidth)
# #             y2=int(detections[0,0,i,6]*frameHeight)
# #             faceBoxes.append([x1,y1,x2,y2])
# #             cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
# #     return frameOpencvDnn,faceBoxes


# # parser=argparse.ArgumentParser()
# # parser.add_argument('--image')

# # args=parser.parse_args()

# # faceProto="opencv_face_detector.pbtxt"
# # faceModel="opencv_face_detector_uint8.pb"
# # ageProto="age_deploy.prototxt"
# # ageModel="age_net.caffemodel"
# # genderProto="gender_deploy.prototxt"
# # genderModel="gender_net.caffemodel"

# # MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
# # ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# # genderList=['Male','Female']

# # faceNet=cv2.dnn.readNet(faceModel,faceProto)
# # ageNet=cv2.dnn.readNet(ageModel,ageProto)
# # genderNet=cv2.dnn.readNet(genderModel,genderProto)

# # video=cv2.VideoCapture(args.image if args.image else 0)
# # padding=20
# # while cv2.waitKey(1)<0 :
# #     hasFrame,frame=video.read()
# #     if not hasFrame:
# #         cv2.waitKey()
# #         break
    
# #     resultImg,faceBoxes=highlightFace(faceNet,frame)
# #     if not faceBoxes:
# #         print("No face detected")

# #     for faceBox in faceBoxes:
# #         face=frame[max(0,faceBox[1]-padding):
# #                    min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
# #                    :min(faceBox[2]+padding, frame.shape[1]-1)]

# #         blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
# #         genderNet.setInput(blob)
# #         genderPreds=genderNet.forward()
# #         gender=genderList[genderPreds[0].argmax()]
# #         print(f'Gender: {gender}')

# #         ageNet.setInput(blob)
# #         agePreds=ageNet.forward()
# #         age=ageList[agePreds[0].argmax()]
# #         print(f'Age: {age[1:-1]} years')

# #         cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
# #         cv2.imshow("Detecting age and gender", resultImg)



# import cv2
# import time

# # Load face detection model
# faceProto = "opencv_face_detector.pbtxt"
# faceModel = "opencv_face_detector_uint8.pb"
# faceNet = cv2.dnn.readNet(faceModel, faceProto)

# # Load gender and age models
# ageProto = "age_deploy.prototxt"
# ageModel = "age_net.caffemodel"
# genderProto = "gender_deploy.prototxt"
# genderModel = "gender_net.caffemodel"

# ageNet = cv2.dnn.readNet(ageModel, ageProto)
# genderNet = cv2.dnn.readNet(genderModel, genderProto)

# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# genderList = ['Male', 'Female']

# # Function to detect face
# def highlightFace(net, frame, conf_threshold=0.7):
#     frameOpencvDnn = frame.copy()
#     h, w = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], swapRB=False)
#     net.setInput(blob)
#     detections = net.forward()
#     faceBoxes = []
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > conf_threshold:
#             x1 = int(detections[0, 0, i, 3] * w)
#             y1 = int(detections[0, 0, i, 4] * h)
#             x2 = int(detections[0, 0, i, 5] * w)
#             y2 = int(detections[0, 0, i, 6] * h)
#             faceBoxes.append([x1, y1, x2, y2])
#             cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     return frameOpencvDnn, faceBoxes

# # Start video capture
# video = cv2.VideoCapture(0)
# padding = 20

# while True:
#     hasFrame, frame = video.read()
#     if not hasFrame:
#         break

#     resultImg, faceBoxes = highlightFace(faceNet, frame)

#     if not faceBoxes:
#         print("No face detected")
#         cv2.imshow("Detecting age and gender", resultImg)
#         if cv2.waitKey(1) == ord('q'):
#             break
#         continue

#     for faceBox in faceBoxes:
#         face = frame[max(0, faceBox[1]-padding):min(faceBox[3]+padding, frame.shape[0]-1),
#                      max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

#         blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

#         genderNet.setInput(blob)
#         genderPreds = genderNet.forward()
#         gender = genderList[genderPreds[0].argmax()]

#         ageNet.setInput(blob)
#         agePreds = ageNet.forward()
#         age = ageList[agePreds[0].argmax()]

#         label = f"{gender}, {age}"
#         cv2.putText(resultImg, label, (faceBox[0], faceBox[1]-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
#         print(f"Detected: {label}")

#     cv2.imshow("Detecting age and gender", resultImg)
#     if cv2.waitKey(1) == ord('q'):
#         break

# video.release()
# cv2.destroyAllWindows()




# import os
# import cv2

# # Use absolute path to ensure the model is found
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, 'models')

# faceModel = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")
# faceProto = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")

# print("Face model path exists:", os.path.exists(faceModel))  # Should be True
# print("Face proto path exists:", os.path.exists(faceProto))  # Should be True

# # Load the model
# faceNet = cv2.dnn.readNet(faceModel, faceProto)


# # Load models
# ageProto = "Gender_and_Age_Detection\\age_deploy.prototxt"
# ageModel = "Gender_and_Age_Detection\\age_net.caffemodel"
# genderProto = "Gender_and_Age_Detection\\gender_deploy.prototxt"
# genderModel = "Gender_and_Age_Detection\\gender_net.caffemodel"   

# # Load networks
# ageNet = cv2.dnn.readNet(ageModel, ageProto)
# genderNet = cv2.dnn.readNet(genderModel, genderProto)

# # Constants
# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# genderList = ['Male', 'Female']
# padding = 20

# # Function to detect faces
# def detect_faces(net, frame, conf_threshold=0.7):
#     h, w = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
#     net.setInput(blob)
#     detections = net.forward()
#     faceBoxes = []

#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > conf_threshold:
#             x1 = int(detections[0, 0, i, 3] * w)
#             y1 = int(detections[0, 0, i, 4] * h)
#             x2 = int(detections[0, 0, i, 5] * w)
#             y2 = int(detections[0, 0, i, 6] * h)
#             faceBoxes.append((x1, y1, x2, y2))
#     return faceBoxes

# # Start video stream
# cap = cv2.VideoCapture(0)
# print("Press 'q' to quit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     faceBoxes = detect_faces(faceNet, frame)

#     for (x1, y1, x2, y2) in faceBoxes:
#         face = frame[max(0, y1 - padding):min(y2 + padding, frame.shape[0] - 1),
#                      max(0, x1 - padding):min(x2 + padding, frame.shape[1] - 1)]

#         blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

#         genderNet.setInput(blob)
#         gender = genderList[genderNet.forward()[0].argmax()]

#         ageNet.setInput(blob)
#         age = ageList[ageNet.forward()[0].argmax()]

#         label = f"{gender}, {age}"
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, label, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

#     cv2.imshow("Age and Gender Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



import os
import cv2

# Use absolute path to ensure the model is found
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

faceModel = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")
faceProto = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")

print("Face model path exists:", os.path.exists(faceModel))  # Should be True
print("Face proto path exists:", os.path.exists(faceProto))  # Should be True

# Load the model
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Load models
ageProto = "Gender_and_Age_Detection\\age_deploy.prototxt"
ageModel = "Gender_and_Age_Detection\\age_net.caffemodel"
genderProto = "Gender_and_Age_Detection\\gender_deploy.prototxt"
genderModel = "Gender_and_Age_Detection\\gender_net.caffemodel"   

# Load networks
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
padding = 20

# Function to detect faces
def detect_faces(net, frame, conf_threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            faceBoxes.append((x1, y1, x2, y2))
    return faceBoxes

# Function to detect gender and age
def detect_gender(frame):
    faceBoxes = detect_faces(faceNet, frame)
    result = []
    
    for (x1, y1, x2, y2) in faceBoxes:
        face = frame[max(0, y1 - padding):min(y2 + padding, frame.shape[0] - 1),
                     max(0, x1 - padding):min(x2 + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        gender = genderList[genderNet.forward()[0].argmax()]

        ageNet.setInput(blob)
        age = ageList[ageNet.forward()[0].argmax()]

        label = f"{gender}, {age}"
        result.append((x1, y1, x2, y2, label))  # Return the bounding box and the label

    return result




