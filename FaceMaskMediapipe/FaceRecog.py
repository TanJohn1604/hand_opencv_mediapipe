import cv2
import mediapipe as mp
import time

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

def show_fps(img, fps):
    """Draw fps number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    fps_text = 'FPS: {:.2f}'.format(fps)
    cv2.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
    return img
def detect_and_predict_mask(img, results, masknet):
    (h, w) = img.shape[:2]
    faces=[]
    locs=[]
    preds=[]
    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = [int(bboxC.xmin * iw), int(bboxC.ymin * ih),
            int(bboxC.width * iw), int(bboxC.height * ih)]

           #  cv2.rectangle(img, bbox, (255, 0, 255), 2)
           #  cv2.putText(img, f'{int(detection.score[0] * 100)}%{id}',
           # (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
           #  2, (255, 0, 255), 2)

            startX=bbox[0]
            startY=bbox[1]
            endX=bbox[0]+bbox[2]
            endY=bbox[1]+bbox[3]
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            startX=int(startX)
            startY = int(startY)
            endX=int(endX)
            endY=int(endY)
            face = img[startY:endY, startX:endX]
            if face.any():

                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                faces.append(face)
                locs.append((startX,startY,endX,endY))
        if len(faces) > 0:
            faces=np.array(faces,dtype="float32")
            preds=masknet.predict(faces,batch_size=32)
    return (locs,preds)



cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

masknet = load_model('mask2')
fps = 0.0
tic = time.time()
while True:
    ret,img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)
    (locs, preds)=detect_and_predict_mask(imgRGB, results, masknet)
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(img, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)

    # if results.detections:
    #     for id, detection in enumerate(results.detections):
    #         # mpDraw.draw_detection(img, detection)
    #         # print(id, detection)
    #         # print(detection.score)
    #         # print(detection.location_data.relative_bounding_box)
    #         bboxC = detection.location_data.relative_bounding_box
    #         ih, iw, ic = img.shape
    #         bbox = [int(bboxC.xmin * iw), int(bboxC.ymin * ih),
    #         int(bboxC.width * iw), int(bboxC.height * ih)]
    #
    #         cv2.rectangle(img, bbox, (255, 0, 255), 2)
    #         cv2.putText(img, f'{int(detection.score[0] * 100)}%{id}',
    #        (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
    #         2, (255, 0, 255), 2)


    img = show_fps(img, fps)
    cv2.imshow("Image", img)
    toc = time.time()
    curr_fps = 1.0 / (toc - tic)
    # calculate an exponentially decaying average of fps number
    fps = curr_fps if fps == 0.0 else (fps * 0.95 + curr_fps * 0.05)
    tic = toc
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()