import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from gtts import gTTS
from playsound import playsound

video = cv2.VideoCapture(0)
# labels = []

while True:
    ret, frame = video.read()
    bbox, label, conf = cv.detect_common_objects(frame)
    output_image = draw_bbox(frame, bbox, label, conf)

    cv2.imshow("Object Detection", output_image)

    # for item in label:
    #     if item in labels:
    #         pass

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
