import math
import cv2
import numpy as np
import time
import handTrackingModule as htm
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 720, 480
p_time = 0

detector = htm.HandDetector(detection_con=0.9)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
vol_range = volume.GetVolumeRange()
min_volume = vol_range[0]
max_volume = vol_range[1]

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
cap.set(10,100)
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()

    img = detector.find_hands(img,draw=False)
    lm_list = detector.find_position(img, draw=False)

    if len(lm_list) != 0:
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 7, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 7, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        # Map length to a volume range (50 to 300 maps to minVolume to maxVolume)
        vol = np.interp(length, [50, 250], [min_volume, max_volume])
        volBar = np.interp(length, [50, 250], [400, 150])
        volPer = np.interp(length, [50, 250], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)
        #print(vol)

        if length < 50:
            cv2.circle(img, (cx, cy), 7, (255, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(
        img, f"{int(volPer)}%", (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2
    )

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(
        img, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2
    )

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
