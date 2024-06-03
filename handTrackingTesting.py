import cv2
import handTrackingModule as htm
import time

p_time = 0
c_time = 0
detector = htm.HandDetector()

cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()
    img = detector.find_hands(img, draw= False)
    lm_list = detector.find_position(img, draw= False)
    if len(lm_list) != 0:
        print(lm_list[4])

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(
        img, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2
    )

    cv2.imshow("image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()