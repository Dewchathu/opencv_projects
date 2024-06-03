import cv2

# Load a pre-trained Haar cascade classifier (replace with your desired classifier)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to grayscale (may be required for some classifiers)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect objects (faces in this example)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around detected objects
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture and close windows
cap.release()
cv2.destroyAllWindows()
