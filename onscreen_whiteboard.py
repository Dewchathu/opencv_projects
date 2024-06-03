import time
import cv2
import numpy as np
from handTrackingModule import HandDetector

# Color palette, brush size, and clear button coordinates and size
palette = {
    'Red': (0, 0, 255),
    'Green': (0, 255, 0),
    'Blue': (255, 0, 0),
    'Eraser': (0, 0, 0)
}
palette_positions = [(20, 20), (20, 80), (20, 140), (20, 200)]
brush_sizes = [5, 10, 15, 20, 25]
brush_positions = [(80 + i * 60, 20) for i in range(len(brush_sizes))]
clear_button_pos = (20, 260)
box_size = 40

# New window size
window_width = 800
window_height = 600


def draw_control_panel(img):
    # Draw color palette
    for i, color in enumerate(palette.values()):
        cv2.rectangle(img, palette_positions[i],
                      (palette_positions[i][0] + box_size, palette_positions[i][1] + box_size), color, cv2.FILLED)

    # Draw brush size options
    for i, size in enumerate(brush_sizes):
        cv2.rectangle(img, brush_positions[i], (brush_positions[i][0] + box_size, brush_positions[i][1] + box_size),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(img, str(size), (brush_positions[i][0] + 5, brush_positions[i][1] + 30), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 0, 0), 2)

    # Draw clear button
    cv2.rectangle(img, clear_button_pos, (clear_button_pos[0] + box_size, clear_button_pos[1] + box_size),
                  (0, 255, 255), cv2.FILLED)
    cv2.putText(img, "Clear", (clear_button_pos[0] + 5, clear_button_pos[1] + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0),
                2)

    return img
def main():
    p_time = 0
    c_time = 0
    detector = HandDetector()
    cap = cv2.VideoCapture(0)

    # Initialize a blank canvas
    canvas = None

    current_color = (0, 0, 255)  # Default color is red
    eraser_mode = False
    brush_size = 15
    prev_x, prev_y = 0, 0

    # Set window size
    cv2.namedWindow("Whiteboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Whiteboard", window_width, window_height)

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)  # Flip the image horizontally

        # Initialize canvas size
        if canvas is None:
            canvas = np.zeros_like(img)

        img = detector.find_hands(img)
        lm_list = detector.find_position(img, draw=False)

        # Draw the control panel
        img = draw_control_panel(img)

        if len(lm_list) != 0:
            # Index finger tip
            x1, y1 = lm_list[8][1], lm_list[8][2]

            # Thumb tip
            x2, y2 = lm_list[4][1], lm_list[4][2]

            # Check if index finger is up
            fingers_up = lm_list[8][2] < lm_list[6][2]

            if fingers_up:
                cv2.circle(img, (x1, y1), brush_size, current_color, cv2.FILLED)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x1, y1

                if lm_list[12][2] > lm_list[10][2]:  # If middle finger is down, draw
                    if eraser_mode:
                        cv2.line(canvas, (prev_x, prev_y), (x1, y1), (0, 0, 0), brush_size + 10)
                    else:
                        cv2.line(canvas, (prev_x, prev_y), (x1, y1), current_color, brush_size)

                prev_x, prev_y = x1, y1
            else:
                prev_x, prev_y = 0, 0

            # Color selection logic
            if lm_list[4][1] < 60:  # Thumb near the palette
                for i, pos in enumerate(palette_positions):
                    if pos[1] < lm_list[4][2] < pos[1] + box_size:
                        if i == len(palette_positions) - 1:  # Eraser
                            eraser_mode = True
                        else:
                            current_color = list(palette.values())[i]
                            eraser_mode = False

            # Brush size selection logic
            if lm_list[4][1] < 60:  # Thumb near the brush size control
                for i, pos in enumerate(brush_positions):
                    if pos[1] < lm_list[4][2] < pos[1] + box_size:
                        brush_size = brush_sizes[i]

            # Clear canvas logic
            if clear_button_pos[1] < lm_list[4][2] < clear_button_pos[1] + box_size and clear_button_pos[0] < \
                    lm_list[4][1] < clear_button_pos[0] + box_size:
                canvas = np.zeros_like(img)

        # Resize img_inv to match the size of img
        img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, canvas)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(
            img, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2
        )

        cv2.putText(
            img, f"Brush Size: {brush_size}", (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2
        )

        cv2.imshow("Whiteboard", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()