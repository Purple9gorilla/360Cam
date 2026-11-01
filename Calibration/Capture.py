import cv2
import sys

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        sys.exit()

    print("Camera accessed successfully. Press 'q' to quit.")

    while True:
        # 2. Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame wasn't grabbed correctly, break the loop
        if not ret:
            print("Error: Can't receive frame.")
            break

        # 3. Display the resulting frame in a window
        cv2.imshow('Live Camera Feed', frame)

        # 4. Break the loop if 'q' key is pressed
        # cv2.waitKey(1) waits for 1ms for a key event
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Live Camera Feed', cv2.WND_PROP_VISIBLE) < 1:
            break
    # 5. When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()