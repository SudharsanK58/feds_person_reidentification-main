import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} is available.")
    else:
        print(f"Camera index {i} is NOT available.")
    cap.release()
