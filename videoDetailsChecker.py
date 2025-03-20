import cv2

cap = cv2.VideoCapture(2)  # Ensure 2 is the correct index for your Iriun camera

# Get the current resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Current Resolution: {width}x{height}")
