import http.server
import socketserver
import cv2
import os
from libs.camera import VideoCamera
from libs.interactive_detection import Detections
from libs.argparser import build_argparser
import configparser

# Set up your configuration (as you did in your Flask code)
config = configparser.ConfigParser()
config.read("config.ini")

# detection control flag and parameters
is_async = eval(config.get("DEFAULT", "is_async"))
is_det = eval(config.get("DEFAULT", "is_det"))
is_reid = eval(config.get("DEFAULT", "is_reid"))
show_track = eval(config.get("TRACKER", "show_track"))
flip_code = eval(config.get("DEFAULT", "flip_code"))
resize_width = int(config.get("CAMERA", "resize_width"))

# Build your camera and detections instances
def setup():
    args = build_argparser().parse_args()
    camera = VideoCamera(args.input, resize_width, args.v4l)
    devices = [args.device, args.device_reidentification]
    detections = Detections(camera.frame, devices, args.grid)
    return camera, detections

# Generator function to yield streaming frames
def gen(camera, detections):
    frame_id = 0
    while True:
        frame_id += 1
        frame = camera.get_frame(flip_code)
        if frame is None:
            print("Video finished. Exiting...")
            os._exit(0)
        # Perform detection on the frame
        frame = detections.person_detection(
            frame, is_async, is_det, is_reid, str(frame_id), show_track
        )
        # Draw the center box
        frame = draw_center_box(frame, box_size=250)
        # Encode the frame in JPEG format
        ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()
        # Build the multipart response
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n")

def draw_center_box(frame, box_size=250):
    """
    Draws a yellow box of size box_size x box_size at the center of the frame.
    """
    frame_h, frame_w = frame.shape[:2]
    center_x = frame_w // 2
    center_y = frame_h // 2
    top_left = (center_x - box_size // 2, center_y - box_size // 2)
    bottom_right = (center_x + box_size // 2, center_y + box_size // 2)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)
    return frame

# Custom HTTP request handler
class StreamingHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                for frame in gen(self.server.camera, self.server.detections):
                    self.wfile.write(frame)
            except Exception as e:
                print("Client disconnected:", str(e))
        else:
            self.send_error(404)
            self.end_headers()

# Custom HTTP server that holds references to your camera and detections
class StreamingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, camera, detections):
        super().__init__(server_address, RequestHandlerClass)
        self.camera = camera
        self.detections = detections

def run_server(port=8000):
    camera, detections = setup()
    server_address = ('', port)
    httpd = StreamingServer(server_address, StreamingHandler, camera, detections)
    print(f"Starting server on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == '__main__':
    run_server()
