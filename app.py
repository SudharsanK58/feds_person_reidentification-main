#!/usr/bin/env python3
import cv2
import os
import configparser
import atexit
from flask import Flask, Response, render_template_string, request
import argparse
import time
import logging
import socket  # Needed to get local IP
import qrcode  # Needed for QR code generation
import io      # Needed for in-memory image handling
import base64  # Needed to encode image for HTML embedding

# --- Setup standard logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Attempt to import your custom libraries ---
try:
    logger.info("Importing libraries...")
    from libs.camera import VideoCamera
    from libs.interactive_detection import Detections # Assumes corrected version is present
    from libs.argparser import build_argparser
    logger.info("Libraries imported successfully.")
except ImportError as e:
    logger.error(f"Error importing libraries: {e}", exc_info=True)
    logger.error("Check 'libs' directory and ensure interactive_detection.py is present.")
    import sys
    sys.exit(1)
except FileNotFoundError as e:
     logger.error(f"Error related to library files: {e}", exc_info=True)
     import sys
     sys.exit(1)

# --- Configuration Loading ---
# (Configuration loading code remains unchanged)
config = configparser.ConfigParser()
config_file = "config.ini"
if os.path.exists(config_file):
    try:
        config.read(config_file)
        logger.info(f"Loaded configuration from {config_file}")
    except configparser.Error as e:
        logger.error(f"Error reading configuration file {config_file}: {e}")
        logger.warning("Proceeding with default fallbacks.")
        for section in ['DEFAULT', 'TRACKER', 'CAMERA', 'DETECTION', 'MODELS', 'COLORS']:
            if not config.has_section(section): config.add_section(section)
else:
    logger.warning(f"{config_file} not found. Using default fallbacks.")
    for section in ['DEFAULT', 'TRACKER', 'CAMERA', 'DETECTION', 'MODELS', 'COLORS']:
         if not config.has_section(section): config.add_section(section)

# --- Read specific flags from config ---
# (Flag reading code remains unchanged)
try:
    is_async = config.getboolean("DEFAULT", "is_async", fallback=True)
    is_det = config.getboolean("DEFAULT", "is_det", fallback=True)
    is_reid = config.getboolean("DEFAULT", "is_reid", fallback=True)
    show_track = config.getboolean("TRACKER", "show_track", fallback=True)
    flip_code_str = config.get("DEFAULT", "flip_code", fallback='None')
    flip_code = None if flip_code_str.strip().lower() == 'none' else int(flip_code_str) # Simplified
    default_resize_width = config.getint("CAMERA", "resize_width", fallback=640)
    logger.info(f"Flags: async={is_async}, det={is_det}, reid={is_reid}, track={show_track}, flip={flip_code}, resize_w={default_resize_width}")
except (ValueError, TypeError) as e:
     logger.warning(f"Invalid value type in config for flags: {e}. Using defaults.")
     is_async, is_det, is_reid, show_track, flip_code, default_resize_width = True, True, True, True, None, 640
except Exception as e:
    logger.error(f"Error processing config flags: {e}", exc_info=True)
    import sys
    sys.exit(1)

# --- Global Variables ---
# (Global variables remain unchanged)
camera = None
detections = None
parsed_args = None
SERVER_PORT = 8000

# --- Function to Draw Center Box --- (Unchanged)
def draw_center_box(frame, box_size=250):
    """
    Draws a yellow box of size box_size x box_size at the center of the frame.
    """
    frame_h, frame_w = frame.shape[:2]
    center_x = frame_w // 2
    center_y = frame_h // 2
    top_left = (center_x - box_size // 2, center_y - box_size // 2)
    bottom_right = (center_x + box_size // 2, center_y + box_size // 2)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2) # Yellow color, thickness 2
    return frame

# --- Core Setup Function --- (Unchanged)
def setup_detection():
    """Initializes Camera and Detections objects USING COMMAND-LINE ARGS."""
    global camera, detections, parsed_args, default_resize_width
    logger.info("Setting up camera and detection models...")
    parser = build_argparser()
    parsed_args = parser.parse_args()
    logger.info(f"Parsed arguments: {parsed_args}")
    current_resize_width = getattr(parsed_args, 'resize_width', default_resize_width)
    if current_resize_width is None: current_resize_width = default_resize_width
    try:
        current_resize_width = int(current_resize_width)
        if current_resize_width <= 0: raise ValueError("resize_width must be positive")
    except (ValueError, TypeError):
        logger.warning(f"Invalid resize_width '{current_resize_width}' provided or in config. Falling back to default: {default_resize_width}")
        current_resize_width = default_resize_width
    logger.info(f"Using resize_width: {current_resize_width}")
    try:
        logger.info(f"Initializing VideoCamera: input={parsed_args.input}, resize={current_resize_width}, v4l={parsed_args.v4l}")
        camera = VideoCamera(parsed_args.input, current_resize_width, parsed_args.v4l)
        initial_frame = camera.get_frame(flip_code)
        if initial_frame is None:
            raise RuntimeError(f"Could not get initial frame from camera source: {parsed_args.input}. Check source/permissions.")
        logger.info("Initial frame acquired successfully.")
    except Exception as e:
        logger.error(f"FATAL ERROR initializing camera: {e}", exc_info=True)
        if camera and hasattr(camera, 'release'): camera.release()
        camera = None
        raise
    try:
        devices = [parsed_args.device, parsed_args.device_reidentification]
        logger.info(f"Initializing Detections: devices={devices}, grid={parsed_args.grid}")
        detections = Detections(initial_frame, devices, parsed_args.grid)
        logger.info("Detections initialized successfully.")
    except Exception as e:
        logger.error(f"FATAL ERROR initializing detection models: {e}", exc_info=True)
        if camera and hasattr(camera, 'release'): camera.release()
        camera = None
        detections = None
        raise
    logger.info("Setup complete.")

# --- Video Frame Generation --- (Unchanged, still calls draw_center_box)
def gen():
    """Generator function to yield processed video frames."""
    global camera, detections, is_async, is_det, is_reid, show_track, flip_code
    if not camera or not detections:
        logger.error("gen(): Camera or Detections not initialized properly. Cannot generate frames.")
        return

    frame_id = 0
    consecutive_errors = 0
    max_consecutive_errors = 10
    request_busy_pause = 0.05

    logger.info("Starting video frame generation...")
    while True:
        frame_id += 1
        processed_frame = None
        try:
            frame = camera.get_frame(flip_code)
            if frame is None:
                logger.info(f"Frame {frame_id}: Video stream ended or camera error. Stopping generator.")
                break

            processed_frame = detections.person_detection(
                frame, is_async, is_det, is_reid, str(frame_id), show_track
            )

            if processed_frame is None:
                 logger.warning(f"Frame {frame_id}: person_detection returned None. Skipping frame.")
                 consecutive_errors += 1
                 continue

            # Draw Center Box (Call remains here)
            # processed_frame = draw_center_box(processed_frame, box_size=250)

            ret, jpeg = cv2.imencode(".jpg", processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ret:
                logger.warning(f"Frame {frame_id}: Error encoding frame to JPEG. Skipping.")
                consecutive_errors += 1
                continue

            frame_bytes = jpeg.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            consecutive_errors = 0

        except cv2.error as e:
             logger.warning(f"OpenCV error processing frame {frame_id}: {e}")
             consecutive_errors += 1; time.sleep(0.1)
        except Exception as e:
            error_str = str(e)
            logger.error(f"Error in gen() loop (frame {frame_id}): {error_str}", exc_info=True)
            consecutive_errors += 1
            if "REQUEST_BUSY" in error_str:
                logger.warning(f"Frame {frame_id}: Inference request busy. Pausing for {request_busy_pause}s.")
                time.sleep(request_busy_pause)
                if consecutive_errors > max_consecutive_errors // 2:
                     logger.error("Stopping gen() due to persistent 'REQUEST_BUSY' errors.")
                     break
            else:
                if consecutive_errors > 2:
                    logger.error(f"Stopping gen() due to repeated errors (last: {error_str}).")
                    break
        if consecutive_errors >= max_consecutive_errors:
            logger.error(f"Stopping gen() due to reaching {max_consecutive_errors} consecutive errors.")
            break
    logger.info("Video frame generator finished.")

# --- Function to Get Local IP --- (Unchanged)
def get_local_ip(target_host='8.8.8.8', target_port=80):
    """Tries to determine the local IP address used for outbound connections."""
    ip_address = '127.0.0.1' # Default fallback
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        s.connect((target_host, target_port))
        ip_address = s.getsockname()[0]
        logger.info(f"Determined local IP: {ip_address}")
    except socket.timeout:
         logger.warning(f"Timeout determining local IP via UDP to {target_host}:{target_port}. Using fallback {ip_address}.")
    except OSError as e:
         logger.warning(f"Network error determining local IP ({e}). Using fallback {ip_address}.")
    except Exception as e:
        logger.warning(f"Unexpected error determining local IP: {e}. Using fallback {ip_address}.")
    finally:
        if s:
            s.close()
    return ip_address

# --- Flask Application ---
app = Flask(__name__)

# --- HTML Template (COMMENTS REMOVED) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>FEDS Live Feed - Fare Monitoring</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/css/bootstrap.min.css">
    <style>
        body { padding-top: 70px; background-color: #f8f9fa; font-family: sans-serif; }
        .legend-box { display: inline-block; width: 15px; height: 15px; margin-right: 5px; vertical-align: middle; border: 1px solid #555; border-radius: 3px; }
        .legend-green { background-color: lime; }
        .legend-red { background-color: red; }
        #video_feed { max-width: 100%; height: auto; border: 2px solid #dee2e6; background-color: #343a40; display: block; margin: 0 auto; border-radius: 0.25rem; }
        .card-header { font-weight: bold; background-color: #6c757d; color: white; }
        .stats-label { color: #495057; font-size: 0.9rem; }
        .stats-value { font-weight: 500; color: #212529; }
        .navbar-brand svg { vertical-align: text-bottom; }
        .alert-info { background-color: #e2f3fc; border-color: #b6e0f8; color: #0c5464; }
        #qr-code-container { text-align: center; margin-top: 20px; padding: 15px; background-color: #e9ecef; border-radius: 0.25rem; }
        #qr-code-container img { max-width: 150px; height: auto; border: 1px solid #ccc; margin-bottom: 10px; }
        #qr-code-container p { margin-bottom: 5px; font-size: 0.9rem; color: #6c757d; }
        #qr-code-container code { background-color: #d1ecf1; padding: 2px 4px; border-radius: 3px; color: #0c5464;}
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark fixed-top shadow-sm">
        <a class="navbar-brand" href="#">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" class="bi bi-camera-video-fill mr-2" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M0 5a2 2 0 0 1 2-2h7.5a2 2 0 0 1 1.983 1.738l3.11-1.382A1 1 0 0 1 16 4.269v7.462a1 1 0 0 1-1.406.913l-3.11-1.382A2 2 0 0 1 9.5 13H2a2 2 0 0 1-2-2z"/></svg>
            FEDS - Live Monitoring
        </a>
    </nav>

    <div class="container-fluid mt-4">

        <div class="row mb-3 justify-content-center">
            <div class="col-lg-8 col-md-10 col-sm-12">
                 <div class="alert alert-info d-flex justify-content-center align-items-center py-2 rounded shadow-sm" role="alert">
                   <span class="font-weight-bold mr-3">Legend:</span>
                   <span class="legend-box legend-green mr-1"></span><span class="mr-3">Validated</span>
                   <span class="legend-box legend-red mr-1"></span><span>Unvalidated / Flagged</span>
                </div>
            </div>
        </div>

        <div class="row justify-content-center mb-3">
            <div class="col-lg-8 col-md-10 col-sm-12">
                <div class="card shadow-sm rounded">
                    <div class="card-header text-white rounded-top">Monitoring Point Details (Simulated)</div>
                    <ul class="list-group list-group-flush">
                        <li class="list-group-item d-flex justify-content-between align-items-center"><span class="stats-label">Location:</span><span class="stats-value">Central Station - Area 5</span></li>
                        <li class="list-group-item d-flex justify-content-between align-items-center"><span class="stats-label">Camera ID:</span><span class="stats-value">CS-A5-CAM-02</span></li>
                        <li class="list-group-item d-flex justify-content-between align-items-center"><span class="stats-label">Current Time:</span><span class="stats-value" id="current-time">Loading...</span></li>
                        <li class="list-group-item d-flex justify-content-between align-items-center"><span class="stats-label">System Status:</span><span class="badge badge-success">Active</span></li>
                        <li class="list-group-item d-flex justify-content-between align-items-center"><span class="stats-label">Last Alert Triggered:</span><span class="stats-value"><span id="last-alert-time">8</span> minutes ago</span></li>
                    </ul>
                    <div class="card-footer text-muted text-center small rounded-bottom">Real-time analysis feed.</div>
                </div>
            </div>
        </div>

        <div class="row justify-content-center">
            <div class="col-lg-8 col-md-10 col-sm-12 text-center">
                <img src="{{ url_for('video_feed') }}" alt="Live video feed loading..." id="video_feed" class="rounded shadow-sm" />
            </div>
        </div>

        <div class="row justify-content-center mt-3">
            <div class="col-lg-8 col-md-10 col-sm-12">
                <div id="qr-code-container" class="shadow-sm">
                    {% if qr_code_uri %}
                    <img src="{{ qr_code_uri }}" alt="QR Code for Local Access" />
                    <p>Scan to access on local network:</p>
                    <p><code>{{ server_url }}</code></p>
                    {% else %}
                    <p class="text-danger">Could not generate QR code.</p>
                    <p class="small">(Check console logs for errors and ensure 'qrcode[pil]' is installed)</p>
                    {% endif %}
                </div>
            </div>
        </div>

    </div> <!-- /container-fluid -->

    <footer class="text-center mt-4 mb-3 text-muted">
        <small>Fare Evader Detection System v1.5 © 2024</small>
    </footer>

    <!-- JS Libraries -->
    <script src="https://cdn.jsdelivr.net/npm/jquery@3.5.1/dist/jquery.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/js/bootstrap.min.js"></script>
    <!-- Fake Stats Update Script -->
    <script>
        function updateDynamicData() {
            // Update time
            var currentTimeEl = document.getElementById('current-time');
            if (currentTimeEl) { // Check if element exists before updating
                 currentTimeEl.textContent = new Date().toLocaleTimeString();
            }

            // Simulate last alert time changing
            var alertTimeSpan = document.getElementById('last-alert-time');
            if (alertTimeSpan) { // Check if element exists
                let currentMinutes = parseInt(alertTimeSpan.textContent) || 8;
                // Only increment occasionally
                if (Math.random() > 0.85) { currentMinutes++; }
                alertTimeSpan.textContent = currentMinutes;
            }
        }

        // Run on load and set interval
        document.addEventListener('DOMContentLoaded', (event) => {
             updateDynamicData(); // Initial call after DOM is loaded
             setInterval(updateDynamicData, 7000); // Update every 7 seconds
        });
    </script>
</body>
</html>
"""

# --- Flask Routes --- (Unchanged)
@app.route('/')
def index():
    """Serves the main HTML page with embedded QR code."""
    qr_code_uri = None
    server_url = None
    try:
        local_ip = get_local_ip()
        server_url = f"http://{local_ip}:{SERVER_PORT}/"
        logger.info(f"Generating QR code for server URL: {server_url}")
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=6, border=2)
        qr.add_data(server_url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        qr_code_uri = f"data:image/png;base64,{img_base64}"
        logger.info("QR code generated successfully.")
    except ImportError:
         logger.error("Error generating QR code: 'qrcode' or 'Pillow' library not found.", exc_info=True)
         logger.error("Please install using: pip install qrcode[pil]")
    except Exception as e:
        logger.error(f"Error generating QR code: {e}", exc_info=True)
    return render_template_string(HTML_TEMPLATE, qr_code_uri=qr_code_uri, server_url=server_url)

@app.route('/video_feed')
def video_feed():
    """Serves the processed video stream."""
    logger.info(f"Client {request.remote_addr} connected to video feed.")
    if not camera or not detections:
         logger.error("Video feed request failed: Camera/Detections not ready.")
         return Response("Processing backend not initialized or failed.", status=503, mimetype='text/plain')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Cleanup Function --- (Unchanged)
def cleanup():
    """Releases resources gracefully on exit. Registered with atexit."""
    global camera, detections
    logger.info("--- Initiating application shutdown sequence ---")
    logger.info("Attempting to release detection resources...")
    if detections and hasattr(detections, 'release') and callable(detections.release):
        try:
            logger.info("Calling detections.release()...")
            detections.release()
            logger.info("Detections resources released.")
        except Exception as e: logger.error(f"Error during detections.release(): {e}", exc_info=True)
    elif detections: logger.warning("Detections object exists but no 'release' method found or it's not callable.")
    else: logger.info("Detections object not initialized or already cleaned.")
    detections = None
    logger.info("Attempting to release camera...")
    if camera and hasattr(camera, 'release') and callable(camera.release):
        try:
            logger.info("Calling camera.release()...")
            camera.release()
            logger.info("Camera released.")
        except Exception as e: logger.error(f"Error during camera.release(): {e}", exc_info=True)
    elif camera: logger.warning("Camera object exists but no 'release' method found or it's not callable.")
    else: logger.info("Camera not initialized or already cleaned.")
    camera = None
    logger.info("--- Shutdown sequence finished ---")

atexit.register(cleanup)

# --- Main Execution --- (Unchanged)
if __name__ == '__main__':
    setup_successful = False
    try:
        setup_detection()
        setup_successful = True
    except Exception as e:
        logger.critical("--- CRITICAL SETUP FAILURE ---")
        logger.critical("Application cannot start due to errors during initialization.")

    if setup_successful:
        logger.info("--- Initialization Successful ---")
        logger.info("--- Starting Flask Server ---")
        local_ip_for_message = get_local_ip()
        logger.info(f"Access the UI locally at: http://127.0.0.1:{SERVER_PORT}/")
        if local_ip_for_message != '127.0.0.1':
            logger.info(f"Access from other devices on the network likely at: http://{local_ip_for_message}:{SERVER_PORT}/")
            logger.info("(QR code on the page points to the network IP if detected)")
        else:
            logger.warning("Could not determine local network IP; QR code may point to localhost.")
        logger.info(f"Serving on http://0.0.0.0:{SERVER_PORT} (accessible from network)")
        logger.info("Press Ctrl+C to stop the server.")

        try:
            try:
                 from waitress import serve
                 logger.info("Running with Waitress WSGI server.")
                 serve(app, host='0.0.0.0', port=SERVER_PORT, threads=10)
            except ImportError:
                 logger.warning("Waitress not found (recommend: pip install waitress).")
                 logger.warning("Falling back to Flask's built-in development server (not recommended for production).")
                 app.run(host='0.0.0.0', port=SERVER_PORT, debug=False, threaded=True, use_reloader=False)
        except KeyboardInterrupt:
            logger.info("\nKeyboardInterrupt received. Shutting down gracefully...")
        except SystemExit as e:
             logger.info(f"\nSystemExit called ({e}). Shutting down...")
        except Exception as e:
            logger.critical(f"\nAn unexpected critical error occurred during server runtime: {e}", exc_info=True)
        finally:
            logger.info("--- Flask server process has stopped ---")
    else:
        logger.info("--- Exiting application due to setup failure ---")
        cleanup()