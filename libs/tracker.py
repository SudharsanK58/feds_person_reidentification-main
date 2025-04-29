from logging import getLogger
from time import sleep
import cv2
import numpy as np
from munkres import Munkres
from timeit import default_timer as timer
from libs.utils import cos_similarity
from libs.utils import get_iou, get_iou2
from libs.utils import get_euclidean_distance
from libs.utils import get_mahalanobis_distance
from libs.utils import affine_translation
from libs.utils import get_box_coordinates
import pickle as pkl
import configparser
from scipy import stats
from timeit import default_timer as timer
import paho.mqtt.client as mqtt
from threading import Thread

from libs.kalman_filter import KalmanFilter

logger = getLogger(__name__)

config = configparser.ConfigParser()
config.read("config.ini")

# Tracking parameters
reid_limit = eval(config.get("TRACKER", "reid_limit"))
sim_thld = eval(config.get("TRACKER", "sim_thld"))
min_sim_thld = eval(config.get("TRACKER", "min_sim_thld"))
skip_iou_thld = eval(config.get("TRACKER", "skip_iou_thld"))
box_iou_thld = eval(config.get("TRACKER", "box_iou_thld"))
save_points = eval(config.get("TRACKER", "save_points"))
max_grid = eval(config.get("TRACKER", "max_grid"))
lost_thld = eval(config.get("TRACKER", "lost_thld"))
hold_track = eval(config.get("TRACKER", "hold_track"))
show_track = eval(config.get("TRACKER", "show_track"))

# basic colors
green = eval(config.get("COLORS", "green"))
skyblue = eval(config.get("COLORS", "skyblue"))
red = eval(config.get("COLORS", "red"))
light_green = eval(config.get("COLORS", "LIGHT_GREEN"))

# track state
TENTATIVE = 1
CONFIRMED = 2
DELETED = 3

# Dictionary to track which IDs should be green
green_tagged_ids = set()

# Define the expected MQTT payloads and their corresponding section indices
PAYLOAD_SECTION_MAP = {
    "201010#Sudharsan#3#12345678#0-3 feet": 0,
    "201010#Sudharsan#3#12345678#3-6 feet": 1,
    "201010#Sudharsan#3#12345678#6-9 feet": 2,
    "201010#Sudharsan#3#12345678#9-12 feet": 3,
}


# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT server with result code {rc}")
    # Subscribe to the specific topic
    client.subscribe("04:e9:e5:16:f7:1b/nfc")

def on_message(client, userdata, msg):
    global green_tagged_ids
    data = msg.payload.decode().strip()
    print(f"Received data: {data}")

    # Check if the received data is one of the expected payloads
    if data in PAYLOAD_SECTION_MAP:
        print(f"Received expected payload: {data}")

        # Get the corresponding section index from the map
        target_section_index = PAYLOAD_SECTION_MAP[data]

        # Access the tracker instance correctly
        tracker = userdata['tracker']

        # Get the segment width calculated by the tracker
        segment_width = tracker.segment_width
        frame_w = tracker.frame_w

        # Calculate the x-coordinate range for the target section
        min_x = target_section_index * segment_width
        # The max_x for the last section should go up to the frame width
        max_x = (target_section_index + 1) * segment_width
        if target_section_index == tracker.num_distance_sections - 1: # Check if it's the last section
             max_x = frame_w

        print(f"Checking for persons in section index {target_section_index} (x range: {min_x} to {max_x})")

        # Iterate through all active tracks
        for track in tracker.tracks:
            # Only consider confirmed tracks that have been assigned an ID
            if track.stats == CONFIRMED and track.person_id is not None:
                # Get the last known position of the track
                # Ensure track_points is not empty
                if tracker.track_points[track.track_id]:
                    center = tracker.track_points[track.track_id][-1] # (x, y) coordinate

                    # Check if the person's center is inside the target section
                    # The x-coordinate must be within the calculated range (inclusive)
                    if min_x <= center[0] <= max_x:
                        # Condition 1: Person is in the correct section
                        # Condition 2: MQTT payload matched (checked by the outer if)

                        # If both conditions are met, tag this person ID as green
                        green_tagged_ids.add(track.person_id)
                        print(f"Person ID {track.person_id} tagged as GREEN (in section {target_section_index} and MQTT matched)")
                    # else:
                    #     print(f"Person ID {track.person_id} is NOT inside the target section ({target_section_index})")
                # else:
                #      print(f"Track {track.track_id} has no track points.")
    # else:
    #     print(f"Received unexpected payload: {data}")


# MQTT Initialization
def start_mqtt(tracker):
    mqtt_client = mqtt.Client(userdata={'tracker': tracker})
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.connect("mqtt.zig-web.com", 1883, 60)

    # Run MQTT in a separate thread so it doesn’t block the main loop
    mqtt_thread = Thread(target=mqtt_client.loop_forever)
    mqtt_thread.start()


class Person:
    pass


class Track:
    # Based on:
    # https://github.com/nwojke/deep_sort/blob/master/deep_sort/track.py
    #
    # hists : count up when detected person was matched
    # miss  : count up when detected person was lost
    # stats : 1: Tentative Confirmed = 2 Deleted = 3
    def __init__(self, track_id, center):
        self.track_id = track_id
        self.person_id = None
        self.hits = 0
        self.miss = 0
        self.stats = TENTATIVE
        self.kf = KalmanFilter(center)
        self.is_matched = False
        self.direction = None

    def update(self):
        self.hits += 1
        self.miss = 0
        self.is_matched = True

    def lost(self):
        self.hits = 0
        self.miss += 1
        self.is_matched = False


class Tracker:
    def __init__(self, detector, frame, grid):
        # initialize tracker parameters
        self.frame = frame  # Store the current frame for reference
        start_mqtt(self)
        self.person_id_detector = detector
        self.frame_id = 0
        self.show_track = show_track
        self.frame_h, self.frame_w = frame.shape[:2]
        self.person_id = 0
        self.track_vecs = None
        self.track_boxes = []
        self.prev_feature_vecs = None
        self.prev_track_boxes = None
        self.track_points = []
        self.track_points_measured = []
        self.euc_distances = []
        self.tracks = []
        self.tracker_prev_time = timer()
        self.tracker_accum_time = 0
        self.counter_stats = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        self.colors = pkl.load(open("pallete", "rb"))
        self.params = f"@Track sim:{sim_thld} min_sim:{min_sim_thld} skip_iou:{skip_iou_thld} box_iou:{box_iou_thld} grid:{grid} lost:{lost_thld} hold_track:{hold_track} show_track:{show_track}"
        self.enable_count = None
        # set tracker boundary and counter ranage ing a frame
        self.grid, self.enable_count = self._set_grid(grid)
        self.track_range = self._set_track_range(frame, self.grid)
        self.m = Munkres()

        # Calculate and store segment width for distance lines
        self.num_distance_sections = 4 # 0-3, 3-6, 6-9, 9-12
        self.segment_width = self.frame_w // self.num_distance_sections


    def _next_id(self):
        self.person_id += 1
        return self.person_id


    def _set_grid(self, grid):
        # to counter person minimum grid need to be grater than or equal to  3
        if 3 <= grid <= max_grid:
            return grid, True
        elif grid > max_grid:
            return max_grid, True
        else:
            return max_grid, False

    def _set_track_range(self, frame, grid):
        frame_h = frame.shape[0]
        frame_w = frame.shape[1]

        margin_w, margin_h = frame_w // grid, frame_h // grid
        margin = margin_w if margin_w < margin_h else margin_h

        top_lx, top_ly = margin, margin
        top_rx, top_ry = frame_w - margin, margin
        bottom_lx, bottom_ly = margin, frame_h - margin
        bottom_rx, bottom_ry = frame_w - margin, frame_h - margin

        top_left = (top_lx, top_ly)
        top_right = (top_rx, top_ry)
        bottom_left = (bottom_lx, bottom_ly)
        bottom_right = (bottom_rx, bottom_ry)

        track_range = {}
        track_range["top_left"] = top_left
        track_range["top_right"] = top_right
        track_range["bottom_left"] = bottom_left
        track_range["bottom_right"] = bottom_right

        return track_range

    def kalman_filter(self, track_id, center: tuple, update: bool) -> tuple:

        kf = self.tracks[track_id].kf
        kf.predict()

        # filtere the observated value when update
        if update:
            kf.update(center)

        cx, cy, vx, vy = kf.X.flatten().astype(int)
        center = cx, cy
        velocity = vx, vy

        message = f"frame_id:{self.frame_id} track_id:{track_id} center:{center} velocity:{velocity}"
        logger.debug(message)

        return center, velocity

    def is_out_of_track_area(self, center: tuple) -> bool:
        top_left = self.track_range["top_left"]
        bottom_right = self.track_range["bottom_right"]
        out_of_top = center[0] < top_left[0]
        out_of_right = center[1] > bottom_right[1]
        out_of_bottom = center[0] > bottom_right[0]
        out_of_left = center[1] < top_left[1]
        return out_of_top or out_of_right or out_of_bottom or out_of_left

    def is_out_of_frame(self, center) -> bool:
        cx, cy = center
        if (cx <= 0 or cx >= self.frame_w) or (cy <= 0 or cy >= self.frame_h):
            return True
        return False

    def get_feature_vecs(self, person_frames):
        feature_vecs = np.zeros((len(person_frames), 256))
        for det_id, person_frame in enumerate(person_frames):
            self.person_id_detector.infer(person_frame)
            feature_vec = self.person_id_detector.get_results()
            feature_vecs[det_id] = feature_vec
        return feature_vecs

    def disable_tracking(self, track_id: int):
        if hold_track:
            return

        self.track_vecs[track_id] = np.zeros((1, 256))
        self.track_boxes[track_id] = [(np.nan, np.nan, np.nan, np.nan)]
        self.track_points[track_id] = [(np.nan, np.nan)]
        self.tracks[track_id].stats = DELETED
        logger.debug(
            f"frame_id:{self.frame_id} disabled person_id:{self.tracks[track_id].person_id}(track_id:{track_id}) {self.tracks[track_id].__dict__}"
        )

    def register_person(self, det_id: int, detection):

        track = self.tracks[detection.track_id]
        if track.stats != CONFIRMED:
            return

        center = self.get_center(detection.box)
        self.track_points.append([center])
        self.track_points_measured.append([center])
        self.track_vecs = np.vstack((self.track_vecs, detection.feature_vec))
        self.track_boxes.append([detection.box])
        self.euc_distances.append([0.0])

        # Create track instance
        track_id = len(self.tracks)
        self.tracks.append(Track(track_id, center))

        header = f"frame_id:{self.frame_id} registered det_id:{det_id} to track_id:{track_id} track:{self.tracks[track_id].__dict__}"
        info1 = f"conf:{detection.confidence:.3f}"
        logger.info(f"{header} {info1}")

    def get_counter_stats(self, track_id: int, direction: str):
        # Count the number of persons who have gone out of the counter area,
        # based on the current position of the detected person and the direction in which
        # they have moved.
        # Set the name of the direction and the names of the four corners associated with
        # the direction to be counted.
        direction_x, direction_y = 0, 1
        direction_corner_dict = {
            "top": ("top_right", direction_y),
            "bottom": ("bottom_right", direction_y),
            "left": ("top_left", direction_x),
            "right": ("top_right", direction_x),
        }
        # Get three coordinates of track points:
        # Start   : center coordinate when person re-identification started
        # End     : center coordinate when person re-identification ended
        # boundary: counter boundary coordinate which "_set_track_range" defined
        track_points = self.track_points[track_id]
        if not track_points: # Handle empty track points list
             return False
        start_tmp = track_points[0]
        end_tmp = track_points[-1]
        corner_name, count_direction = direction_corner_dict[direction]
        boundary = self.track_range[corner_name][count_direction]
        start = start_tmp[count_direction]
        end = end_tmp[count_direction]

        # Check direction and count up by the direction
        out_of_track = False
        track = self.tracks[track_id]
        if direction in ["right", "bottom"]:
            out_of_track = start <= boundary <= end
            message = f"frame_id:{self.frame_id} {direction}: person_id:{track.person_id} ({start} <= {boundary} <= {end})"
        if direction in ["top", "left"]:
            out_of_track = start >= boundary >= end
            message = f"frame_id:{self.frame_id} {direction}: person_id:{track.person_id} ({start} >= {boundary} >= {end})"

        if out_of_track and track.direction != direction:
            self.counter_stats[direction] += 1
            track.direction = direction
            logger.debug(message)
            return out_of_track
        return out_of_track

    def count_person(self, track_id):
        track_points = self.track_points[track_id]
        if track_points is None or len(track_points) < 3:
            return

        for direction in ["top", "bottom", "left", "right"]:
            if self.get_counter_stats(track_id, direction):
                self.disable_tracking(track_id)

    def draw_counter_stats(self, frame):
        if not self.enable_count:
            return frame

        # Get each corner's coordinate
        top_left = self.track_range["top_left"]
        top_right = self.track_range["top_right"]
        bottom_left = self.track_range["bottom_left"]
        bottom_right = self.track_range["bottom_right"]
        # Draw countable area
        cv2.line(frame, top_left, top_right, skyblue, 1)
        cv2.line(frame, top_left, bottom_left, skyblue, 1)
        cv2.line(frame, bottom_left, bottom_right, skyblue, 1)
        cv2.line(frame, bottom_right, top_right, skyblue, 1)
        # Draw counter stats
        counter_info = ""
        for direction in ["right", "bottom", "left", "top"]:
            counter_info = (
                f"{direction}:{self.counter_stats[direction]} " + counter_info
            )
        cv2.putText(
            frame,
            counter_info,
            (20, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        return frame

    def draw_params(self, frame):
        cv2.putText(
            frame,
            self.params,
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 10, 10),
            1,
        )
        return frame

    def draw_det_box(self, frame, det_id, box, color=(0, 255, 0)):
        xmin, ymin, xmax, ymax = box
        cv2.putText(
            frame,
            str(det_id),
            (xmax - 3, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )
        cv2.rectangle(
            frame, (xmin, ymin), (xmax, ymax), color, 1,
        )
        return frame


    def draw_reid_box(self, frame, track_id, box, conf, color):
        track = self.tracks[track_id]

        # If the track's ID is in the green_tagged_ids list, make it green
        if track.person_id is not None and track.person_id in green_tagged_ids:
            color = green
        else:
            color = red

        # Draw ID on the detected person
        # Ensure track_points is not empty before accessing the last element
        if self.track_points[track_id]:
            x = self.track_points[track_id][-1][0]
            y = self.track_points[track_id][-1][1]
            cv2.putText(
                frame,
                str(track.person_id),
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        xmin, ymin, xmax, ymax = box
        ymax += 30
        text = f"ID: {track.person_id}"
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        xtext = xmin + size[0][0] + 15

        cv2.rectangle(frame, (xmin, ymin - 22), (xtext, ymin), color, -1)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.rectangle(frame, (xmin, ymin - 22), (xtext, ymin), color)
        cv2.putText(
            frame,
            text,
            (xmin + 3, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        return frame


    def draw_track_points(self, frame, track_points, color):
        if self.show_track:
            track_points = np.array(track_points)
            cv2.polylines(
                frame, [track_points], isClosed=False, color=color, thickness=2,
            )
        return frame

    def draw_track_info(self, frame, track_id, box=None, confidence=None):
        # Draw traking info only when track states is confirmed
        track = self.tracks[track_id]
        if track.stats != CONFIRMED:
            return frame

        # Get person's color for draw rectrangle and tracked points
        # color is now determined inside draw_reid_box based on green_tagged_ids
        # We still call get_color here, but the actual color used for the box
        # is decided in draw_reid_box. Let's get the color for the track points.
        track_point_color = green if track.person_id is not None and track.person_id in green_tagged_ids else self.get_color(track.person_id)


        # Set similarity as confidence
        conf = f"{track.person_id}" if confidence else "lost.."

        # Draw reid box (color determined inside this function)
        frame = self.draw_reid_box(frame, track_id, box, conf, track_point_color) # Pass track_point_color, though draw_reid_box recalculates it

        # Draw track porints
        tp = np.array(self.track_points[track_id])
        track_points = tp[~np.isnan(tp).any(axis=1)].astype(int)
        if len(track_points) > 2:
            # Use the color determined for track points
            frame = self.draw_track_points(frame, track_points, track_point_color)

        # Count a person by the direction when the person is out of counter area
        if track_points.any():
            if self.enable_count and self.is_out_of_track_area(track_points[-1]):
                self.count_person(track_id)
        return frame

    def draw_distance_lines(self, frame):
        """Draws 3 vertical lines (invisibly) and distance labels for 4 sections on the frame."""
        frame_h, frame_w = frame.shape[:2]
        num_sections = self.num_distance_sections # Use the stored attribute
        num_lines = num_sections - 1 # This means 3 lines
        segment_width = self.segment_width # Use the stored attribute

        # line_color = (0, 255, 255) # Yellow in BGR - No longer used for drawing lines
        # line_thickness = 1 # No longer used for drawing lines

        labels = ["0-3 Feet", "3-6 Feet", "6-9 Feet", "9-12 Feet"]
        label_color = (255, 255, 255) # White
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        label_y_offset = 20 # Pixels from the top

        # Draw the 3 lines (commented out to make them "transparent")
        # for i in range(num_lines):
        #     line_x = (i + 1) * segment_width
        #     pt1 = (line_x, 0)
        #     pt2 = (line_x, frame_h)
        #     cv2.line(frame, pt1, pt2, line_color, line_thickness)

        # Draw the labels for the 4 sections
        for i in range(num_sections):
            label_text = labels[i]
            # Calculate text size to center it
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            # Center of section i is between i*segment_width and (i+1)*segment_width
            section_center_x = (i * segment_width) + segment_width // 2
            label_x = section_center_x - text_w // 2
            label_org = (label_x, label_y_offset + text_h) # Place text baseline at y_offset + text_h

            # Ensure label is not drawn outside the frame (especially horizontally)
            label_org = (max(0, label_x), label_y_offset + text_h)
            # Also ensure it doesn't go past the right edge
            if label_org[0] + text_w > frame_w:
                 label_org = (frame_w - text_w, label_org[1])


            cv2.putText(frame, label_text, label_org, font, font_scale, label_color, font_thickness)

        return frame


    def first_detection(self, feature_vecs, boxes):

        self.track_vecs = feature_vecs
        self.prev_feature_vecs = feature_vecs
        self.prev_track_boxes = boxes
        for box in boxes:
            self.track_boxes.append([box])
            center = self.get_center(box)
            self.track_points.append([center])
            self.track_points_measured.append([center])
            self.euc_distances.append([0.0])
            # Create track instance
            track_id = len(self.tracks)
            self.tracks.append(Track(track_id, center))

    def get_center(self, box: tuple) -> tuple:
        # cy, cy : center coordinate of a person
        xmin, ymin, xmax, ymax = box
        cx = (xmax - xmin) / 2 + xmin
        cy = (ymax - ymin) / 2 + ymin
        return cx, cy

    def get_color(self, person_id: int) -> tuple:
        # Difine person's color from Pallete (self.colors) with 100 colors.
        # The 100th person_id will use the first index's color again.
        if person_id is None: # Handle case before person_id is assigned
             return (0,0,0) # Default color like black
        color_id = person_id - len(self.colors) * (person_id // len(self.colors))
        return self.colors[color_id]

    def get_box_info(self, det_id: int, boxes, feature_vecs) -> tuple:
        box = boxes[det_id]
        center = self.get_center(box)
        feature_vec = feature_vecs[det_id].reshape(1, 256)
        return box, center, feature_vec

    def get_box_iou(self, box: tuple, reid_box: tuple) -> float:
        if np.nan in reid_box:
            return 0.0
        det_box = affine_translation(box)
        reid_box = affine_translation(reid_box)
        return get_iou(det_box, reid_box)

    def get_person_info(self, det_id, track_id, confidences, boxes, feature_vecs):
        # 1. Get the highest cosine similarity as confidence
        confidence = confidences[det_id]

        # 2. Get detected bbox information
        box, center, feature_vec = self.get_box_info(det_id, boxes, feature_vecs)

        # 3. Get distance between dettected box and track box
        # and check if thease boxes are at the closest to location in last track points.
        # Ensure track_points is not empty before accessing the last element
        if not self.track_points[track_id]:
             euc_dist = 0.0 # Or some default/error value
        else:
            track_point = np.array(self.track_points[track_id][-1]).reshape(-1, 2)
            euc_dist = get_euclidean_distance(center, track_point).item(0)


        # 5. Get box iou between detected box and track box
        track_box = self.track_boxes[track_id][-1]
        box_iou = self.get_box_iou(box, track_box)

        # Create an instance of detected person with gathered infomation
        detection = Person()
        detection.track_id = track_id
        detection.confidence = confidence
        detection.feature_vec = feature_vec
        detection.box = box
        detection.box_w = box[2] - box[0]  # xmax - xmin
        detection.box_h = box[3] - box[1]  # ymax - ymin
        detection.center = center
        detection.euc_dist = euc_dist
        detection.box_iou = box_iou

        return detection

    def is_overlapped(self, box, boxes):
        # Check if the bounding box (bbox) overlaps with others.
        # If IoU is greater than iou_thld, it is in a state of overlap with others.
        for box_ in boxes:
            box_iou = get_iou(box, box_)
            if box_iou > skip_iou_thld:
                return True
        return False

    def solve_occlusion_problem(self, frame, det_id, detection, boxes):
        # If a detected box is overlapped with the other boxes, draw skyblue box over them.
        # Create a temporary list excluding the current box for overlap check
        temp_boxes = boxes[:det_id] + boxes[det_id+1:]
        if self.is_overlapped(detection.box, temp_boxes):
            # Draw skyblue ractangle over the detected person
            frame = self.draw_det_box(frame, det_id, detection.box, color=(255, 255, 0))
            return frame, True
        else:
            # Draw green ractangle over the detected person
            ##frame = self.draw_det_box(frame, det_id, detection.box, color=(0, 255, 0))
            return frame, False

    def evaluate_euc_distance(self, detection, n=2):
        # Exclude initial value
        euc_distances = self.euc_distances[detection.track_id][1:]
        # Get last 30 data
        n = len(euc_distances) - 30
        euc_distances = np.array(euc_distances[n:])

        if euc_distances.size > 3:
            mean = euc_distances.mean()
            std = stats.tstd(euc_distances)
            # min, max = stats.norm.interval(alpha=0.95, loc=mean, scale=std)
            min, max = stats.norm.interval(alpha=0.99, loc=mean, scale=std)
        else:
            mean, std = 0.0, 0.0
            min, max = 0.0, detection.box_w # Use box width as a fallback max distance

        # Memo: After all, if the Euclidean distance is smaller than the width of the bbox,
        # the distance is considered valid.
        # (Because the 95% confidence interval is not accurate enough for matching.)
        ## is_valid_dist = 0 < detection.euc_dist < max
        is_valid_dist = detection.euc_dist < detection.box_w

        detection.euc_dist_min = min
        detection.euc_dist_max = max
        detection.euc_dist_mean = mean
        detection.euc_dist_std = std

        return is_valid_dist

    def evaluate_mah_distance(self, detection):
        track_points = self.track_points[detection.track_id].copy()
        # Get last 30 points
        n = len(track_points) - 30
        track_points = np.array(track_points)[n:]

        if track_points.size > 3:
            mah_dist = get_mahalanobis_distance(detection.center, track_points)
        else:
            mah_dist = 1.0

        # Mahalanobis distance threshold
        # https://github.com/nwojke/deep_sort/blob/master/deep_sort/kalman_filter.py
        # In thi program, get Mahalanobis distance from center position (x,y)
        is_valid_mah_dist = mah_dist < 5.9915
        ##is_valid_mah_dist = mah_dist < 9.4877

        detection.mah_dist = mah_dist

        return is_valid_mah_dist

    def evaluate(self, detection):
        update = False

        # confidence
        detection.is_valid_conf = detection.confidence > min_sim_thld
        # euclidean distance
        detection.is_valid_dist = self.evaluate_euc_distance(detection)
        # shape
        detection.is_valid_iou = detection.box_iou > box_iou_thld
        # mahalanobis distance
        ##detection.is_valid_mah_dist = self.evaluate_mah_distance(detection)

        update = (
            detection.is_valid_conf
            # and detection.is_valid_mah_dist
            and detection.is_valid_dist
        )
        return update

    def update(self, frame, detection):
        track_id = detection.track_id
        track = self.tracks[track_id]

        # 1. Update feature vector
        self.track_vecs[track_id] = detection.feature_vec

        # 2. Update bouding box
        # *Replace* predicted box added in preprocess with detected person box
        if self.track_boxes[track_id]: # Ensure list is not empty before popping
            self.track_boxes[track_id].pop(-1)
        self.track_boxes[track_id].append(detection.box)

        # 3. Apply kalmanfilter and update a track point
        center = detection.center

        # *Replace* predicted center added in preprocess with filtered person center
        if self.track_points[track_id]: # Ensure list is not empty before popping
            self.track_points[track_id].pop(-1)
        center, _ = self.kalman_filter(track_id, center, update=True)
        self.track_points[track_id].append(center)

        # 4. Assign new person_id to the track with three consecutive matches and set its status to CONFIRMED
        track.update()
        # If the track matched three times in a row, the status is set to CONFIRMED.
        if track.stats == TENTATIVE and track.hits > 3:
            track.stats = CONFIRMED
            track.person_id = self._next_id()

        # 5. Add euclidean distance which is used evaluate()
        self.euc_distances[track_id].append(detection.euc_dist)

        # 6. Draw tracked information into the frame
        frame = self.draw_track_info(
            frame, track_id, detection.box, detection.confidence
        )

        return frame

    def not_found(self, det_id, detection):

        # not register persons if they are out of tracking area
        if self.is_out_of_track_area(detection.center):
            message = f"frame_id:{self.frame_id} det_id:{det_id} out of couter area"
            logger.debug(message)
            return

        # Register as a new person if the match was poor AND the box doesn't overlap well
        if not detection.is_valid_iou:
            message = f"frame_id:{self.frame_id} det_id:{det_id} track_id:{detection.track_id} box_iou:{detection.box_iou} - Registering as new person"
            logger.debug(message)
            # Create a *new* track for this detection
            center = self.get_center(detection.box)
            self.track_points.append([center])
            self.track_points_measured.append([center])
            # Need to append the feature vector to self.track_vecs
            if self.track_vecs is None:
                 self.track_vecs = detection.feature_vec
            else:
                 self.track_vecs = np.vstack((self.track_vecs, detection.feature_vec))
            self.track_boxes.append([detection.box])
            self.euc_distances.append([0.0])
            new_track_id = len(self.tracks) # Get the next available track_id
            self.tracks.append(Track(new_track_id, center))
            logger.info(f"frame_id:{self.frame_id} Registered new person from det_id:{det_id} to new track_id:{new_track_id}")
        else:
             # If match was poor but box_iou was high, it might be the same person but
             # the re-id feature vector was bad, or the distance metric failed.
             # In this case, we don't register a new person, and the original track
             # (detection.track_id) will be marked as lost in the main loop.
             message = f"frame_id:{self.frame_id} det_id:{det_id} track_id:{detection.track_id} box_iou:{detection.box_iou} - Match failed, but high IoU. Not registering new person."
             logger.debug(message)


    def lost(self, frame, track):
        # 1. Disable tracking when lost counter exceeded lost thld
        track_id = track.track_id
        if track.miss > lost_thld:
            self.disable_tracking(track_id)
            return frame

        # 2. Count up lost counter (track.miss +1)
        track.lost()

        # 3. draw tracking information into the frame excluding tentative track
        if track.stats == CONFIRMED:
            # Ensure track_boxes is not empty before accessing the last element
            if self.track_boxes[track.track_id]:
                pred_box = self.track_boxes[track.track_id][-1]
                frame = self.draw_track_info(frame, track_id, box=pred_box, confidence=None)
            else:
                 # If track_boxes is empty, maybe just draw the track points if available
                 if self.track_points[track_id]:
                      # This case might need specific drawing logic if no box is available
                      pass # For now, do nothing if no box to draw

        return frame

    def preprocess(self):
        # Initialize active track idx
        active_track_ids = [t.track_id for t in self.tracks if t.stats != DELETED]

        for track_id in active_track_ids:
            track = self.tracks[track_id]
            # initialize matched state which is used for "lost_track_ids"
            track.is_matched = False

            # predict step with kalman filter and get center coodinate
            center, _ = self.kalman_filter(track_id, None, update=False)
            self.track_points[track_id].append(center)

            # Predict box from previous person boxes
            # Ensure track_boxes is not empty before accessing the last element
            if self.track_boxes[track_id]:
                prev_box = self.track_boxes[track_id][-1]
                box = get_box_coordinates(prev_box, center)
                box = np.array(box, dtype=int)
                self.track_boxes[track_id].append(tuple(box))
            else:
                 # If no previous box, append a placeholder or handle differently
                 self.track_boxes[track_id].append((np.nan, np.nan, np.nan, np.nan))


            # Disable the track when the center predicted by kalmanfilter is out of frame
            if self.is_out_of_frame(center):
                message = f"frame_id:{self.frame_id} person_id:{track.person_id}(track_id:{track.track_id}) is out of frame"
                logger.debug(message)
                self.disable_tracking(track.track_id)
                continue

            # 3. Slice track points and boxes  when the number of them exceeds save_points
            track_points = self.track_points[track_id]
            if len(track_points) > save_points:
                n = len(track_points) - save_points
                self.track_points[track_id] = track_points[n:]

            track_boxes = self.track_boxes[track_id]
            if len(track_boxes) > save_points:
                n = len(track_boxes) - save_points
                self.track_boxes[track_id] = track_boxes[n:]

        # Get active track idx again to remove "DELETE" track in "is_out_out_frame" check
        active_track_ids = [t.track_id for t in self.tracks if t.stats != DELETED]

        return active_track_ids

    def person_reidentification(self, frame, person_frames, boxes):

        active_track_ids = self.preprocess()

        # If no person and no active track, return frame.
        # If no person is present but there are no active tracks, lost() should be done.
        if not person_frames and not active_track_ids:
            frame = self.draw_params(frame)
            frame = self.draw_counter_stats(frame)
            frame = self.draw_distance_lines(frame) # Draw lines even with no detections/tracks
            return frame

        if not person_frames and active_track_ids:
            for track_id in active_track_ids:
                track = self.tracks[track_id]
                frame = self.lost(frame, track)
            frame = self.draw_params(frame)
            frame = self.draw_counter_stats(frame)
            frame = self.draw_distance_lines(frame) # Draw lines
            return frame

        # The first feature vectors registration
        feature_vecs = self.get_feature_vecs(person_frames[:reid_limit])
        if self.track_vecs is None:
            self.first_detection(feature_vecs, boxes)
            # After first detection, draw info and lines
            for track_id in range(len(self.tracks)):
                 track = self.tracks[track_id]
                 # Draw info for newly created tracks (they are TENTATIVE)
                 # draw_track_info only draws CONFIRMED, so this won't draw boxes yet.
                 # If you want to draw TENTATIVE boxes, modify draw_track_info or add new logic here.
                 # For now, let's rely on the main loop drawing after confirmation.
                 pass # No drawing needed immediately after first_detection

            frame = self.draw_params(frame)
            frame = self.draw_counter_stats(frame)
            frame = self.draw_distance_lines(frame) # Draw lines
            return frame


        # ----------- Preprocess -------------------- #
        frame_id = f"frame_id:{self.frame_id}"

        # Cost Matrix
        # Compare the cosine similarity between the detected person's feature vectors and
        # the retained feature vectors
        # Ensure self.track_vecs[active_track_ids] is not empty if active_track_ids is not empty
        if active_track_ids and self.track_vecs is not None and self.track_vecs[active_track_ids].shape[0] > 0:
             similarities = cos_similarity(feature_vecs, self.track_vecs[active_track_ids])
             similarities[np.isnan(similarities)] = 0
        else:
             # If no active tracks with valid vectors, similarities matrix is empty
             similarities = np.zeros((len(feature_vecs), 0))


        if similarities.size < 1 and (len(feature_vecs) > 0 or len(active_track_ids) > 0):
             # Handle case where one dimension is zero, Munkres might fail
             # If there are detections but no active tracks, or vice versa,
             # the cost matrix will be non-square or have a zero dimension.
             # Munkres expects a cost matrix where rows = detections, columns = tracks.
             # If detections > 0 and active_tracks == 0, Munkres should assign all detections to 'no track'.
             # If detections == 0 and active_tracks > 0, Munkres should assign 'no detection' to all tracks.
             # If both are 0, the earlier check handles it.
             # If detections > 0 and active_tracks > 0 but similarities is empty (e.g., all NaN),
             # Munkres will compute on a zero matrix, assigning detections to tracks arbitrarily.
             # Let's ensure cost_matrix has the correct shape for Munkres even if similarities is empty.
             cost_matrix = np.ones((len(feature_vecs), len(active_track_ids))) # High cost for all assignments
        else:
             # Get the min cost(distance) from cos similarity
             cost_matrix = 1 - similarities


        # Box IoU Matrix
        # To account for uncertainty in the shape of bounding boxes, halve the cost of
        # IOU and add it to the cost matrix.
        # Ensure track_boxes is not empty for active tracks
        if active_track_ids and self.track_boxes and all(self.track_boxes[tid] for tid in active_track_ids):
            track_boxes_active = np.array(self.track_boxes, dtype=object)[active_track_ids]
            track_boxes_last = np.array([box[-1] for box in track_boxes_active])
            box_iou_matrix = np.zeros((len(feature_vecs), len(active_track_ids)))
            for i, box in enumerate(boxes):
                box_iou_matrix[i, :] = get_iou2(box, track_boxes_last)
            box_iou_matrix = (1 - box_iou_matrix) * 0.5
            # Ensure cost_matrix and box_iou_matrix have compatible shapes before adding
            if cost_matrix.shape == box_iou_matrix.shape:
                 cost_matrix = cost_matrix + box_iou_matrix
            else:
                 # If shapes don't match (e.g., one dimension was zero), skip adding IoU cost
                 logger.warning(f"Cost matrix shape {cost_matrix.shape} and IoU matrix shape {box_iou_matrix.shape} mismatch. Skipping IoU cost.")
        elif len(feature_vecs) > 0 and len(active_track_ids) > 0:
             # If there are detections and active tracks, but no valid track boxes,
             # cost_matrix is already initialized with high costs. No IoU cost added.
             pass
        else:
             # If either detections or active tracks are zero, cost_matrix is already handled.
             pass


        # Solve Assignment problem
        # Munkres requires cost_matrix to have at least one dimension > 0
        if cost_matrix.shape[0] > 0 and cost_matrix.shape[1] > 0:
            assignments = self.m.compute(cost_matrix.tolist())
            track_ids_assigned = [active_track_ids[col] for row, col in assignments]
            det_ids_assigned = [row for row, col in assignments]
            confidences_assigned = [similarities[row][col] for row, col in assignments]
        else:
            assignments = []
            track_ids_assigned = []
            det_ids_assigned = []
            confidences_assigned = []


        update_detection_dict, not_found_detection_dict = {}, {}
        assigned_det_ids = set(det_ids_assigned)
        assigned_track_ids = set(track_ids_assigned)

        # Process assigned detections
        for i, det_id in enumerate(det_ids_assigned):
            track_id = track_ids_assigned[i]
            confidence = confidences_assigned[i]

            # get "detected" person's information
            detection = self.get_person_info(
                det_id, track_id, confidences_assigned, boxes, feature_vecs
            )
            # track a copy(snapshot) of  tracked person's information
            # tracks = self.tracks.copy() # No need for a copy here, work directly on self.tracks
            track = self.tracks[track_id] # Get the actual track object

            # solve occlustion problem
            # skip update process if detected box and reidentificated box are overlapped.
            boxes_ = boxes.copy() # Need a copy to safely remove one element for the check
            frame, result = self.solve_occlusion_problem(
                frame, det_id, detection, boxes_
            )

            if result:
                message = f"{frame_id} occlusion det_id:{det_id} person_id:{track.person_id}(track_id:{track_id})"
                logger.info(message)
                # Mark the assigned track as matched even if occluded, so it doesn't get lost immediately?
                # Or should occlusion prevent matching? The original code skipped the rest of the loop,
                # meaning the track's is_matched would remain False. Let's keep that behavior.
                continue # Skip evaluation and update for this detection

            # Evaluate ID assignment
            update = self.evaluate(detection)

            if update or detection.confidence > sim_thld:
                detection.update = "update"
                update_detection_dict[det_id] = detection
                track.is_matched = True # Mark the track as matched
            else:
                detection.update = "not_found"
                not_found_detection_dict[det_id] = detection
                # Do NOT mark track.is_matched = True here, as the match was poor.

        # Process detections that were NOT assigned by Munkres (shouldn't happen if cost matrix is full?)
        # Munkres assigns every row to a column if possible. If detections > tracks, some detections won't be assigned.
        # If tracks > detections, some tracks won't be assigned.
        # Let's handle detections not assigned to *active* tracks.
        # These should be considered potential new persons.
        all_det_ids = set(range(len(boxes)))
        unassigned_det_ids = all_det_ids - assigned_det_ids

        for det_id in unassigned_det_ids:
             # These detections didn't get a good match via Munkres.
             # Treat them as potential new persons.
             # Create a dummy detection object for logging/processing
             box, center, feature_vec = self.get_box_info(det_id, boxes, feature_vecs)
             detection = Person()
             detection.track_id = -1 # Indicate no assigned track
             detection.confidence = 0 # No match confidence
             detection.feature_vec = feature_vec
             detection.box = box
             detection.box_w = box[2] - box[0]
             detection.box_h = box[3] - box[1]
             detection.center = center
             detection.euc_dist = float('inf') # High distance
             detection.box_iou = 0 # Assume low IoU with any existing track if unassigned

             detection.update = "unassigned"
             # Check if this unassigned detection should become a new track
             # Similar logic to the 'not_found' case, but it's definitely a new person.
             if not self.is_out_of_track_area(detection.center):
                 message = f"frame_id:{self.frame_id} det_id:{det_id} unassigned - Registering as new person"
                 logger.debug(message)
                 # Create a *new* track for this detection
                 self.track_points.append([center])
                 self.track_points_measured.append([center])
                 # Need to append the feature vector to self.track_vecs
                 if self.track_vecs is None:
                      self.track_vecs = detection.feature_vec
                 else:
                      self.track_vecs = np.vstack((self.track_vecs, detection.feature_vec))
                 self.track_boxes.append([box])
                 self.euc_distances.append([0.0])
                 new_track_id = len(self.tracks) # Get the next available track_id
                 self.tracks.append(Track(new_track_id, center))
                 logger.info(f"frame_id:{self.frame_id} Registered new person from unassigned det_id:{det_id} to new track_id:{new_track_id}")
             else:
                 message = f"frame_id:{self.frame_id} det_id:{det_id} unassigned - Out of counter area, not registering."
                 logger.debug(message)


        # Update assigned detections that passed evaluation
        for det_id, detection in update_detection_dict.items():
            frame = self.update(frame, detection)
            self.show_log(det_id, detection)

        # Process assigned detections that failed evaluation ("not_found")
        # These were assigned to an existing track but the match was poor.
        # The not_found method now decides whether to register a *new* person
        # based on box_iou.
        for det_id, detection in not_found_detection_dict.items():
            self.not_found(det_id, detection)
            self.show_log(det_id, detection)


        # Lost tracks
        # These are active tracks that were NOT assigned a detection by Munkres
        # OR were assigned a detection but the match was poor (is_matched is False).
        lost_track_ids = [
            t.track_id for t in self.tracks if t.stats != DELETED and not t.is_matched
        ]
        for track_id in lost_track_ids:
            track = self.tracks[track_id]
            frame = self.lost(frame, track)

        # ----------- post process -------------------- #
        # After the loops, draw the other information into the frame
        frame = self.draw_counter_stats(frame)
        frame = self.draw_params(frame)
        frame = self.draw_distance_lines(frame) # Draw the new lines and labels

        # preserv feature vectors which is used in the register_person function
        # This seems to be preserving the feature vectors of the *current* frame's detections,
        # not the track vectors. The variable names are a bit confusing here.
        # Assuming the intent is to keep the *last processed* detection info for some reason, keep it.
        # If the intent was to update the track vectors, that happens in self.update.
        self.prev_feature_vecs = feature_vecs
        self.prev_track_boxes = boxes

        return frame

    def show_log(self, det_id, detection):
        frame_id = f"frame_id:{self.frame_id}"
        # Handle cases where detection.track_id might be -1 (unassigned)
        track_info = f"track_id:{detection.track_id}"
        person_info = "person_id:N/A"
        if detection.track_id != -1 and detection.track_id is not None and detection.track_id < len(self.tracks):
             track = self.tracks[detection.track_id]
             person_info = f"person_id:{track.person_id}"

        header = f"{detection.update} det_id:{det_id} to {person_info}({track_info}) sim:{detection.confidence:.3f}"
        info1 = f"euc_dist:{detection.is_valid_dist if hasattr(detection, 'is_valid_dist') else 'N/A'}({detection.euc_dist:.3f}) {detection.euc_dist_min if hasattr(detection, 'euc_dist_min') else 'N/A':.3f} < {detection.euc_dist:.3f} < {detection.euc_dist_max if hasattr(detection, 'euc_dist_max') else 'N/A':.3f}({detection.box_w:.3f}) euc_dist_mean:{detection.euc_dist_mean if hasattr(detection, 'euc_dist_mean') else 'N/A':.3f} euc_dist_std:{detection.euc_dist_std if hasattr(detection, 'euc_dist_std') else 'N/A':.3f}"
        ##info2 = f"mah_dist:{detection.is_valid_mah_dist}({detection.mah_dist:.3f})"
        info3 = f"bbox center:{detection.center} box_iou:{detection.box_iou:.3f}"
        # Avoid accessing track.__dict__ if track_id is -1 or invalid
        track_stats_info = 'N/A'
        if detection.track_id != -1 and detection.track_id is not None and detection.track_id < len(self.tracks):
             track_stats_info = self.tracks[detection.track_id].stats
        info4 = f"track_stats:{track_stats_info}"

        ##logger.info(f"{frame_id} {header} {info1} {info2} {info3} {info4}")
        logger.info(f"{frame_id} {header} {info1} {info3} {info4}")