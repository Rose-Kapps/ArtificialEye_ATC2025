import os
# Force Qt to use XCB plugin on Raspberry Pi
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import subprocess
subprocess.Popen(["/bin/bash", "-c", "~/check_hotspot.sh"])

import cv2
import numpy as np
import threading
import math
import asyncio
import time
from collections import deque
from evdev import InputDevice, ecodes, list_devices
try:
    ft = cv2.freetype.createFreeType2()
    ft.loadFontData(fontFileName='/usr/share/fonts/truetype/msttcorefonts/arial.ttf', id=0)
except AttributeError:
    print("OpenCV built without freetype support; falling back to Hershey.")
    ft = None


# ----------------------------------------
# Constants - window & buttons
# ----------------------------------------
WIDTH           = 1280              # full screen width
HEIGHT          = 720               # full screen height
CAM_AR          = 4 / 3             # camera aspect ratio

# We include reset and pause as additional buttons
BUTTON_COUNT    = 7                 # number of buttons including reset & pause/play
BTN_PANEL_W     = 320               # width when menu is open
BTN_W           = BTN_PANEL_W       # width of each button
BTN_H           = HEIGHT // BUTTON_COUNT  # height of each button

# Toggle (arrow) button dimensions
ARROW_BTN_W     = 40                # width of pull-out toggle
ARROW_BTN_H     = 80                # height of pull-out toggle


# Precompute button rectangles (when menu is open)
mode_buttons = [
    ((0, i * BTN_H), (BTN_W, (i + 1) * BTN_H)) for i in range(BUTTON_COUNT)
]

# Global state
image_mode        = "original"
zoom_factor       = 1.0
target_zoom       = 1.0
pan_offset        = [0.0, 0.0]
target_pan_offset = [0.0, 0.0]
zoom_lock         = threading.Lock()
is_touching       = False
paused            = False            # paused state flag
paused_frame      = None             # stores last frame when paused
cap = None  # Global so it can be flushed from anywhere
latest_frame = None
cap_running = True
cap_lock = threading.Lock()
need_restart_capture = False
flip_180 = False

click_timestamps = deque(maxlen=20)  # Store recent click times
EXIT_CLICK_COUNT = 10
EXIT_TIME_WINDOW = 5  # seconds


# Pull-out menu state
enable_menu       = False

# UDP input pipeline
PORT              = 5000
pipeline          = f"udp://@:{PORT}"

def frame_reader():
    global cap, latest_frame, cap_running, need_restart_capture
    while cap_running:
        if need_restart_capture:
            time.sleep(0.1)
            continue
        with cap_lock:
            if cap is None:
                continue
            ret, frame = cap.read()
            if ret:
                latest_frame = frame


# Touch helpers

def get_touch_device():
    for dev_path in list_devices():
        dev = InputDevice(dev_path)
        if 'touchscreen' in dev.name.lower() or 'ft5406' in dev.name.lower():
            return dev
    raise RuntimeError("No touchscreen detected.")


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def smooth_zoom(old_zoom, new_zoom, alpha=0.1):
    return old_zoom * (1 - alpha) + new_zoom * alpha


def smooth_pan(old_pan, new_pan, alpha=0.05):
    return [
        old_pan[0] * (1 - alpha) + new_pan[0] * alpha,
        old_pan[1] * (1 - alpha) + new_pan[1] * alpha,
    ]

# Mouse callback - handles clicks and toggling the menu or pause/play
def mouse_callback(event, x, y, flags, param):
    global image_mode, enable_menu, target_zoom, zoom_factor, paused, paused_frame, cap

    if event == cv2.EVENT_LBUTTONDOWN:
        now = time.time()
        click_timestamps.append(now)
        # Remove old timestamps
        while click_timestamps and now - click_timestamps[0] > EXIT_TIME_WINDOW:
            click_timestamps.popleft()
        if len(click_timestamps) >= EXIT_CLICK_COUNT:
            print("Exiting: 10 clicks within 5 seconds detected.")
            cv2.destroyAllWindows()
            os._exit(0)  # Force quit immediately

        panel_w = BTN_PANEL_W if enable_menu else ARROW_BTN_W
        # Coordinates of the arrow toggle
        ax1 = panel_w - ARROW_BTN_W
        ax2 = panel_w
        ay1 = (HEIGHT - ARROW_BTN_H) // 2
        ay2 = ay1 + ARROW_BTN_H
        # Toggle menu
        if ax1 <= x <= ax2 and ay1 <= y <= ay2:
            enable_menu = not enable_menu
            return
        # If menu is open, check each button
        if enable_menu:
            for i, ((x1, y1), (x2, y2)) in enumerate(mode_buttons):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    # Button actions
                    if i == 0:
                        image_mode = "original"
                    elif i == 1:
                        image_mode = "mixed"
                    elif i == 2:
                        image_mode = "invert"
                    elif i == 3:
                        image_mode = "binary"
                    elif i == 4:
                        # Reset zoom & pan
                        zoom_factor = 1.0
                        target_zoom = 1.0
                        pan_offset[:] = [0.0, 0.0]
                        target_pan_offset[:] = [0.0, 0.0]
                    elif i == 5:
                        paused = not paused
                        if paused:
                            print("Paused")
                        else:
                            print("Resumed - resetting capture")
                            paused_frame = None
                            global need_restart_capture
                            need_restart_capture = True
                            with cap_lock:
                                if cap is not None:
                                    cap.release()
                                cap = cv2.VideoCapture(pipeline)
                            need_restart_capture = False
                    elif i == 6:
                        global flip_180
                        flip_180 = not flip_180
                        print("Flip 180 toggled:", flip_180)


#
                    # Clear paused_frame on any mode change so new frame is used next unpause
                    if not paused:
                        paused_frame = None
                    print(f"Action: {['Original','Mixed','Invert','Binary','Reset','Pause', 'Flip'][i]}")
                    return

# Touch listener - pinch-to-zoom and pan
def touch_listener():
    global target_zoom, target_pan_offset, is_touching
    dev = get_touch_device()
    slot_id = 0
    active = {}
    prev_dist = None
    prev_pt = None
    PAN_THRESHOLD = 100  # ? Prevents jumpy movements

    for ev in dev.read_loop():
        if ev.type == ecodes.EV_ABS:
            c, v = ev.code, ev.value
            if c == ecodes.ABS_MT_SLOT:
                slot_id = v
            elif c == ecodes.ABS_MT_TRACKING_ID:
                if v == -1:
                    active.pop(slot_id, None)
                else:
                    active[slot_id] = (0, 0)
            elif c == ecodes.ABS_MT_POSITION_X:
                x, y = active.get(slot_id, (0, 0))
                active[slot_id] = (v, y)
            elif c == ecodes.ABS_MT_POSITION_Y:
                x, y = active.get(slot_id, (0, 0))
                active[slot_id] = (x, v)

            pts = list(active.values())
            is_touching = len(pts) > 0

            if len(pts) >= 2:
                d = distance(pts[0], pts[1])
                if prev_dist is not None:
                    with zoom_lock:
                        old = target_zoom
                        delta = d - prev_dist
                        target_zoom = max(1.0, min(old + delta / 300.0, 5.0))
                        scale = old / target_zoom
                        target_pan_offset[0] *= scale
                        target_pan_offset[1] *= scale
                prev_dist = d
                prev_pt = None

            elif len(pts) == 1:
                pt = pts[0]
                if prev_pt is not None:
                    dx, dy = pt[0] - prev_pt[0], pt[1] - prev_pt[1]
                    if abs(dx) < PAN_THRESHOLD and abs(dy) < PAN_THRESHOLD:
                        with zoom_lock:
                            target_pan_offset[0] = max(-0.5, min(0.5, target_pan_offset[0] + dx / 800))
                            target_pan_offset[1] = max(-0.5, min(0.5, target_pan_offset[1] - dy / 480))
                prev_pt = pt
                prev_dist = None

            else:
                prev_pt = prev_dist = None
                is_touching = False

# Image-processing helper
def process_image(img, mode="original"):
    if mode == "contrast":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        ycr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycr)
        return cv2.cvtColor(cv2.merge((clahe.apply(y), cr, cb)), cv2.COLOR_YCrCb2BGR)
    if mode == "sharpen":
        k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        return cv2.filter2D(img, -1, k)
    if mode == "mixed": return process_image(process_image(img, "contrast"), "sharpen")
    if mode == "invert": return cv2.bitwise_not(img)
    if mode == "binary":
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        b = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    return img

# Main loop - streaming with zoom/pan, pull-out menu and pause/play
def stream_and_zoom_video():
    global zoom_factor, pan_offset, target_pan_offset, target_zoom, image_mode, paused, paused_frame, cap, cap_running
    cap = cv2.VideoCapture(pipeline)
    if not cap.isOpened(): print("Cannot open video stream."); return
    cap_running = True
    threading.Thread(target=frame_reader, daemon=True).start()

    win = "GoPro Stream Pinch & Pull-out Menu"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(win, mouse_callback)

    labels  = ["ORIGINAL","CONTRAST","INVERT","BINARY","RESET","PAUSE", "FLIP"]
    colours = [(175,225,225),(225,175,225),(225,225,175),(175,225,175),(200,200,200),(100,100,100),(255,180,180)]


    print("Streaming - pinch to zoom, slide to pan, press 'q' to quit.")
    while True:
        # Frame acquisition or freeze
        if not paused:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()
            if flip_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            paused_frame = frame.copy()
        else:
            if paused_frame is None:
                continue
            frame = paused_frame.copy()
            if flip_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)




        h, w = frame.shape[:2]

        # Smooth zoom & pan
        with zoom_lock:
            if is_touching:
                zoom_factor = smooth_zoom(zoom_factor, target_zoom)
                pan_offset[:] = smooth_pan(pan_offset, target_pan_offset)
            else:
                target_zoom = zoom_factor
                target_pan_offset[:] = pan_offset[:]

        # Crop & resize to full-window view
        nw, nh = int(w/zoom_factor), int(h/zoom_factor)
        cx = np.clip(int(w/2 + pan_offset[1]*w), nw//2, w-nw//2)
        cy = np.clip(int(h/2 + pan_offset[0]*h), nh//2, h-nh//2)
        crop = frame[cy-nh//2:cy+nh//2, cx-nw//2:cx+nw//2]
        view = process_image(cv2.resize(crop, (w, h)), image_mode)

        # Build canvas + menu background
        panel_w = BTN_PANEL_W if enable_menu else ARROW_BTN_W
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # Draw menu if open
        if enable_menu:
            canvas[:, :BTN_PANEL_W] = (30,30,30)
            for i, ((x1,y1),(x2,y2)) in enumerate(mode_buttons):
                cv2.rectangle(canvas, (x1,y1), (x2,y2), colours[i], -1)
                label = labels[i]
                font_height = int(BTN_H * 0.4)

                if ft:
                    # Get text size from freetype
                    (text_w, text_h), _ = ft.getTextSize(label, fontHeight=font_height, thickness=2)
                else:
                    # Estimate using default font
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)

                # Calculate centred coordinates
                text_x = x1 + (BTN_W - text_w) // 2
                text_y = y1 + (BTN_H + text_h) // 2

                # Draw the text
                if ft:
                    ft.putText(canvas, label, (text_x, text_y), fontHeight=font_height, color=(0,0,0), thickness=2, line_type=cv2.LINE_AA)
                else:
                    cv2.putText(canvas, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 5, cv2.LINE_AA)
                                
        # Crop the view to exactly fill the remaining area without squeeze
        avail_w = WIDTH - panel_w
        x0 = max(0, (view.shape[1] - avail_w)//2)
        clipped = view[:, x0:x0+avail_w]
        canvas[:, panel_w:] = clipped

        # Draw toggle arrow
        ax1 = panel_w - ARROW_BTN_W; ax2 = panel_w
        ay1 = (HEIGHT-ARROW_BTN_H)//2; ay2 = ay1+ARROW_BTN_H
        cv2.rectangle(canvas, (ax1,ay1), (ax2,ay2), (100,100,100), -1)
        mid = (ay1+ay2)//2
        if enable_menu:
            pts = np.array([[ax1+10,mid],[ax2-10,mid-20],[ax2-10,mid+20]])
        else:
            pts = np.array([[ax2-10,mid],[ax1+10,mid-20],[ax1+10,mid+20]])
        cv2.fillConvexPoly(canvas, pts, (255,255,255))

        cv2.imshow(win, canvas)
        if (cv2.waitKey(5) & 0xFF) == ord('q'):
            break

    with cap_lock:
        if cap is not None:
            cap.release()

    cv2.destroyAllWindows()

# Async main
def main():
    try:
        threading.Thread(target=touch_listener, daemon=True).start()
        stream_and_zoom_video()
    finally:
        global cap_running
        cap_running = False

if __name__ == "__main__":
    main()