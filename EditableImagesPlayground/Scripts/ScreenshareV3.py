import math
import base64
import json
import sys
import time
import urllib.error
import numpy as np
from PIL import Image
import mss
import requests
import socket
import threading
import queue

# Use a persistent requests session
session = requests.Session()

def capture_screen_to_json(img, pc_name, roblox_username):
    """
    Convert a PIL Image (RGBA) to a JSON string with keys:
      'name', 'roblox_username', 'width', 'height', 'pixels', 'time'.
    'pixels' is the base64-encoded RGBA data.
    'time' is the Unix timestamp with at least three decimal places.
    """
    width, height = img.size

    # Convert image to NumPy array for quick pixel manipulation
    arr = np.array(img, dtype=np.uint8)
    # Identify fully transparent pixels (alpha == 0) and set them to opaque black
    transparent = (arr[:, :, 3] == 0)
    arr[transparent, :3] = 0
    arr[transparent, 3] = 255

    # Base64-encode the raw bytes
    encoded_pixels = base64.b64encode(arr.tobytes()).decode('ascii')

    # Current time with at least three decimal places
    time_str = f"{time.time():.3f}"

    image_data = {
        "name": pc_name,
        "roblox_username": roblox_username,
        "width": width,
        "height": height,
        "pixels": encoded_pixels,
        "time": time_str
    }
    return json.dumps(image_data, separators=(',', ':'))

def send_request(url, payload, method, headers, timeout=5):
    """
    Sends an HTTP request (POST or PUT) using a persistent session.
    Raises an exception if the request fails.
    """
    data = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    method = method.upper()
    if method == "POST":
        resp = session.post(url, data=data, headers=headers, timeout=timeout)
    elif method == "PUT":
        resp = session.put(url, data=data, headers=headers, timeout=timeout)
    else:
        raise ValueError("Unsupported HTTP method")
    resp.raise_for_status()

def capture_frames(frame_queue, max_pixels, monitor_num, delay, stop_event, pc_name, roblox_username):
    """
    Continuously captures frames from the specified monitor
    and enqueues them for sending. Uses a local MSS instance
    to avoid thread-safety issues on Windows.
    """
    with mss.mss() as sct:
        # Validate the monitor index
        if monitor_num < 1 or monitor_num > len(sct.monitors):
            print(f"Invalid monitor number. Please choose a number between 1 and {len(sct.monitors)-1}.")
            stop_event.set()
            return
        
        monitor = sct.monitors[monitor_num]

        while not stop_event.is_set():
            # Grab a raw screenshot
            sct_img = sct.grab(monitor)

            # Convert BGRA to RGBA
            img = Image.frombytes(
                "RGBA",
                (sct_img.width, sct_img.height),
                sct_img.bgra,
                "raw",
                "BGRA"
            )

            # Scale down if necessary
            total_pixels = img.width * img.height
            if total_pixels > max_pixels:
                scale_factor = math.sqrt(max_pixels / total_pixels)
                new_w = max(1, int(img.width * scale_factor))
                new_h = max(1, int(img.height * scale_factor))
                img = img.resize((new_w, new_h), Image.LANCZOS)

            # Convert the image to JSON (including time)
            frame_json = capture_screen_to_json(img, pc_name, roblox_username)
            frame_payload = {"frame": frame_json}

            # Enqueue the payload
            try:
                frame_queue.put(frame_payload, timeout=1)
            except queue.Full:
                # If the queue is full, skip this frame
                pass

            # Wait briefly before capturing the next frame
            time.sleep(delay)

def send_frames(frame_queue, screen_key, headers, stop_event):
    """
    Dequeues frames and sends them via PUT requests sequentially.
    A new frame is only sent after the previous one succeeds.
    """
    put_url = f"https://amc.czcs.xyz/{screen_key}"

    while not stop_event.is_set():
        try:
            # Wait for a frame from the queue
            frame_payload = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            send_request(put_url, frame_payload, "PUT", headers)
            print("Sent screen data")
        except Exception as e:
            print(f"Error sending frame: {e}")
            stop_event.set()  # Stop processing if a PUT fails
        finally:
            frame_queue.task_done()

def main():
    pc_name = socket.gethostname()

    print("Please input the following information correctly...")

    # Get user inputs
    screen_key = input("Enter ScreenKey: ").strip()
    if not screen_key:
        print("ScreenKey cannot be empty.")
        sys.exit(1)

    max_json_input = input("Enter max JSON bytes (ex: 300000): ").strip()
    if not max_json_input.isdigit():
        print("Invalid input for max JSON bytes.")
        sys.exit(1)
    MAX_JSON_BYTES = int(max_json_input)
    
    # Derived limits: account for ~33% Base64 overhead
    MAX_RAW_BYTES = int(MAX_JSON_BYTES * 0.75)
    MAX_PIXELS = MAX_RAW_BYTES // 4  # Each pixel = 4 bytes (RGBA)

    monitor_input = input("Enter monitor number (ex: 1): ").strip()
    monitor_num = int(monitor_input) if monitor_input.isdigit() else 1

    # Ask the user for their Roblox username
    roblox_username = input("Enter your Roblox username: ").strip()
    if not roblox_username:
        print("Roblox username cannot be empty.")
        sys.exit(1)

    # Prepare headers
    headers = {
        "IHate": "Zen",
        "Content-Type": "application/json"
    }

    # Capture and send initial screenshot via POST
    print("Capturing initial screen snapshot...")
    with mss.mss() as sct_init:
        if monitor_num < 1 or monitor_num > len(sct_init.monitors):
            print(f"Invalid monitor number. Please choose a number between 1 and {len(sct_init.monitors)-1}.")
            sys.exit(1)

        monitor = sct_init.monitors[monitor_num]
        sct_img = sct_init.grab(monitor)
        img_init = Image.frombytes(
            "RGBA",
            (sct_img.width, sct_img.height),
            sct_img.bgra,
            "raw",
            "BGRA"
        )

        total_pixels = img_init.width * img_init.height
        if total_pixels > MAX_PIXELS:
            scale_factor = math.sqrt(MAX_PIXELS / total_pixels)
            new_w = max(1, int(img_init.width * scale_factor))
            new_h = max(1, int(img_init.height * scale_factor))
            img_init = img_init.resize((new_w, new_h), Image.LANCZOS)

        # Convert the initial image to JSON (including time and Roblox username)
        init_json = capture_screen_to_json(img_init, pc_name, roblox_username)

    # POST the initial screenshot
    post_url = "https://amc.czcs.xyz"
    init_payload = {"key": screen_key, "init": init_json}
    send_request(post_url, init_payload, "POST", headers)
    print("Initial screenshot sent successfully.")

    # Prepare the queue and threads for continuous capture and sending
    frame_queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=capture_frames,
        args=(frame_queue, MAX_PIXELS, monitor_num, 0.05, stop_event, pc_name, roblox_username),
        daemon=True
    )
    send_thread = threading.Thread(
        target=send_frames,
        args=(frame_queue, screen_key, headers, stop_event),
        daemon=True
    )

    # Start threads
    capture_thread.start()
    send_thread.start()

    print("Starting continuous capture... (Press Ctrl+C to stop)")
    try:
        # Keep the main thread alive until interrupted or an error occurs
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping capture and sending...")
        stop_event.set()

    # Wait for threads to finish
    capture_thread.join()
    send_thread.join()
    sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except (urllib.error.HTTPError, urllib.error.URLError) as err:
        print(f"\nHTTP/URL error: {err}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
