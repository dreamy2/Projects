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

# Initialize MSS for screen capture (persistent)
sct = mss.mss()
monitor = sct.monitors[1]  # primary monitor

# Initialize persistent HTTP session
session = requests.Session()

def capture_screen_to_json(max_pixels):
    """
    Captures the screen, scales it down if necessary, and converts the image
    into a JSON string with keys: 'width', 'height', 'pixels'. The 'pixels'
    value is the base64-encoded RGBA data.
    """
    # Capture the screen using MSS
    sct_img = sct.grab(monitor)
    # MSS provides BGRA data; convert to RGBA by swapping channels
    img = Image.frombytes("RGBA", (sct_img.width, sct_img.height), sct_img.bgra, "raw", "BGRA")
    
    width, height = img.size
    total_pixels = width * height

    # Scale down if image is too large
    if total_pixels > max_pixels:
        scale_factor = math.sqrt(max_pixels / total_pixels)
        new_w = max(1, int(width * scale_factor))
        new_h = max(1, int(height * scale_factor))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        width, height = new_w, new_h

    # Convert image to a NumPy array for fast pixel processing
    arr = np.array(img, dtype=np.uint8)
    # Identify fully transparent pixels (alpha == 0) and set them to opaque black
    transparent = (arr[:, :, 3] == 0)
    arr[transparent, 0] = 0
    arr[transparent, 1] = 0
    arr[transparent, 2] = 0
    arr[transparent, 3] = 255

    # Get raw bytes from the array and Base64 encode them
    encoded_pixels = base64.b64encode(arr.tobytes()).decode('ascii')

    # Build the JSON object and return a compact JSON string
    image_data = {
        "width": width,
        "height": height,
        "pixels": encoded_pixels
    }
    return json.dumps(image_data, separators=(',', ':'))

def send_request(url, payload, method, headers, timeout=5):
    """
    Sends an HTTP request using the requests library with a persistent session.
    The payload is a dictionary converted to a JSON string.
    Raises an exception if the request fails.
    """
    data = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    try:
        if method.upper() == "POST":
            resp = session.post(url, data=data, headers=headers, timeout=timeout)
        elif method.upper() == "PUT":
            resp = session.put(url, data=data, headers=headers, timeout=timeout)
        else:
            raise ValueError("Unsupported HTTP method")
    except requests.exceptions.RequestException as e:
        raise Exception(e)

    if not (200 <= resp.status_code < 300):
        raise Exception(f"HTTP {method} failed with status code {resp.status_code}")

if __name__ == "__main__":
    try:
        # Ask user for ScreenKey and max JSON bytes
        screen_key = input("Enter ScreenKey: ").strip()
        if not screen_key:
            print("ScreenKey cannot be empty.")
            sys.exit(1)

        max_json_input = input("Enter max JSON bytes (e.g., 600000): ").strip()
        if not max_json_input.isdigit():
            print("Invalid input for max JSON bytes.")
            sys.exit(1)
        MAX_JSON_BYTES = int(max_json_input)
        
        # Compute derived limits
        MAX_RAW_BYTES  = int(MAX_JSON_BYTES * 0.75)  # Allow for ~33% base64 overhead
        MAX_PIXELS     = MAX_RAW_BYTES // 4          # Each pixel = 4 bytes (RGBA)
        
        print("Capturing initial screen snapshot...")
        init_json = capture_screen_to_json(MAX_PIXELS)

        # Set the required headers
        headers = {
            "IAm": "System",
            "Content-Type": "application/json"
        }

        # POST the initial screenshot
        post_url = "https://amc.czcs.xyz"
        init_payload = {
            "key": screen_key,
            "init": init_json
        }
        send_request(post_url, init_payload, "POST", headers)
        print("Initial screenshot sent successfully.")
        print("Starting continuous capture... (Press Ctrl+C to stop)")

        # Continuous capture: every 0.05 seconds, capture and send a new frame
        while True:
            frame_json = capture_screen_to_json(MAX_PIXELS)
            put_url = f"https://amc.czcs.xyz/{screen_key}"
            frame_payload = {
                "frame": frame_json
            }
            send_request(put_url, frame_payload, "PUT", headers)
            print("Sent screen data")
            # Uncomment the following line if you wish to enforce a 0.05s delay:
            # time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nScreen capture stopped by user.")
        sys.exit(0)
    except urllib.error.HTTPError as he:
        print(f"\nHTTP error: {he.code} - {he.reason}")
        sys.exit(1)
    except urllib.error.URLError as ue:
        print(f"\nURL error: {ue.reason}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
