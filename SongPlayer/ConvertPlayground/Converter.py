import os
import math
import base64
import json
from PIL import Image

# Constants for file-size limits
MAX_JSON_BYTES = 1_000_000  # Target max JSON size ~1MB
MAX_RAW_BYTES  = int(MAX_JSON_BYTES * 0.75)  # base64 adds ~33% overhead
MAX_PIXELS     = MAX_RAW_BYTES // 4          # Each pixel = 4 bytes (RGBA)

# Get the absolute path of this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Iterate over all files in the script's directory
for filename in os.listdir(script_dir):
    # Only process .png files
    if not filename.lower().endswith('.png'):
        continue

    # Build the full path to the PNG
    png_path = os.path.join(script_dir, filename)

    # Open the image with Pillow and convert to RGBA
    img = Image.open(png_path).convert('RGBA')
    width, height = img.size
    total_pixels = width * height

    # If image is too large, scale it down to fit under 1MB JSON
    if total_pixels > MAX_PIXELS:
        scale_factor = math.sqrt(MAX_PIXELS / total_pixels)
        new_w = max(1, int(width * scale_factor))
        new_h = max(1, int(height * scale_factor))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        width, height = new_w, new_h
        print(f"Resized {filename} from {total_pixels} to {width * height} pixels to fit size limit.")

    # Get raw RGBA bytes
    pixel_bytes = bytearray(img.tobytes())  # length = width * height * 4

    # Convert fully transparent pixels (alpha=0) to black (R=G=B=0, A=255)
    for i in range(0, len(pixel_bytes), 4):
        if pixel_bytes[i + 3] == 0:  # alpha byte
            pixel_bytes[i]     = 0   # R
            pixel_bytes[i + 1] = 0   # G
            pixel_bytes[i + 2] = 0   # B
            pixel_bytes[i + 3] = 255 # A (opaque)

    # Base64-encode the final RGBA data
    encoded_pixels = base64.b64encode(bytes(pixel_bytes)).decode('ascii')

    # Create a JSON structure with image metadata and pixel data
    image_data = {
        "width": width,
        "height": height,
        "pixels": encoded_pixels
    }

    # Build the output filename (same directory as the script)
    json_filename = os.path.join(
        script_dir,
        f"{os.path.splitext(filename)[0]}_Data.json"
    )

    # Write the JSON file (minimize whitespace to keep file size small)
    with open(json_filename, 'w') as f:
        json.dump(image_data, f, separators=(',', ':'))

    print(f"Saved {os.path.basename(json_filename)} ({width}x{height} pixels).")
