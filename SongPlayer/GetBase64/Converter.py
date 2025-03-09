import os
import base64
import glob
from PIL import Image
import io

# Maximum dimensions for 720p (width x height)
MAX_WIDTH = 1280
MAX_HEIGHT = 720
# Maximum allowed Base64 size in bytes (approximately 1 MB)
MAX_B64_SIZE = 1_000_000

def get_resized_dimensions(orig_width, orig_height, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """Return new dimensions preserving aspect ratio and fitting within max dimensions."""
    ratio = min(max_width / orig_width, max_height / orig_height, 1)
    return int(orig_width * ratio), int(orig_height * ratio)

def image_to_base64(img: Image.Image, optimize=True, compress_level=9) -> (str, bytes):
    """
    Convert a Pillow Image to a Base64 encoded PNG.
    Saves the image with interlace=0 and then forcefully sets the interlace method byte to 0.
    Returns a tuple of the Base64 string and the raw PNG data.
    """
    buffer = io.BytesIO()
    # Save image as PNG with non-interlaced output
    img.save(buffer, format="PNG", optimize=optimize, compress_level=compress_level, interlace=0)
    png_data = buffer.getvalue()
    
    # Convert to bytearray so we can patch the interlace method.
    # According to the PNG spec, the interlace method is at byte index 28.
    png_bytes = bytearray(png_data)
    if len(png_bytes) >= 29:
        png_bytes[28] = 0  # Force non-interlaced
    patched_png_data = bytes(png_bytes)
    
    encoded = base64.b64encode(patched_png_data).decode("utf-8")
    return encoded, patched_png_data

def check_interlace(png_data: bytes) -> int:
    """
    Check the interlace method in the PNG header.
    The interlace byte is at index 28 (0-indexed).
    Returns the interlace method (0 = non-interlaced, 1 = interlaced).
    """
    if len(png_data) < 29:
        return -1
    return png_data[28]

def process_image(image_file):
    try:
        with Image.open(image_file) as img:
            # Convert image to RGBA if not already in RGB/RGBA.
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA")
            orig_width, orig_height = img.size
            new_width, new_height = get_resized_dimensions(orig_width, orig_height)

            # Resize image to fit within 720p dimensions (or smaller if already below).
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            b64_str, png_data = image_to_base64(resized_img)

            # Debug: Check interlace method from PNG header.
            interlace_method = check_interlace(png_data)
            print(f"Interlace method for {image_file}: {interlace_method}")

            # If the Base64 string exceeds 1 MB, iteratively reduce the size.
            scale_factor = 0.9  # Reduce dimensions by 10% per iteration if needed
            iteration = 0
            while len(b64_str) > MAX_B64_SIZE and new_width > 10 and new_height > 10:
                iteration += 1
                new_width = int(new_width * scale_factor)
                new_height = int(new_height * scale_factor)
                print(f"Iteration {iteration}: Resizing to {new_width}x{new_height} (Base64 size: {len(b64_str)} bytes)")
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                b64_str, png_data = image_to_base64(resized_img)
                interlace_method = check_interlace(png_data)
                print(f"Interlace method after iteration {iteration}: {interlace_method}")

            return b64_str, new_width, new_height
    except Exception as e:
        print(f"Failed to process {image_file}: {e}")
        return None, None, None

def main():
    # Get the directory where this script is located.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define image file extensions to look for.
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(script_dir, ext)))
    
    if not image_files:
        print("No image files found in the folder.")
        return

    for image_file in image_files:
        print(f"Processing {image_file}...")
        b64_data, final_width, final_height = process_image(image_file)
        if b64_data is None:
            continue
        
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        output_filename = os.path.join(script_dir, f"{base_name}_{final_width}x{final_height}_Base64_V2.txt")
        with open(output_filename, "w") as f:
            f.write(b64_data)
        print(f"Exported Base64 data to {output_filename} (size: {len(b64_data)} bytes)")

if __name__ == "__main__":
    main()
