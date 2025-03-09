import os
import base64
import glob
from PIL import Image
import io

# Maximum dimensions for 720p (width x height)
MAX_WIDTH = 1280
MAX_HEIGHT = 720
# Maximum allowed Base64 size in bytes (set to 1,000,000 bytes ~ 1 MB)
MAX_B64_SIZE = 1_000_000

def get_resized_dimensions(orig_width, orig_height, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """Return new dimensions preserving aspect ratio and fitting within max dimensions."""
    ratio = min(max_width / orig_width, max_height / orig_height, 1)
    return int(orig_width * ratio), int(orig_height * ratio)

def image_to_base64(img: Image.Image, optimize=True, compress_level=9) -> str:
    """Convert a Pillow Image to a Base64 encoded PNG."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=optimize, compress_level=compress_level)
    png_data = buffer.getvalue()
    encoded = base64.b64encode(png_data).decode("utf-8")
    return encoded

def process_image(image_file):
    try:
        with Image.open(image_file) as img:
            # Ensure image is in RGB/RGBA format for PNG consistency.
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA")
            orig_width, orig_height = img.size
            new_width, new_height = get_resized_dimensions(orig_width, orig_height)

            # Start with the 720p (or smaller) dimensions
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            b64_str = image_to_base64(resized_img)

            # If the Base64 string is too big, iteratively scale down further until under 1MB.
            scale_factor = 0.9  # reduce dimensions by 10% each iteration if needed
            iteration = 0
            while len(b64_str) > MAX_B64_SIZE and (new_width > 10 and new_height > 10):
                iteration += 1
                new_width = int(new_width * scale_factor)
                new_height = int(new_height * scale_factor)
                print(f"Iteration {iteration}: Resizing to {new_width}x{new_height} (Base64 size: {len(b64_str)} bytes)")
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                b64_str = image_to_base64(resized_img)

            return b64_str, new_width, new_height
    except Exception as e:
        print(f"Failed to process {image_file}: {e}")
        return None, None, None

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define image file extensions to look for
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
        output_filename = os.path.join(script_dir, f"{base_name}_{final_width}x{final_height}_Base64.txt")
        with open(output_filename, "w") as f:
            f.write(b64_data)
        print(f"Exported Base64 data to {output_filename} (size: {len(b64_data)} bytes)")

if __name__ == "__main__":
    main()
