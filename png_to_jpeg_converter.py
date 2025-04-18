import os
import argparse
from PIL import Image
import concurrent.futures
import sys

def convert_png_to_jpeg(png_file_path, output_folder, quality=95):
    """Converts a single PNG file to JPEG format."""
    try:
        img = Image.open(png_file_path)
        # Ensure image is in RGB mode (JPEG doesn't support alpha)
        if img.mode == 'RGBA' or img.mode == 'LA' or (img.mode == 'P' and 'transparency' in img.info):
             # Create a new white background image
            bg = Image.new('RGB', img.size, (255, 255, 255))
            # Paste the image onto the background using the alpha channel as mask
            bg.paste(img, (0, 0), img.split()[-1] if img.mode == 'RGBA' or img.mode == 'LA' else None)
            img = bg
        elif img.mode != 'RGB':
             img = img.convert('RGB')


        # Construct output filename
        base_filename = os.path.basename(png_file_path)
        name_without_ext = os.path.splitext(base_filename)[0]
        jpeg_filename = f"{name_without_ext}.jpg"
        output_path = os.path.join(output_folder, jpeg_filename)

        # Save as JPEG
        img.save(output_path, 'JPEG', quality=quality)
        print(f"Converted '{png_file_path}' to '{output_path}'")
        return True
    except Exception as e:
        print(f"Error converting '{png_file_path}': {e}", file=sys.stderr)
        return False

def process_folder(input_folder, output_folder, quality=95, max_workers=None):
    """Processes all PNG files in the input folder using multiple threads."""
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' not found.", file=sys.stderr)
        return

    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"Created output folder: '{output_folder}'")
        except OSError as e:
            print(f"Error creating output folder '{output_folder}': {e}", file=sys.stderr)
            return

    png_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith('.png')
    ]

    if not png_files:
        print(f"No PNG files found in '{input_folder}'.")
        return

    print(f"Found {len(png_files)} PNG files to convert.")

    successful_conversions = 0
    failed_conversions = 0

    # Use ThreadPoolExecutor for concurrent processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map the conversion function to the list of PNG files
        future_to_file = {executor.submit(convert_png_to_jpeg, png_file, output_folder, quality): png_file for png_file in png_files}
        for future in concurrent.futures.as_completed(future_to_file):
            if future.result():
                successful_conversions += 1
            else:
                failed_conversions += 1

    print("\nConversion Summary:")
    print(f"  Successfully converted: {successful_conversions}")
    print(f"  Failed conversions:     {failed_conversions}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PNG images in a folder to JPEG format using multiple threads.")
    parser.add_argument("input_folder", help="Path to the folder containing PNG images.")
    parser.add_argument("-o", "--output_folder", default="jpeg_output", help="Path to the folder where JPEG images will be saved (default: 'jpeg_output' in the current directory).")
    parser.add_argument("-q", "--quality", type=int, default=95, help="JPEG quality (1-100, default: 95).")
    parser.add_argument("-w", "--workers", type=int, default=None, help="Maximum number of worker threads (default: None, uses Python's default).")

    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder, args.quality, args.workers)