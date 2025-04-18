import argparse
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import numpy as np
import os
import sys # Import sys for exit
from concurrent.futures import ProcessPoolExecutor, as_completed # Added for multiprocessing

# --- GPU Acceleration Setup ---
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpx_ndi
    import scipy.ndimage as ndi # Keep scipy for CPU fallback
    # Check if GPU is available
    try:
        cp.cuda.Device(0).compute_capability
        gpu_available = True
        print("CuPy found and GPU detected. Enabling GPU acceleration.")
    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"CuPy found but GPU error: {e}. Falling back to CPU.")
        gpu_available = False
        cp = np # Make cp an alias for np if GPU fails
        cpx_ndi = ndi # Make cpx_ndi an alias for ndi
except ImportError:
    print("CuPy or SciPy not found. Falling back to CPU. Install CuPy (for GPU) and SciPy with pip.")
    gpu_available = False
    cp = np # Make cp an alias for np if not installed
    import scipy.ndimage as ndi # Need scipy anyway for CPU
    cpx_ndi = ndi # Make cpx_ndi an alias for ndi
# -----------------------------

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm library not found. Progress bar disabled. Install with: pip install tqdm")
    # Define a dummy tqdm function if the library is not installed
    def tqdm(iterable, *args, **kwargs):
        # If wrapping an iterable, return it. If just used for total/desc, return a dummy object.
        if iterable is not None:
            return iterable
        else:
            # Need a dummy object that supports update and set_description
            class DummyTqdm:
                def __init__(self, *args, **kwargs): pass
                def update(self, n=1): pass
                def set_description(self, desc): pass
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return DummyTqdm(*args, **kwargs)

def add_grain(img_array, xp, amount=0.05):
    """Adds photographic grain to an image array (NumPy or CuPy)."""
    # Assumes input img_array is float32 in [0, 1]
    noise = xp.random.normal(scale=amount, size=img_array.shape).astype(xp.float32)
    grained_array = xp.clip(img_array + noise, 0.0, 1.0)
    return grained_array

def apply_vignette(img_array, xp, xndi, strength=0.8):
    """Applies a vignette effect to the image array (NumPy or CuPy)."""
    # Assumes input img_array is float32 in [0, 1]
    height, width = img_array.shape[:2]

    # Create coordinate grids
    x = xp.arange(width)
    y = xp.arange(height)
    xx, yy = xp.meshgrid(x, y)

    # Calculate distance from center
    center_x = width / 2
    center_y = height / 2
    dist = xp.sqrt((xx - center_x)**2 + (yy - center_y)**2)

    # Normalize distance
    max_dist = xp.sqrt((width/2)**2 + (height/2)**2)
    norm_dist = dist / max_dist

    # Calculate vignette mask values
    vignette_values = 1.0 - strength * norm_dist
    vignette_values = xp.clip(vignette_values, 0.0, 1.0).astype(xp.float32)

    # Apply blur to the mask for smoother transition
    blur_radius_px = min(width, height) / 10 # Blur radius in pixels
    # gaussian_filter requires sigma, not radius. Approximate sigma.
    # A common approximation is sigma = radius / 3 or radius / 2. Let's use radius / 3.
    sigma = blur_radius_px / 3.0
    mask_blurred = xndi.gaussian_filter(vignette_values, sigma=sigma)

    # Blend the mask with the image
    mask_3channel = xp.stack([mask_blurred]*3, axis=-1)
    vignetted_array = xp.clip(img_array * mask_3channel, 0.0, 1.0)

    return vignetted_array

def apply_halation(img_array, xp, xndi, threshold=0.85, blur_radius=20, strength=0.4, color=(255, 100, 50)):
    """Applies a halation effect to an image array (NumPy or CuPy)."""
    # Assumes input img_array is float32 in [0, 1]

    # Calculate luminance
    luminance = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]

    # Create highlight mask
    highlight_mask_array = (luminance >= threshold).astype(xp.float32)

    # Blur the MASK
    # gaussian_filter requires sigma. Approximate sigma = radius / 3.
    sigma = blur_radius / 3.0
    blurred_mask_array = xndi.gaussian_filter(highlight_mask_array, sigma=sigma)

    # Create the colored halation layer
    halation_color_normalized = xp.array(color, dtype=xp.float32) / 255.0
    blurred_mask_3channel = xp.stack([blurred_mask_array]*3, axis=-1)
    halation_layer_array = blurred_mask_3channel * halation_color_normalized

    # Blend (Additive)
    result_array = img_array + halation_layer_array * strength
    result_array = xp.clip(result_array, 0.0, 1.0)

    return result_array

def estimate_color_balance(image, thumb_size=100):
    # ... (keep this as is, uses Pillow and NumPy, runs on CPU quickly)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    thumbnail = image.copy()
    thumbnail.thumbnail((thumb_size, thumb_size))
    img_array_np = np.array(thumbnail, dtype=np.float32) # Use np specifically here
    avg_color = np.mean(img_array_np, axis=(0, 1))
    return avg_color

def adjust_colors_adaptive(img_array, xp, avg_color, neutral_threshold=10):
    """Adaptively applies warmth or coolness to an image array (NumPy or CuPy)."""
    # Assumes input img_array is float32 in [0, 255]
    # Saturation is handled separately by Pillow on CPU before this function

    avg_r, avg_g, avg_b = avg_color

    # Define base adjustment factors (as NumPy arrays initially, convert if needed)
    cooling_factor_np = np.array([0.95, 0.95, 1.15], dtype=np.float32)
    warming_factor_np = np.array([1.1, 1.0, 0.9], dtype=np.float32)

    # Determine adjustment based on average color
    color_diff = avg_r - avg_b
    if color_diff > neutral_threshold:
        adjustment_factor_np = cooling_factor_np
    elif color_diff < -neutral_threshold:
        adjustment_factor_np = warming_factor_np
    else:
        adjustment_factor_np = cooling_factor_np

    # Convert factor to CuPy if using GPU
    adjustment_factor = xp.asarray(adjustment_factor_np)

    # Apply the chosen adjustment factor
    # Input array is [0, 255], factor is ~1.0
    img_array_adjusted = img_array * adjustment_factor
    img_array_adjusted = xp.clip(img_array_adjusted, 0, 255)

    return img_array_adjusted

def process_single_image(input_path, output_path, grain_amount=0.05, vignette_strength=0.6, saturation_factor=0.8, blur_radius=0.5, halation_threshold=0.85, halation_blur=20, halation_strength=0.4, halation_color=(255, 100, 50)):
    """Applies a series of effects to a single image file, using GPU if available."""
    try:
        img = Image.open(input_path)
    except FileNotFoundError:
        return f"Error: Input file not found at {input_path}"
    except Exception as e:
        if "cannot identify image file" in str(e):
             return f"Skipping non-image file or corrupted image: {input_path}"
        else:
            return f"Error opening image {input_path}: {e}"

    original_mode = img.mode
    original_format = img.format

    # Ensure image is RGB for processing
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # --- CPU Steps First ---
    # 1. Estimate color balance (uses NumPy)
    avg_color = estimate_color_balance(img)

    # 2. Adjust Saturation (uses Pillow Enhance - CPU)
    if abs(saturation_factor - 1.0) > 1e-6: # Apply only if factor is not 1.0
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_factor)
    # -----------------------

    # --- Prepare for Array Processing (CPU or GPU) ---
    # Convert image to NumPy array (float32, 0-255 range for color adjust)
    img_array_np = np.array(img, dtype=np.float32)

    # Select backend (NumPy/SciPy or CuPy)
    if gpu_available:
        xp = cp
        xndi = cpx_ndi
        # Transfer to GPU
        img_array = xp.asarray(img_array_np)
    else:
        xp = np
        xndi = ndi
        img_array = img_array_np # Already a NumPy array
    # -------------------------------------------------

    # --- Array Processing Steps (CPU or GPU) ---

    # 1. Adaptive Color Balance (Input: 0-255, Output: 0-255)
    img_array = adjust_colors_adaptive(img_array, xp, avg_color)

    # Convert to float [0, 1] for effects
    img_array = img_array / 255.0

    # 2. Blur (Input: 0-1, Output: 0-1)
    if blur_radius > 0:
        # gaussian_filter needs sigma. Approx sigma = radius / 3
        sigma = blur_radius / 3.0
        img_array = xndi.gaussian_filter(img_array, sigma=(sigma, sigma, 0)) # Blur spatial dims only

    # 3. Halation (Input: 0-1, Output: 0-1)
    if halation_strength > 1e-6:
        img_array = apply_halation(img_array, xp, xndi, threshold=halation_threshold, blur_radius=halation_blur, strength=halation_strength, color=halation_color)

    # 4. Grain (Input: 0-1, Output: 0-1)
    if grain_amount > 1e-6:
        img_array = add_grain(img_array, xp, grain_amount)

    # 5. Vignette (Input: 0-1, Output: 0-1)
    if vignette_strength > 1e-6:
        img_array = apply_vignette(img_array, xp, xndi, vignette_strength)

    # -------------------------------------------

    # --- Final Conversion and Save ---
    # Convert back to uint8 [0, 255]
    img_array = xp.clip(img_array * 255, 0, 255)

    # If on GPU, transfer back to CPU
    if gpu_available:
        final_array_np = cp.asnumpy(img_array).astype(np.uint8)
    else:
        final_array_np = img_array.astype(np.uint8)

    # Convert NumPy array back to Pillow Image
    final_img = Image.fromarray(final_array_np, 'RGB')

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            return f"Error creating output directory {output_dir}: {e}"

    # Save the image
    try:
        save_format = original_format if original_format in ['JPEG', 'PNG', 'TIFF'] else 'JPEG'
        if save_format == 'JPEG':
            # JPEG doesn't support alpha, ensure RGB
            if final_img.mode == 'RGBA' or final_img.mode == 'P': # Should be RGB already, but double check
                 final_img = final_img.convert('RGB')
            final_img.save(output_path, quality=90, format=save_format)
        else:
            # Handle potential mode issues for other formats if necessary
            # Example: PNG supports RGBA, so convert back if original was RGBA?
            # For simplicity, saving as RGB for now.
            final_img.save(output_path, format=save_format)
        return True # Indicate success
    except Exception as e:
        return f"Error saving image {output_path}: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply an analog film effect to an image or a directory of images.")
    # Modified arguments for input path and output directory
    parser.add_argument("input_path", help="Path to the input image file or directory.")
    parser.add_argument("output_dir", help="Path to the output directory.")
    # --- Keep existing effect arguments ---
    parser.add_argument("--grain", type=float, default=0.05, help="Amount of grain to add (e.g., 0.05).")
    parser.add_argument("--vignette", type=float, default=0.6, help="Strength of the vignette effect (0.0 to 1.0, e.g., 0.6).")
    parser.add_argument("--saturation", type=float, default=0.8, help="Saturation factor (1.0 is original, <1 desaturates, >1 saturates, e.g., 0.8).")
    parser.add_argument("--blur", type=float, default=0.5, help="Gaussian blur radius (e.g., 0.5). Set to 0 to disable.")
    parser.add_argument("--halation_threshold", type=float, default=0.85, help="Luminance threshold for halation highlights (0.0 to 1.0, e.g., 0.85).")
    parser.add_argument("--halation_blur", type=float, default=20, help="Blur radius for the halation glow (e.g., 20).")
    parser.add_argument("--halation_strength", type=float, default=0.4, help="Strength of the halation effect (e.g., 0.4).")
    parser.add_argument("--halation_color_r", type=int, default=255, help="Red component of halation color (0-255).")
    parser.add_argument("--halation_color_g", type=int, default=100, help="Green component of halation color (0-255).")
    parser.add_argument("--halation_color_b", type=int, default=50, help="Blue component of halation color (0-255).")
    parser.add_argument("--prefix", type=str, default="analog_", help="Prefix to add to output filenames.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes to use (default: number of CPUs).")


    args = parser.parse_args()

    halation_clr = (args.halation_color_r, args.halation_color_g, args.halation_color_b)
    input_path = args.input_path
    output_dir = args.output_dir
    max_workers = args.workers # Get max_workers from args

    # --- Validate paths and create output directory ---
    if not os.path.exists(input_path):
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: Cannot create output directory: {output_dir} - {e}")
        sys.exit(1)

    # --- Identify images to process ---
    image_files_to_process = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif') # Add more if needed

    if os.path.isdir(input_path):
        print(f"Processing directory: {input_path}")
        # Use scandir for potentially better performance on large directories
        try:
            for entry in os.scandir(input_path):
                if entry.is_file() and entry.name.lower().endswith(valid_extensions):
                    image_files_to_process.append(entry.path)
        except OSError as e:
            print(f"Error reading directory {input_path}: {e}")
            sys.exit(1)

        if not image_files_to_process:
             print(f"No image files found in directory: {input_path}")
             sys.exit(0)

    elif os.path.isfile(input_path):
        if input_path.lower().endswith(valid_extensions):
            print(f"Processing single file: {input_path}")
            image_files_to_process.append(input_path)
        else:
            print(f"Error: Input file is not a supported image type: {input_path}")
            sys.exit(1)
    else:
         print(f"Error: Input path is not a valid file or directory: {input_path}")
         sys.exit(1)


    # --- Process identified images using multiprocessing ---
    # Note: Using multiprocessing with GPU requires careful memory management.
    # Each worker process will try to use the GPU. Limit workers if memory issues arise.
    num_workers = max_workers if max_workers is not None else os.cpu_count()
    if gpu_available:
        # Often recommended to use fewer workers than CPU cores when using GPU per worker
        # to avoid OOM errors. Let's cap it, e.g., at 4 or based on GPU memory.
        # This is a simple heuristic; optimal number depends on GPU and image size.
        gpu_info = cp.cuda.Device(0).mem_info
        total_mem_gb = gpu_info[1] / (1024**3)
        # Rough estimate: Allow ~1-2GB per worker? Very rough.
        suggested_gpu_workers = max(1, int(total_mem_gb / 2)) # Example heuristic
        if num_workers > suggested_gpu_workers:
             print(f"Warning: Reducing workers from {num_workers} to {suggested_gpu_workers} for GPU processing to potentially avoid memory issues.")
             num_workers = suggested_gpu_workers

    print(f"Found {len(image_files_to_process)} image(s) to process using up to {num_workers} workers ({'GPU' if gpu_available else 'CPU'}).")
    success_count = 0
    fail_count = 0
    error_messages = [] # Collect error/skip messages

    # Outer progress bar for files
    with tqdm(total=len(image_files_to_process), desc="Overall Progress", unit="file") as file_pbar:
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            # Submit all tasks
            for input_file_path in image_files_to_process:
                base_filename = os.path.basename(input_file_path)
                output_filename = f"{args.prefix}{base_filename}"
                output_file_path = os.path.join(output_dir, output_filename)

                # Submit the task and store the future mapped to its input path
                future = executor.submit(
                    process_single_image,
                    input_file_path,
                    output_file_path,
                    grain_amount=args.grain,
                    vignette_strength=args.vignette,
                    saturation_factor=args.saturation,
                    blur_radius=args.blur,
                    halation_threshold=args.halation_threshold,
                    halation_blur=args.halation_blur,
                    halation_strength=args.halation_strength,
                    halation_color=halation_clr
                )
                futures[future] = input_file_path # Map future to input path

            # Process completed tasks as they finish
            for future in as_completed(futures):
                input_file_path = futures[future]
                base_filename = os.path.basename(input_file_path)
                try:
                    result = future.result() # Get the return value (True or error string)
                    if result is True:
                        success_count += 1
                    else:
                        # If result is not True, it's an error/skip message string
                        fail_count += 1
                        error_messages.append(f"{base_filename}: {result}")
                except Exception as e:
                    # Catch exceptions raised *during* future.result() or if the process crashed
                    fail_count += 1
                    error_messages.append(f"{base_filename}: Unexpected error during processing - {e}")

                # Update the overall progress bar for each completed task
                file_pbar.update(1)

    print("\n--- Processing Summary ---")
    print(f"Successfully processed: {success_count} image(s)")
    print(f"Failed/Skipped:       {fail_count} image(s)")
    if error_messages:
        print("Failures/Skips:")
        for msg in error_messages:
            print(f"  - {msg}")
    print(f"Output directory:     {output_dir}")
    print("--------------------------")