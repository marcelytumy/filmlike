# film like (bad film emulation)

This Python script applies various analog film-like effects to images, including grain, vignette, halation, color adjustments, saturation changes, and subtle blurring. It can process single images or entire directories and leverages GPU acceleration via CuPy if available.

## Features

*   Adds photographic grain.
*   Applies a vignette effect.
*   Simulates halation around highlights.
*   Adaptively adjusts color balance (warmth/coolness).
*   Adjusts saturation.
*   Applies a subtle Gaussian blur.
*   Supports processing individual files or directories.
*   Optional GPU acceleration using CuPy for faster processing.
*   Uses multiprocessing for parallel processing of multiple files.

## Installation

1.  **Clone the repository (or download the script):**
    ```bash
    git clone <your-repo-url>
    cd filmtest
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **(Optional) Install CuPy for GPU acceleration:**
    If you have an NVIDIA GPU and CUDA installed, you can install CuPy for significant speed improvements. Follow the official CuPy installation guide: [https://docs.cupy.dev/en/stable/install.html](https://docs.cupy.dev/en/stable/install.html). Make sure to install the version matching your CUDA toolkit (e.g., `pip install cupy-cuda11x` or `pip install cupy-cuda12x`). You might need to uncomment it in `requirements.txt` first.

## Usage

### Analog Effect (`analog_effect.py`)

Run the script from your terminal:

```bash
python analog_effect.py <input_path> <output_dir> [options]
```

**Arguments:**

*   `input_path`: Path to the input image file or a directory containing images.
*   `output_dir`: Path to the directory where processed images will be saved.

**Options:**

*   `--grain FLOAT`: Amount of grain (default: 0.05).
*   `--vignette FLOAT`: Vignette strength (0.0 to 1.0, default: 0.6).
*   `--saturation FLOAT`: Saturation factor (1.0=original, default: 0.8).
*   `--blur FLOAT`: Gaussian blur radius (default: 0.5). Set to 0 to disable.
*   `--halation_threshold FLOAT`: Luminance threshold for halation (0.0 to 1.0, default: 0.85).
*   `--halation_blur FLOAT`: Blur radius for halation glow (default: 20).
*   `--halation_strength FLOAT`: Strength of halation effect (default: 0.4).
*   `--halation_color_r INT`: Red component of halation color (0-255, default: 255).
*   `--halation_color_g INT`: Green component of halation color (0-255, default: 100).
*   `--halation_color_b INT`: Blue component of halation color (0-255, default: 50).
*   `--prefix STR`: Prefix for output filenames (default: "analog_").
*   `--workers INT`: Number of worker processes (default: number of CPUs).

**Example:**

```bash
# Process a single image
python analog_effect.py ./my_photo.jpg ./output_images --grain 0.07 --vignette 0.7

# Process all images in a directory
python analog_effect.py ./input_folder ./processed_folder --saturation 0.75 --prefix "film_"
```

### PNG to JPEG Converter (`png_to_jpeg_converter.py`)

This script converts PNG images in a specified folder to JPEG format. It handles transparency by placing the image on a white background before conversion. It uses multiple threads for faster processing of multiple files.

Run the script from your terminal:

```bash
python png_to_jpeg_converter.py <input_folder> [options]
```

**Arguments:**

*   `input_folder`: Path to the folder containing PNG images.

**Options:**

*   `-o`, `--output_folder`: Path to the folder where JPEG images will be saved (default: `jpeg_output` in the current directory).
*   `-q`, `--quality INT`: JPEG quality (1-100, default: 95).
*   `-w`, `--workers INT`: Maximum number of worker threads (default: Python's default).

**Example:**

```bash
# Convert all PNGs in 'input_pngs' folder, save to 'output_jpegs' with quality 90
python png_to_jpeg_converter.py ./input_pngs -o ./output_jpegs -q 90
```

## Notes

*   The `analog_effect.py` script attempts to detect and use a compatible GPU via CuPy if installed. Otherwise, it falls back to CPU processing using NumPy and SciPy.
*   Supported input image formats for `analog_effect.py` include common types like PNG, JPG/JPEG, TIFF, BMP. Output format is typically JPEG (quality 90) or PNG/TIFF if the original was one of those.
*   Multiprocessing (`analog_effect.py`) or multithreading (`png_to_jpeg_converter.py`) is used to speed up processing when handling multiple files. The number of workers can be adjusted with the `--workers` argument.
