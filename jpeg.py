import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# Standard JPEG luminance quantization matrix
Q_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def rgb_to_ycbcr(image):
    """Convert an RGB image to YCbCr."""
    rgb = np.array(image)
    ycbcr = np.dot(rgb[..., :3], [[0.299, 0.587, 0.114],
                                  [-0.168736, -0.331264, 0.5],
                                  [0.5, -0.418688, -0.081312]]) + [0, 128, 128]
    return ycbcr

def chroma_subsample(ycbcr_image):
    """Apply 4:2:0 chroma subsampling."""
    y = ycbcr_image[:, :, 0]
    cb = ycbcr_image[::2, ::2, 1]
    cr = ycbcr_image[::2, ::2, 2]
    return y, cb, cr

def zigzag_order(block):
    """Rearrange a block in a zigzag order."""
    zigzag_indices = [
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63]
    ]
    return block.flatten()[[i for sublist in zigzag_indices for i in sublist]]

def dct_2d(block):
    """Perform 2D Discrete Cosine Transform on an 8x8 block using scipy."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct_2d(block):
    """Perform 2D Inverse Discrete Cosine Transform on an 8x8 block."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def quantize(block, q_matrix):
    """Quantize the DCT coefficients using the quantization matrix."""
    return np.round(block / q_matrix)

def dequantize(block, q_matrix):
    """Dequantize the DCT coefficients using the quantization matrix."""
    return block * q_matrix

def process_blocks(image, block_size=8, func=None, *args):
    """Apply a function to each block of the image."""
    h, w = image.shape
    processed = np.zeros_like(image, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                processed[i:i+block_size, j:j+block_size] = func(block, *args)
    return processed

def jpeg_compress(image, q_matrix):
    """Compress the image using JPEG standards."""
    compressed = process_blocks(image, block_size=8, func=lambda b: quantize(dct_2d(b), q_matrix))
    return compressed

def jpeg_decompress(compressed, q_matrix):
    """Decompress the image using JPEG standards."""
    decompressed = process_blocks(compressed, block_size=8, func=lambda b: idct_2d(dequantize(b, q_matrix)))
    return np.clip(decompressed, 0, 255)  # Clip values to valid range

def main(input_bmp, output_bmp, output_jpeg):
    # Load the BMP image
    image = Image.open(input_bmp).convert('RGB')
    image_array = np.array(image, dtype=np.float32)
    
    # Step 1: RGB to YCbCr Conversion
    ycbcr_image = rgb_to_ycbcr(image)
    Image.fromarray(ycbcr_image.astype(np.uint8)).save("ycbcr_image.jpg")
    
    # Step 2: Apply Chroma Subsampling (4:2:0)
    y, cb, cr = chroma_subsample(ycbcr_image)
    Image.fromarray(y.astype(np.uint8)).save("y_channel.jpg")
    Image.fromarray(cb.astype(np.uint8)).save("cb_channel.jpg")
    Image.fromarray(cr.astype(np.uint8)).save("cr_channel.jpg")

    # Step 3: Process blocks for DCT and Quantization
    compressed = jpeg_compress(y, Q_MATRIX)
    
    # Save intermediate compressed result (after DCT and quantization)
    np.save("compressed_blocks.npy", compressed)
    
    # Step 4: Decompression (Inverse operations)
    decompressed = jpeg_decompress(compressed, Q_MATRIX)
    
    # Save decompressed image
    decompressed_cropped = decompressed[:image_array.shape[0], :image_array.shape[1]]
    Image.fromarray(decompressed_cropped.astype(np.uint8)).save(output_bmp)
    Image.fromarray(decompressed_cropped.astype(np.uint8)).save(output_jpeg, format="JPEG")
    print(f"Recovered BMP saved as: {output_bmp}")
    print(f"Compressed JPEG saved as: {output_jpeg}")

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("BMP Files", "*.bmp")])
    return file_path

def run_compression():
    input_bmp_file = select_file()
    if not input_bmp_file:
        return
    output_bmp_file = "recovered_image.bmp"
    output_jpeg_file = "compressed_image.jpg"
    
    main(input_bmp_file, output_bmp_file, output_jpeg_file)
    messagebox.showinfo("Success", f"Images saved as {output_bmp_file} and {output_jpeg_file}")

# Create the Tkinter GUI
root = tk.Tk()
root.title("JPEG Compression Tool")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack(padx=10, pady=10)

select_button = tk.Button(frame, text="Select BMP File", command=run_compression)
select_button.pack(side=tk.LEFT)

root.mainloop()
