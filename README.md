# jpeg-compression

This project implements basic JPEG image compression using Discrete Cosine Transform (DCT), Zigzag pattern encoding, and Huffman coding. The goal is to reduce the size of the image while retaining quality, following the steps commonly used in JPEG compression.

## Project Structure

The project includes the following components:
- **DCT (Discrete Cosine Transform)**: Converts 8x8 image blocks from spatial domain to frequency domain.
- **Zigzag Pattern**: Used to reorder the DCT coefficients for more efficient compression.
- **Quantization**: Reduces the precision of the DCT coefficients based on a quality factor.
- **Huffman Encoding**: A lossless compression technique applied to further reduce the file size.
- **YCbCr Color Space**: Converts the image from RGB to YCbCr color space for better compression efficiency.

## Features

- **Image Compression**: Compresses images by dividing them into 8x8 blocks, applying DCT, quantization, and Zigzag encoding.
- **Huffman Encoding**: Further compresses the quantized coefficients using Huffman coding.
- **Decompression**: Recovers the image by reversing the compression process with Huffman decoding, inverse Zigzag, inverse DCT, and RGB conversion.
- **Quality Control**: Allows the setting of a compression quality factor, which controls the balance between image quality and compression ratio.

## Requirements

To run the project, you need the following Python dependencies:

- Python 3.x
- `numpy`
- `PIL` (Pillow)
- `scipy`

Install the dependencies using:

```bash
pip install numpy pillow scipy


## Steps Involved

### 1. **Color Space Conversion**
   - The image is first converted from RGB to YCbCr color space, as YCbCr is more efficient for compression (Y contains brightness information, while Cb and Cr contain chrominance data).

### 2. **Blocking the Image**
   - The image is divided into 8x8 blocks, which is the standard block size used in JPEG compression.

### 3. **Applying DCT**
   - The Discrete Cosine Transform (DCT) is applied to each 8x8 block of the image. This step transforms spatial domain data into frequency domain data.

### 4. **Quantization**
   - The DCT coefficients are quantized using a predefined quantization matrix. This step reduces the precision of the data, which significantly reduces the file size.

### 5. **Compression**
   - The quantized DCT coefficients are then encoded using run-length encoding (RLE) and Huffman coding.

### 6. **Decompression**
   - For decompression, the process is reversed: the compressed data is decoded, inverse quantization is applied, and then the inverse DCT is performed on each block to reconstruct the image.

## JPEG Compression and Decompression Implementation

### `jpeg_2.py`

This Python script performs JPEG compression and decompression. Below is a breakdown of the key steps in the script:

```python
import numpy as np
from PIL import Image
import scipy.fftpack as fft
import heapq
import os

# Define the quantization matrix
Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Function to convert RGB to YCbCr
def rgb_to_ycbcr(image):
    return image.convert("YCbCr")

# Function to apply DCT
def apply_dct(block):
    return fft.dct(fft.dct(block.T, norm='ortho').T, norm='ortho')

# Function to apply inverse DCT
def apply_idct(block):
    return fft.idct(fft.idct(block.T, norm='ortho').T, norm='ortho')

# Quantization step
def quantize(block, quant_matrix):
    return np.round(block / quant_matrix)

# Inverse quantization step
def inverse_quantize(block, quant_matrix):
    return block * quant_matrix

# JPEG compression function
def jpeg_compress(image_path):
    image = Image.open(image_path)
    image = rgb_to_ycbcr(image)
    pixels = np.array(image)
    
    compressed_blocks = []
    
    for y in range(0, pixels.shape[0], 8):
        for x in range(0, pixels.shape[1], 8):
            block = pixels[y:y+8, x:x+8]
            dct_block = apply_dct(block)
            quantized_block = quantize(dct_block, Q)
            compressed_blocks.append(quantized_block)
    
    return compressed_blocks

# JPEG decompression function
def jpeg_decompress(compressed_blocks, output_size):
    decompressed_image = np.zeros(output_size)
    
    block_idx = 0
    for y in range(0, output_size[0], 8):
        for x in range(0, output_size[1], 8):
            quantized_block = compressed_blocks[block_idx]
            dct_block = inverse_quantize(quantized_block, Q)
            decompressed_block = apply_idct(dct_block)
            decompressed_image[y:y+8, x:x+8] = decompressed_block
            block_idx += 1
    
    return decompressed_image

# Save the decompressed image
def save_decompressed_image(decompressed_image, output_path):
    decompressed_image = np.uint8(decompressed_image)
    decompressed_image = Image.fromarray(decompressed_image, mode='YCbCr')
    decompressed_image = decompressed_image.convert('RGB')
    decompressed_image.save(output_path)

# Example usage
image_path = 'input_image.jpg'  # Specify the input image path
output_path = 'decompressed_image.jpg'  # Specify the output decompressed image path

compressed_blocks = jpeg_compress(image_path)
image = Image.open(image_path)
output_size = image.size
decompressed_image = jpeg_decompress(compressed_blocks, output_size)
save_decompressed_image(decompressed_image, output_path)