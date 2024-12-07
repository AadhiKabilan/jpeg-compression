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
   ```
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

```bash
   python "jpeg 2.py"
```
This Python script performs JPEG compression and decompression. Below is a breakdown of the key steps in the script:
