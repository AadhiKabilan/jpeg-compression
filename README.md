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
