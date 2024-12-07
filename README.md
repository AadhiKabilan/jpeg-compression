"# jpeg-compression" 
This project implements a simple JPEG compression and decompression algorithm in Python. The process involves converting an image into the YCbCr color space, applying Discrete Cosine Transform (DCT), quantization, Run-Length Encoding (RLE), and compression.

## Features
- Convert RGB images to YCbCr color space
- Apply DCT for compression
- Quantize the DCT coefficients
- Perform Run-Length Encoding (RLE) for efficient storage
- Decompress the image back to the original using inverse operations

## Requirements
- Python 3.x
- NumPy
- Pillow (PIL)
- SciPy

## Installation

To get started, clone this repository to your local machine:

```bash
git clone https://github.com/AadhiKabilan/jpeg-compression.git
