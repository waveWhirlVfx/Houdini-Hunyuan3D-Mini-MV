# Hunyuan 3D Generator for Houdini

A Houdini plugin for generating 3D models from images using Hunyuan3D, Tencent's text-to-3D and image-to-3D model.

![2025-04-0516-49-40-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/9863482d-c09a-4d82-934d-adc3236f6498)

![2025-04-0516-49-40-ezgif com-video-to-gif-converter (1)](https://github.com/user-attachments/assets/943a59a8-54d7-4141-a3aa-a93e1ff3d1c4)

## Overview

This plugin integrates Hunyuan3D models directly into Houdini, allowing artists and technical directors to generate 3D models from single images or multi-view images with a user-friendly interface. The tool supports both Hunyuan's "mini" mode (single image) and "mv" mode (multi-view images).

## Features

- **Dual Generation Modes**:
  - **Mini Mode**: Generate 3D models from a single image
  - **MV Mode**: Generate 3D models from three images (front, left, back views)
  
- **Multiple Model Variants**:
  - Standard, Turbo, and Fast versions for both mini and mv modes
  
- **Customizable Parameters**:
  - Steps: Control the number of denoising steps
  - Octree Resolution: Adjust the detail level of generated meshes
  - Number of Chunks: Configure memory usage and processing
  
- **Batch Processing**:
  - Process multiple jobs sequentially
  - Queue management for efficient workflow
  
- **Additional Features**:
  - Background removal for input images
  - Multiple output formats (OBJ, GLB, STL)
  - GPU memory management
  - Detailed progress tracking
  - Performance statistics

## Prerequisites

- Houdini 19.5 or newer
- Python 3.9+
- CUDA-compatible GPU with at least 8GB VRAM (for GPU acceleration)
- Hunyuan3D Python package

## Installation

1. Install the required Python packages: in Houdini you can use Mlops for pip install.(Cuda 12.4 required)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install hy3dgen trimesh psutil, ninja, pybind11, diffusers, einops, opencv-python, numpy, transformers, omegaconf, trimesh, pymeshlab, pygltflib, xatlas,gradio, fastapi, uvicorn, rembg, onnxruntime
```
![Recording2025-04-05170145-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/0870d74c-65e5-4d90-ac82-307aa83369d7)


2. Copy paste the `Hunyuan min-mv-combined.py` code Directly to your Houdini Python panel.

## Usage

1. In Houdini, go to `Windows > Python Panel` and select the Hunyuan3D panel (or create a new panel and load the script)

2. Select your generation mode:
   - **Mini Mode**: For single image input
   - **MV Mode**: For front, left, and back view inputs

3. Add one or more jobs using the "Add Job" button

4. For each job:
   - Select your input image(s)
   - Preview the images to ensure proper selection

5. Configure common parameters:
   - Set the number of steps (higher = better quality but slower)
   - Adjust octree resolution for mesh detail
   - Set number of chunks based on your memory constraints
   - Select device (CUDA recommended for speed)
   - Choose model variant based on speed/quality tradeoff

6. Set additional parameters:
   - Output path for the generated mesh
   - Enable/disable background removal (mini mode)
   - Select output format
   - Set Houdini node name for the imported mesh

7. Click "Generate 3D Model" to start processing

8. Monitor progress in the status window and progress bar

9. When complete, the model will be loaded automatically into your Houdini scene

## Performance Tips

- Use a CUDA-compatible GPU for best performance
- The "turbo" model variants offer a good balance between speed and quality
- Use the "Clean GPU & Unload Models" button between generation sessions to free memory
- For large batches, consider using lower steps (15-20) for initial tests, then higher steps (30+) for final quality
- Higher octree resolution increases detail but requires more memory and processing time

## Troubleshooting

- **Out of Memory Errors**: Reduce octree resolution and number of chunks
- **Slow Performance**: Try a faster model variant or switch to GPU processing
- **Invalid Images**: Ensure images are in PNG, JPG, or BMP format and properly visible in the preview
- **Import Issues**: Check that the output path is valid and writable

## Credits

This plugin interfaces with Tencent's Hunyuan3D model. Please follow Tencent's licensing and usage terms when using this tool.

## License

[MIT License](LICENSE)

## Contact

For issues, suggestions, or contributions, please open an issue on this repository.
