## How to compute metrics (on Windows)

### PSNR&SSIM

1. Install Mitsuba, and add its path to environment variable "PATH"
2. Run "tonemap.py" to transform .exr file to .png file
3. Run "metric.py" to compute PSNR&SSIM

### VMAF

1. Install **ffmpeg**, and add its path to environment variable "PATH"
2. Run generate "video.py" 
3. Follow the instructions of [VMAF](https://github.com/Netflix/vmaf) to use **ffmpeg** to compute VMAF metric between GT and ExtraNet results