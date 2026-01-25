'''
A script for resizing images, to prevent excessively large image resolutions 
from consuming too many tokens when being passed to an MLLM.
'''

from PIL import Image, ImageOps
import os

def load_and_resize_image(
    image_path: str,
    save_path: str,
    max_side: int = 1024,
    image_format: str = "JPEG",
    quality: int = 90
) -> str:
    """
    Load an image, resize it while keeping aspect ratio, and save it.

    Args:
        image_path (str): Path to the input image.
        save_path (str): Path to save the resized image.
        max_side (int): Maximum length of the longer side.
        image_format (str): Output image format (JPEG / PNG / WEBP).
        quality (int): Image quality (for lossy formats like JPEG).

    Returns:
        str: Path to the saved resized image.
    """

    # 1. 打开图片
    img = Image.open(image_path)

    # 2. 处理 EXIF 方向（非常重要，尤其是手机/截图）
    img = ImageOps.exif_transpose(img)

    # 3. 统一转成 RGB（避免 RGBA / L 模式问题）
    if img.mode != "RGB":
        img = img.convert("RGB")

    # 4. 按最长边缩放
    width, height = img.size
    max_current_side = max(width, height)

    if max_current_side > max_side:
        scale = max_side / max_current_side
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    # 5. 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 6. 保存图片
    save_kwargs = {}
    if image_format.upper() in ["JPEG", "JPG", "WEBP"]:
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True

    img.save(save_path, format=image_format, **save_kwargs)

    return save_path

if __name__ == '__main__':
    pass