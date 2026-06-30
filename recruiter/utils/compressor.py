import os
import io
import fitz
from PIL import Image
from docx import Document


def compress_pdf(file_path: str, output_path: str = None):
    doc = fitz.open(file_path)

    # Use incremental save with aggressive compression,
    # then fall back to full save if the result is smaller.
    out_path = output_path or file_path

    buffer = io.BytesIO()
    doc.save(buffer, garbage=4, deflate=True, clean=True)
    doc.close()

    compressed = buffer.getvalue()
    original = os.path.getsize(file_path)

    # Only overwrite if compression actually helped
    if len(compressed) < original:
        with open(out_path, "wb") as f:
            f.write(compressed)
        return len(compressed), original
    else:
        # Already optimised — keep original
        return original, original


def compress_image(file_path: str, output_path: str = None):
    out_path = output_path or file_path
    img = Image.open(file_path)

    ext = os.path.splitext(file_path)[1].lower()
    original = os.path.getsize(file_path)

    if ext in (".jpg", ".jpeg"):
        # JPEG: reduce quality to 70 (good balance)
        img.save(out_path, "JPEG", quality=70, optimize=True)
    elif ext == ".png":
        # PNG: convert to WebP for much smaller size
        webp_path = out_path.rsplit(".", 1)[0] + ".webp"
        img.save(webp_path, "WEBP", quality=75)
        if os.path.getsize(webp_path) < original:
            # Remove old PNG, keep WebP
            if out_path != file_path:
                os.remove(out_path)
            os.rename(webp_path, out_path)
        else:
            if os.path.exists(webp_path):
                os.remove(webp_path)
            img.save(out_path, "PNG", optimize=True)
    elif ext == ".webp":
        img.save(out_path, "WEBP", quality=75)

    compressed = os.path.getsize(out_path)
    return compressed, original


def compress_docx(file_path: str, output_path: str = None):
    out_path = output_path or file_path
    original = os.path.getsize(file_path)

    doc = Document(file_path)

    # Re-save — python-docx already uses deflate ZIP internally
    doc.save(out_path)

    compressed = os.path.getsize(out_path)
    return compressed, original


async def compress_file(file_path: str, file_name: str) -> tuple[int, int]:
    """Compress a file in-place. Returns (new_size, original_size)."""
    ext = os.path.splitext(file_name)[1].lower()

    if ext == ".pdf":
        return compress_pdf(file_path)
    elif ext in (".jpg", ".jpeg", ".png", ".webp"):
        return compress_image(file_path)
    elif ext == ".docx":
        return compress_docx(file_path)
    else:
        original = os.path.getsize(file_path)
        return original, original
