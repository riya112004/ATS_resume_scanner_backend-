import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from docx import Document

# Set Tesseract Path for Windows/Linux
if os.name == 'nt': # Windows
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
else: # Linux
    # On Linux, tesseract is usually in the PATH automatically
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'

async def extract_text_from_file(file_path: str) -> str:
    """Extracts text from a given file path based on its extension."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == ".pdf":
        text = await extract_text_from_pdf(file_path)
        # If text is empty (scanned image), try OCR
        if not text or len(text.strip()) < 10:
            print("Normal PDF extraction failed. Attempting OCR...")
            text = await extract_text_with_ocr(file_path)
        return text
    elif ext in [".webp", ".png", ".jpg", ".jpeg"]:
        print(f"Image file detected ({ext}). Using OCR directly...")
        return await extract_image_text_with_ocr(file_path)
    elif ext == ".docx":
        return await extract_text_from_docx(file_path)
    elif ext == ".doc":
        raise ValueError(".doc format is not yet supported. Please use .docx or .pdf.")
    else:
        raise ValueError(f"Unsupported file format: {ext}")

async def extract_image_text_with_ocr(file_path: str) -> str:
    """Extracts text directly from an image file."""
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"Image OCR Error: {e}")
        raise ValueError(f"OCR failed for image: {str(e)}. Ensure Tesseract-OCR is installed.")

async def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            page_text = page.get_text("text") 
            if page_text:
                text += page_text + "\n"
        doc.close()
    except Exception as e:
        print(f"PyMuPDF error: {e}")
    return text.strip()

async def extract_text_with_ocr(file_path: str) -> str:
    """Converts PDF pages to images using PyMuPDF and uses Tesseract to extract text."""
    text = ""
    doc = None
    try:
        # Open PDF with PyMuPDF
        doc = fitz.open(file_path)
        
        for i in range(len(doc)):
            page = doc.load_page(i)
            # Render page to a high-resolution image (300 DPI)
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            
            # Convert pixmap to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Perform OCR
            page_text = pytesseract.image_to_string(img)
            if page_text:
                text += page_text + "\n"
                print(f"OCR Page {i+1} completed.")
        
    except Exception as e:
        print(f"OCR Error: {e}")
        raise ValueError(f"OCR failed: {str(e)}. Ensure Tesseract-OCR is installed at C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
    finally:
        if doc:
            doc.close() # CRITICAL: Close the file handle to release the lock
        
    return text.strip()

async def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()
