# Tender-Aggregator/utils/document_processor.py

import logging
from pathlib import Path
from typing import Optional

# Check for dependencies and provide helpful error messages
try:
    import PyPDF2
except ImportError:
    print("ERROR: PyPDF2 is not installed. Please run 'pip install \"PyPDF2<3.0\"'")
    PyPDF2 = None

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is not installed. Please run 'pip install Pillow'")
    Image = None

try:
    import pytesseract
except ImportError:
    print("ERROR: pytesseract is not installed. Please run 'pip install pytesseract'")
    pytesseract = None

# Configure logging for this utility
logger = logging.getLogger("document_processor")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] (DocProc) %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """
    Extracts text from a PDF file.
    First, it tries direct text extraction. If that fails or yields little text,
    it falls back to OCR using Tesseract.
    """
    if not all([PyPDF2, Image, pytesseract]):
        logger.error("One or more required libraries (PyPDF2, Pillow, pytesseract) are not installed. Cannot process PDF.")
        return None
        
    if not pdf_path.is_file():
        logger.error(f"PDF file not found at: {pdf_path}")
        return None

    logger.info(f"Processing PDF: {pdf_path.name}")
    full_text = ""

    # --- Method 1: Direct Text Extraction ---
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            if reader.is_encrypted:
                try:
                    reader.decrypt('')
                except Exception:
                    logger.warning(f"Could not decrypt PDF {pdf_path.name}. OCR will likely fail.")
            
            for page in reader.pages:
                full_text += page.extract_text() or ""
        
        logger.info(f"Direct text extraction yielded {len(full_text)} characters.")
        # If we get a reasonable amount of text, we can consider it successful.
        # Threshold can be adjusted. 500 chars is arbitrary.
        if len(full_text.strip()) > 500:
            logger.info("Direct text extraction seems successful. Returning result.")
            return full_text.strip()
        else:
            logger.info("Direct text extraction yielded very little text. Falling back to OCR.")
            full_text = "" # Reset text for OCR pass
    except Exception as e:
        logger.warning(f"Could not extract text directly from {pdf_path.name}: {e}. Attempting OCR.")
        full_text = "" # Reset text for OCR pass

    # --- Method 2: OCR Fallback ---
    try:
        # We use Pillow to open the PDF and pytesseract to process each page image
        # This requires Tesseract engine and poppler utils to be installed on the system
        from pdf2image import convert_from_path
        pages_as_images = convert_from_path(pdf_path, dpi=300)
        
        for i, page_image in enumerate(pages_as_images):
            logger.info(f"Performing OCR on page {i+1} of {len(pages_as_images)}...")
            text_from_page = pytesseract.image_to_string(page_image, lang='eng') # Specify English
            full_text += text_from_page + "\n\n"
        
        logger.info(f"OCR process yielded {len(full_text)} characters.")
        return full_text.strip()
    except ImportError:
        logger.error("pdf2image is not installed ('pip install pdf2image'). Cannot perform OCR.")
        return None
    except Exception as e:
        logger.error(f"An error occurred during OCR processing for {pdf_path.name}: {e}")
        logger.error("Ensure Tesseract OCR engine and poppler-utils are installed on your system.")
        # On Fedora/CentOS: sudo dnf install tesseract poppler-utils
        # On Debian/Ubuntu: sudo apt-get install tesseract-ocr poppler-utils
        return None

if __name__ == '__main__':
    # This allows for direct testing of the script
    parser = argparse.ArgumentParser(description="Extract text from a PDF file using direct and OCR methods.")
    parser.add_argument("pdf_file", type=Path, help="Path to the PDF file to process.")
    args = parser.parse_args()

    extracted_text = extract_text_from_pdf(args.pdf_file)

    if extracted_text:
        output_file = args.pdf_file.with_suffix('.txt')
        output_file.write_text(extracted_text, encoding='utf-8')
        print(f"\nSuccessfully extracted text. Output saved to: {output_file}")
    else:
        print("\nFailed to extract text from the PDF.")
