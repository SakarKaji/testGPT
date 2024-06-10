import fitz
import os
import cv2
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from pdfplumber import open as open_pdf
from typing import Union
from pdf2image import convert_from_bytes
from PIL import Image
# from surya.ocr import run_ocr
# from surya.model.detection import segformer
# from surya.model.recognition.model import load_model
# from surya.model.recognition.processor import load_processor


class GetOCRText:
    def __init__(self) -> None:
        self._image = None
        self.doctr = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True).to(device)

    def _preprocess_image(self, img):
        resized_image = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def extract_text(self, cell_image: Union[None, bytes] = None, image_file: bool = False, file_path: Union[None, str] = None):
        text = ""

        if image_file:
            if file_path is None:
                raise ValueError("file_path must be provided when image_file is True.")
            pdf_file = DocumentFile.from_images(file_path)
            result = self.doctr(pdf_file)
            output = result.export()
        else:
            if cell_image is None:
                raise ValueError("cell_image must be provided when image_file is False.")

            if isinstance(cell_image, bytes):
                images = convert_from_bytes(cell_image)
                pdf_file = DocumentFile.from_images(images)
                result = self.doctr(pdf_file)
            else:
                self._image = cell_image
                preprocessed_image = self._preprocess_image(self._image)
                result = self.doctr([preprocessed_image])
                output = result.export()

        for obj1 in output['pages'][0]["blocks"]:
            for obj2 in obj1["lines"]:
                for obj3 in obj2["words"]:
                    text += (f"{obj3['value']} ").replace("\n", "")
                text += "\n"
            text += "\n"
        if text:
            return text.strip()
        return " "

def process_images_and_generate_doc(pdf_path: str, upload_dir: str):
    ocr = GetOCRText()
    pdf_writer = fitz.open()

    pdf_doc = fitz.open(pdf_path)
    
    for page_index in range(len(pdf_doc)):
        page = pdf_doc[page_index]
        image_list = page.get_images()

        if not image_list:
            continue

        for image_index, img in enumerate(image_list, start=1):
            xref = img[0]
            pix = fitz.Pixmap(pdf_doc, xref)

            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)(
                    "RGB", [pix.width, pix.height], pix.samples)

            image_path = f"temp_page_{page_index}_image_{image_index}.png"
            pix.save(image_path)

            extracted_text = ocr.extract_text(
                image_file=True, file_path=image_path)
            

            # Create a new page with the same dimensions as the original page
            pdf_page = pdf_writer.new_page(width=page.rect.width, height=page.rect.height)
            pdf_page.insert_text((10, 10), extracted_text, fontsize=10)
            os.remove(image_path)

    save_path = os.path.join(upload_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_searchable.pdf")
    pdf_writer.save(save_path)
    pdf_writer.close()
    return save_path

# Example usage
process_images_and_generate_doc("Unified Branch Operations Manual.pdf", 'hello')
