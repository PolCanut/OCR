import cv2
import pytesseract
import os
import shutil
import numpy as np

# Ruta donde tienes tesseract instalado
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGE_FOLDER = r"letras"
OUTPUT_FOLDER = r"txts"
REVISION_FOLDER = r"revision"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REVISION_FOLDER, exist_ok=True)

WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

def add_margin(img, margin=10):
    return cv2.copyMakeBorder(img, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=[255,255,255])

def preprocess_image(img):
    # Redimensionar para caracteres pequeños
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Binarización adaptativa
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Reducción de ruido
    thresh = cv2.medianBlur(thresh, 3)

    # Añadir margen
    thresh = add_margin(thresh, margin=10)

    # Invertir si fondo oscuro
    white_ratio = cv2.countNonZero(thresh) / (thresh.shape[0]*thresh.shape[1])
    if white_ratio < 0.5:
        thresh = cv2.bitwise_not(thresh)

    return thresh

def ocr_image(img):
    config = f'--oem 3 --psm 10 -c tessedit_char_whitelist={WHITELIST}'
    text = pytesseract.image_to_string(img, config=config).strip()
    # Limpiar caracteres inválidos
    text = ''.join([c for c in text if c in WHITELIST])
    return text

# Contador de imágenes procesadas
counter = 1

# Procesar todas las imágenes
for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        img_path = os.path.join(IMAGE_FOLDER, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] No se pudo leer {filename}")
            continue
        
        processed_img = preprocess_image(img)
        text = ocr_image(processed_img)
        
        if text == "":
            # Mover imagen a revisión sin crear .txt
            revision_path = os.path.join(REVISION_FOLDER, filename)
            shutil.move(img_path, revision_path)
            print(f"[{counter}] REVISION {filename} -> sin letras detectadas")
        else:
            # Guardar texto solo si se detecta al menos un carácter
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(OUTPUT_FOLDER, txt_filename)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            
            print(f"[{counter}] OK Procesado {filename} -> '{text}'")
        
        counter += 1