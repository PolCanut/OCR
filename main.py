import cv2
import pytesseract
import os
import numpy as np

# Ruta donde tienes tesseract instalado (ajústala si es diferente)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Carpeta de imágenes
IMAGE_FOLDER = r"letras"
# Carpeta donde se guardarán los txt
OUTPUT_FOLDER = r"txts"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Configuración OCR
config_single = r'--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
config_line   = r'--oem 3 --psm 7  -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

def add_margin(img, margin=5):
    """
    Añade un borde blanco alrededor de la imagen.
    margin = cantidad de píxeles de margen.
    """
    return cv2.copyMakeBorder(
        img,
        margin, margin, margin, margin,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255]  # blanco
    )

def preprocess_image(img):
    """
    Preprocesa la imagen para mejorar el OCR:
    - Redimensiona
    - Convierte a gris
    - Mejora contraste
    - Binariza
    - Reduce ruido
    - Añade margen
    """
    # Escalar imagen para mejorar OCR de caracteres pequeños
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # Convertir a gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mejora contraste usando CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Binarización adaptativa
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Invertir si el fondo es negro y letra blanca
    white_ratio = cv2.countNonZero(thresh) / (thresh.shape[0] * thresh.shape[1])
    if white_ratio < 0.5:
        thresh = cv2.bitwise_not(thresh)

    # Reducción de ruido (opcional)
    thresh = cv2.medianBlur(thresh, 3)

    # Añadir margen
    thresh = add_margin(thresh, margin=10)

    return thresh

# Procesar todas las imágenes
for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        img_path = os.path.join(IMAGE_FOLDER, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] No se pudo leer {filename}")
            continue
        
        processed_img = preprocess_image(img)
        
        # Intentar primero como carácter único
        text = pytesseract.image_to_string(processed_img, config=config_single).strip()
        
        # Si no detecta nada, probar como línea corta
        if text == "":
            text = pytesseract.image_to_string(processed_img, config=config_line).strip()
        
        # Limpiar caracteres inválidos (solo alfanuméricos)
        text = ''.join([c for c in text if c.isalnum()])

        # Guardar texto en archivo .txt
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(OUTPUT_FOLDER, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        print(f"[OK] Procesado {filename} -> '{text}'")