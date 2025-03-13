import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import subprocess
import os

AI_KEYWORDS = ["Stable Diffusion", "Midjourney", "DALL·E", "Leonardo.Ai", "Runway", "DeepAI", "DreamStudio"]


def analyze_exif(image_path):
    try:
        result = subprocess.run(["exiftool", image_path], capture_output=True, text=True)
        exif_data = result.stdout

        # Sprawdzenie, czy w EXIF są wzmianki o AI
        for keyword in AI_KEYWORDS:
            if keyword.lower() in exif_data.lower():
                return f"Podejrzenie AI! Wykryto '{keyword}' w metadanych EXIF."

        return "Brak podejrzeń o AI w EXIF."
    except Exception as e:
        return f"Błąd analizy EXIF: {e}"


def error_level_analysis(image_path, output_path="ela_image.jpg", quality=90):
    try:
        original = Image.open(image_path).convert("RGB")
        resaved_path = "temp_resaved.jpg"
        original.save(resaved_path, quality=quality)
        resaved = Image.open(resaved_path)

        ela_image = ImageChops.difference(original, resaved)
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff if max_diff != 0 else 1.0
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        ela_image.save(output_path)

        os.remove(resaved_path)

        if max_diff < 50:
            return output_path, "Podejrzenie AI! Obraz ma niskie różnice ELA – może być wygenerowany."
        else:
            return output_path, "ELA wskazuje na naturalne zdjęcie."

    except Exception as e:
        return None, f"Błąd analizy ELA: {e}"


def detect_faces(image_path):
    """Wykrywanie twarzy na zdjęciu."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    detected_faces_path = "faces_detected.jpg"
    cv2.imwrite(detected_faces_path, image)

    if len(faces) > 0:
        if len(faces) > 3:
            return detected_faces_path, f"Podejrzenie AI! Wykryto {len(faces)} twarze – możliwa generacja."
        else:
            return detected_faces_path, f"Wykryto {len(faces)} twarz(e) – wygląda naturalnie."
    else:
        return detected_faces_path, "Brak wykrytych twarzy – trudno ocenić."


if __name__ == "__main__":
    image_path = input("Podaj ścieżkę do obrazu: ").strip().strip('"')

    print("\nAnaliza EXIF...")
    exif_result = analyze_exif(image_path)
    print(exif_result)

    print("\nWykrywanie manipulacji (ELA)...")
    ela_output, ela_result = error_level_analysis(image_path)
    print(ela_result)

    print("\nWykrywanie twarzy...")
    faces_output, faces_result = detect_faces(image_path)
    print(faces_result)

    print("\n**Podsumowanie:**")
    if "Podejrzenie AI" in exif_result or "Podejrzenie AI" in ela_result or "Podejrzenie AI" in faces_result:
        print("WNIOSEK: Prawdopodobnie jest to obraz wygenerowany przez AI!")
    else:
        print("WNIOSEK: Obraz wygląda na naturalny.")

