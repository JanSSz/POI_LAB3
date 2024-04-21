import os
import cv2
import numpy as np
from skimage.feature import graycomatrix
from skimage.feature import graycoprops as greycoprops
import pandas as pd

def crop_textures(image_dir, output_dir, crop_size):
    # Lista przechowująca wektory cech dla wszystkich obrazów
    all_feature_vectors = []

    # Liczba fragmentów wzdłuż osi X i Y dla najmniejszego zdjęcia
    min_img_size = float('inf')
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(image_dir, filename)
            image = cv2.imread(img_path)
            min_img_size = min(min_img_size, min(image.shape[:2]))

    num_x = (min_img_size // crop_size)
    num_y = (min_img_size // crop_size)

    # Iteracja przez pliki w katalogu z obrazami
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(image_dir, filename)
            image = cv2.imread(img_path)

            # Utwórz folder dla każdej kategorii tekstury, jeśli nie istnieje
            category_name = os.path.splitext(filename)[0]
            category_folder = os.path.join(output_dir, category_name)
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)

            # Wycięcie fragmentów tekstury o zadanym rozmiarze - wycinarka
            for i in range(0, num_y * crop_size, crop_size):
                for j in range(0, num_x * crop_size, crop_size):
                    crop = image[i:i+crop_size, j:j+crop_size]

                    # Zapisz przycięte fragmenty przed konwersją na szarość
                    cut_up_folder = os.path.join(category_folder, "cut_up")
                    if not os.path.exists(cut_up_folder):
                        os.makedirs(cut_up_folder)
                    cut_up_img_path = os.path.join(cut_up_folder, f"{category_name}_{i}_{j}_cut_up.jpg")
                    cv2.imwrite(cut_up_img_path, crop)

                    # Konwersja obrazu na skale szarości
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

                    # Zapisz przycięte fragmenty w szarych kolorach
                    gray_folder = os.path.join(category_folder, "gray")
                    if not os.path.exists(gray_folder):
                        os.makedirs(gray_folder)
                    gray_img_path = os.path.join(gray_folder, f"{category_name}_{i}_{j}_gray.jpg")
                    cv2.imwrite(gray_img_path, gray_crop)

                    # Obliczanie macierzy zdarzeń dla fragmentu tekstury
                    glcm = graycomatrix(gray_crop, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)

                    # Obliczanie danych z zadania
                    dissimilarity = greycoprops(glcm, 'dissimilarity').ravel()
                    correlation = greycoprops(glcm, 'correlation').ravel()
                    contrast = greycoprops(glcm, 'contrast').ravel()
                    energy = greycoprops(glcm, 'energy').ravel()
                    homogeneity = greycoprops(glcm, 'homogeneity').ravel()
                    asm = greycoprops(glcm, 'ASM').ravel()

                    # Utworzenie nazwy pliku danych
                    features_filename = f"{category_name}_{i}_{j}_features.txt"

                    # Zapis danych do pliku
                    features_file_path = os.path.join(category_folder, features_filename)
                    with open(features_file_path, 'w') as f:
                        f.write(f"Dissimilarity: {dissimilarity}\n")
                        f.write(f"Correlation: {correlation}\n")
                        f.write(f"Contrast: {contrast}\n")
                        f.write(f"Energy: {energy}\n")
                        f.write(f"Homogeneity: {homogeneity}\n")
                        f.write(f"ASM: {asm}\n")

                    # Dodanie wektora cech do listy wszystkich wektorów cech - sumowanie
                    all_feature_vectors.append({
                        'Category': category_name,
                        'File': features_filename,
                        'Dissimilarity': dissimilarity,
                        'Correlation': correlation,
                        'Contrast': contrast,
                        'Energy': energy,
                        'Homogeneity': homogeneity,
                        'ASM': asm
                    })

    # Konwersja listy wektorów cech na ramkę danych Pandas
    df = pd.DataFrame(all_feature_vectors)

    # Ścieżka do pliku CSV
    csv_file_path = os.path.join(output_dir, 'feature_vectors.csv')

    # Zapisanie ramki danych do pliku CSV
    df.to_csv(csv_file_path, index=False)

# Ścieżka do katalogu zawierającego zdjęcia
data_image_directory = "C:\\Users\\Hyperbook\\Desktop\\data_image_directory"

# Ścieżka do katalogu, w którym zostaną zapisane wycięte fragmenty tekstur
output_directory = "C:\\Users\\Hyperbook\\Desktop\\output_directory"

# Rozmiar wyciętych fragmentów
crop_size = 128

# Wywołanie funkcji do wycinania fragmentów tekstur
crop_textures(data_image_directory, output_directory, crop_size)
