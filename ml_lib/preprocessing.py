# Preprocessing : redimensionne toutes les images en 32x32 et les transforme en tableaux de pixels pour la lib Rust
from PIL import Image
import os
import numpy as np

def charger_dataset(data_dir, taille=(32, 32)):
    images = []
    labels = []

    # On associe chaque dossier à un label numérique
    classes = {"avions": 0, "voitures": 1, "bateau": 2}

    for classe, label in classes.items():
        dossier = os.path.join(data_dir, classe)
        for fichier in os.listdir(dossier):
            if fichier.endswith((".jpg", ".jpeg", ".png")):
                chemin = os.path.join(dossier, fichier)
                img = Image.open(chemin).convert("RGB")
                img = img.resize(taille)
                pixels = np.array(img, dtype=np.float32) / 255.0
                images.append(pixels.flatten())
                labels.append(label)

    return np.array(images), np.array(labels)

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    X, y = charger_dataset(data_dir)
    print(f"Dataset chargé : {len(X)} images")
    print(f"Taille d'une image : {X[0].shape} pixels")
    print(f"Classes : 0=avion, 1=voiture, 2=bateau")