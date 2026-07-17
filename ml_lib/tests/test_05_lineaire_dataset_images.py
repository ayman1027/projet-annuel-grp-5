import ctypes
import os
import numpy as np
from PIL import Image
import random

# Chargement lib Rust
lib_path = os.path.join(os.path.dirname(__file__), "../target/debug/libml_lib.dylib")
ml_lib = ctypes.CDLL(lib_path)

ml_lib.create_linear_model.argtypes = [ctypes.c_size_t, ctypes.c_float]
ml_lib.create_linear_model.restype = ctypes.c_void_p
ml_lib.train_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_float]
ml_lib.train_step.restype = None
ml_lib.predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
ml_lib.predict.restype = ctypes.c_float
ml_lib.free_linear_model.argtypes = [ctypes.c_void_p]
ml_lib.free_linear_model.restype = None

# Chargement des images
def charger_images(dossier, label, taille=(32, 32)):
    data = []
    for fichier in os.listdir(dossier):
        if fichier.endswith((".jpg", ".jpeg", ".png")):
            chemin = os.path.join(dossier, fichier)
            img = Image.open(chemin).convert("RGB").resize(taille)
            pixels = np.array(img, dtype=np.float32).flatten() / 255.0
            data.append((pixels, label))
    return data

data_dir = os.path.join(os.path.dirname(__file__), "../data")
points  = charger_images(os.path.join(data_dir, "avions"),   1.0)
points += charger_images(os.path.join(data_dir, "voitures"), -1.0)

random.shuffle(points)

# Split 80% train / 20% test
split  = int(len(points) * 0.8)
train  = points[:split]
test   = points[split:]

N = 3072  # 32*32*3 pixels par image

# Entraînement
model = ml_lib.create_linear_model(N, 0.01)

for epoch in range(100):
    random.shuffle(train)
    for (pixels, label) in train:
        x_ctypes = pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ml_lib.train_step(model, x_ctypes, N, label)

# Test
correct = 0
for (pixels, label) in test:
    x_ctypes = pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    pred = ml_lib.predict(model, x_ctypes, N)
    if (pred >= 0 and label == 1.0) or (pred < 0 and label == -1.0):
        correct += 1

accuracy = correct / len(test) * 100
print(f"Images test : {len(test)}")
print(f"Précision modèle linéaire sur images réelles : {accuracy:.1f}%")

ml_lib.free_linear_model(model)