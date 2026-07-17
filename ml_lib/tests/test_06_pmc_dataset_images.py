import ctypes
import os
import numpy as np
from PIL import Image
import random

# Chargement lib Rust
lib_path = os.path.join(os.path.dirname(__file__), "../target/debug/libml_lib.dylib")
ml_lib = ctypes.CDLL(lib_path)

ml_lib.create_pmc.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float]
ml_lib.create_pmc.restype = ctypes.c_void_p
ml_lib.pmc_train_step.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float]
ml_lib.pmc_train_step.restype = None
ml_lib.pmc_predict.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
ml_lib.pmc_predict.restype = ctypes.c_float
ml_lib.free_pmc.argtypes = [ctypes.c_void_p]
ml_lib.free_pmc.restype = None

# Chargement images
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
points   = charger_images(os.path.join(data_dir, "avions"),   1.0)
points  += charger_images(os.path.join(data_dir, "voitures"), -1.0)
random.shuffle(points)

split = int(len(points) * 0.8)
train = points[:split]
test  = points[split:]

N = 3072  # 32*32*3

# Le PMC prend toujours 2 entrées dans notre implémentation actuelle on va donc réduire les images à 2 features via moyenne RGB
def reduire_features(pixels):
    r = pixels[0:1024].mean()
    g = pixels[1024:2048].mean()
    return r, g

pmc = ml_lib.create_pmc(2, 16, 0.01)

for epoch in range(500):
    random.shuffle(train)
    for (pixels, label) in train:
        x1, x2 = reduire_features(pixels)
        ml_lib.pmc_train_step(pmc, x1, x2, label)

# Test
correct = 0
for (pixels, label) in test:
    x1, x2 = reduire_features(pixels)
    pred = ml_lib.pmc_predict(pmc, x1, x2)
    if pred == label:
        correct += 1

accuracy = correct / len(test) * 100
print(f"Images test : {len(test)}")
print(f"Précision PMC sur images réelles : {accuracy:.1f}%")

ml_lib.free_pmc(pmc)