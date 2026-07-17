import ctypes
import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Chargement lib Rust
lib_path = os.path.join(os.path.dirname(__file__), "../target/debug/libml_lib.dylib")
ml_lib = ctypes.CDLL(lib_path)

ml_lib.create_linear_model.argtypes = [ctypes.c_size_t, ctypes.c_float]
ml_lib.create_linear_model.restype  = ctypes.c_void_p
ml_lib.train_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_float]
ml_lib.train_step.restype  = None
ml_lib.predict.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
ml_lib.predict.restype     = ctypes.c_float
ml_lib.free_linear_model.argtypes = [ctypes.c_void_p]
ml_lib.free_linear_model.restype  = None

# Dataset
random.seed(42)
points = []
for _ in range(50):
    points.append((random.uniform(-1, 0), random.uniform(0, 1),  1.0))
for _ in range(50):
    points.append((random.uniform(0, 1),  random.uniform(-1, 0), -1.0))

# Entraînement
LEARNING_RATE = 0.01
EPOCHS        = 1000
N             = 2  # 2 features : x1 et x2

model = ml_lib.create_linear_model(N, LEARNING_RATE)

for epoch in range(EPOCHS):
    random.shuffle(points)
    for (x1, x2, y) in points:
        arr   = np.array([x1, x2], dtype=np.float32)
        x_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ml_lib.train_step(model, x_ptr, N, y)

# Affichage des points
fig, ax = plt.subplots(figsize=(7, 7))
for (x1, x2, y) in points:
    ax.scatter(x1, x2, color="blue" if y == 1.0 else "red", s=30)

# Calcul précision
correct = 0
for (x1, x2, y) in points:
    arr   = np.array([x1, x2], dtype=np.float32)
    x_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    pred  = ml_lib.predict(model, x_ptr, N)
    if (pred >= 0 and y == 1.0) or (pred < 0 and y == -1.0):
        correct += 1
accuracy = correct / len(points) * 100
print(f"Précision modèle linéaire : {accuracy:.1f}%")

# Tracé de la frontière via grille
x_vals = np.linspace(-1.5, 1.5, 100)
y_vals = []
for xv in x_vals:
    arr   = np.array([xv, 0.0], dtype=np.float32)
    x_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    y_vals.append(-ml_lib.predict(model, x_ptr, N))

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_title(f"Modèle Linéaire — Points séparables (précision={accuracy:.1f}%)")
plt.tight_layout()
plt.show()

ml_lib.free_linear_model(model)