import ctypes
import os
import random
import math
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

# Points en cercles - rouges au centre, bleus autour
random.seed(42)
points = []

for _ in range(50):
    angle = random.uniform(0, 2 * math.pi)
    rayon = random.uniform(0, 0.5)
    points.append((rayon * math.cos(angle), rayon * math.sin(angle), -1.0))

for _ in range(50):
    angle = random.uniform(0, 2 * math.pi)
    rayon = random.uniform(0.7, 1.0)
    points.append((rayon * math.cos(angle), rayon * math.sin(angle), 1.0))

N = 2
model = ml_lib.create_linear_model(N, ctypes.c_float(0.01))

for epoch in range(1000):
    random.shuffle(points)
    for (x1, x2, y) in points:
        arr = np.array([x1, x2], dtype=np.float32)
        x_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ml_lib.train_step(model, x_ptr, N, ctypes.c_float(y))

correct = 0
for (x1, x2, y) in points:
    arr = np.array([x1, x2], dtype=np.float32)
    x_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    pred = ml_lib.predict(model, x_ptr, N)
    if (pred >= 0 and y == 1.0) or (pred < 0 and y == -1.0):
        correct += 1

accuracy = correct / len(points) * 100
print(f"Precision modele lineaire : {accuracy:.1f}%")
print("Le modele lineaire echoue car une droite ne peut pas separer des cercles")

fig, ax = plt.subplots(figsize=(7, 7))
for (x1, x2, y) in points:
    ax.scatter(x1, x2, color="blue" if y == 1.0 else "red", s=30)

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_title(f"Modele Lineaire - Cas NON separable (precision={accuracy:.1f}%)")
plt.tight_layout()
plt.show()

ml_lib.free_linear_model(model)