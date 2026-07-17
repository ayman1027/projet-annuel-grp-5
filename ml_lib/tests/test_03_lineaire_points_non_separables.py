import ctypes
import os
import random
import math
import matplotlib.pyplot as plt

# Chargement lib Rust
lib_path = os.path.join(os.path.dirname(__file__), "../target/debug/libml_lib.dylib")
ml_lib = ctypes.CDLL(lib_path)

ml_lib.create_linear_model.argtypes = [ctypes.c_float]
ml_lib.create_linear_model.restype  = ctypes.c_void_p
ml_lib.train_step.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float]
ml_lib.train_step.restype  = None
ml_lib.free_linear_model.argtypes = [ctypes.c_void_p]
ml_lib.free_linear_model.restype  = None

# Points en cercles — rouges au centre, bleus autour
random.seed(42)
points = []

# 50 points rouges au centre (rayon < 0.5)
for _ in range(50):
    angle = random.uniform(0, 2 * math.pi)
    rayon = random.uniform(0, 0.5)
    x1 = rayon * math.cos(angle)
    x2 = rayon * math.sin(angle)
    points.append((x1, x2, -1.0))

# 50 points bleus autour (rayon entre 0.7 et 1.0)
for _ in range(50):
    angle = random.uniform(0, 2 * math.pi)
    rayon = random.uniform(0.7, 1.0)
    x1 = rayon * math.cos(angle)
    x2 = rayon * math.sin(angle)
    points.append((x1, x2, 1.0))

# Entraînement du modèle linéaire
model = ml_lib.create_linear_model(0.01)
for epoch in range(1000):
    random.shuffle(points)
    for (x1, x2, y) in points:
        ml_lib.train_step(model, x1, x2, y)

# Récupération des poids
class LinearModel(ctypes.Structure):
    _fields_ = [("w1", ctypes.c_float), ("w2", ctypes.c_float),
                ("bias", ctypes.c_float), ("lr", ctypes.c_float)]

m = ctypes.cast(model, ctypes.POINTER(LinearModel))[0]

# Calcul précision — combien de points bien classés ?
correct = 0
for (x1, x2, y) in points:
    prediction = m.w1 * x1 + m.w2 * x2 + m.bias
    if (prediction >= 0 and y == 1.0) or (prediction < 0 and y == -1.0):
        correct += 1
accuracy = correct / len(points) * 100
print(f"Précision modèle linéaire : {accuracy:.1f}%")
print("→ Le modèle linéaire échoue car une droite ne peut pas séparer des cercles")

# Affichage
fig, ax = plt.subplots(figsize=(7, 7))
for (x1, x2, y) in points:
    ax.scatter(x1, x2, color="blue" if y == 1.0 else "red", s=30)

x_vals = [-1.5, 1.5]
if abs(m.w2) > 0.0001:
    y_vals = [(-m.w1 * x - m.bias) / m.w2 for x in x_vals]
    ax.plot(x_vals, y_vals, 'g-', linewidth=2, label=f"Frontière linéaire (précision={accuracy:.1f}%)")

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_title("Modèle Linéaire — Cas NON linéairement séparable (ÉCHEC)")
ax.legend()
plt.tight_layout()
plt.show()

ml_lib.free_linear_model(model)