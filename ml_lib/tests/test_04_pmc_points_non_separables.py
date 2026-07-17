import ctypes
import os
import random
import math
import matplotlib.pyplot as plt

# Chargement lib Rust
lib_path = os.path.join(os.path.dirname(__file__), "../target/debug/libml_lib.dylib")
ml_lib = ctypes.CDLL(lib_path)

ml_lib.create_pmc.argtypes  = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float]
ml_lib.create_pmc.restype   = ctypes.c_void_p
ml_lib.pmc_train_step.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float]
ml_lib.pmc_train_step.restype  = None
ml_lib.pmc_predict.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
ml_lib.pmc_predict.restype  = ctypes.c_float
ml_lib.free_pmc.argtypes    = [ctypes.c_void_p]
ml_lib.free_pmc.restype     = None

# Même dataset que test_non_lineaire.py — points en cercles
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

# Entraînement PMC : 2 entrées, 8 neurones cachés, learning rate 0.1
pmc = ml_lib.create_pmc(2, 8, 0.1)

for epoch in range(5000):
    random.shuffle(points)
    for (x1, x2, y) in points:
        ml_lib.pmc_train_step(pmc, x1, x2, y)

# Calcul précision
correct = sum(
    1 for (x1, x2, y) in points
    if ml_lib.pmc_predict(pmc, x1, x2) == y
)
accuracy = correct / len(points) * 100
print(f"Précision PMC : {accuracy:.1f}%")

# Affichage points
fig, ax = plt.subplots(figsize=(7, 7))
for (x1, x2, y) in points:
    ax.scatter(x1, x2, color="blue" if y == 1.0 else "red", s=30)

# Affichage frontière de décision du PMC on colorie chaque pixel de la grille selon la prédiction du PMC
step = 0.05
for gx in [i * step - 1.5 for i in range(61)]:
    for gy in [i * step - 1.5 for i in range(61)]:
        pred = ml_lib.pmc_predict(pmc, gx, gy)
        couleur = "lightblue" if pred == 1.0 else "lightsalmon"
        ax.scatter(gx, gy, color=couleur, s=8, alpha=0.3, zorder=0)

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_title(f"PMC — Cas NON linéaire (précision={accuracy:.1f}%)")
plt.tight_layout()
plt.show()

ml_lib.free_pmc(pmc)