from flask import Flask, request, jsonify, render_template
import ctypes
import os
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Je charge ma lib Rust compilee - c'est elle qui fait les calculs
lib_path = os.path.join(os.path.dirname(__file__), "ml_lib/target/debug/libml_lib.dylib")
ml_lib = ctypes.cdll.LoadLibrary(lib_path)

# Je dis a Python comment appeler les fonctions Rust du PMC
# sans ca Python envoie n'importe quoi et ca plante
ml_lib.load_pmc.argtypes = [ctypes.c_char_p]
ml_lib.load_pmc.restype  = ctypes.c_void_p
ml_lib.pmc_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)]
ml_lib.pmc_predict.restype  = None
ml_lib.free_pmc.argtypes = [ctypes.c_void_p]
ml_lib.free_pmc.restype  = None

# Pareil pour le RBFN
ml_lib.load_rbfn.argtypes = [ctypes.c_char_p]
ml_lib.load_rbfn.restype  = ctypes.c_void_p
ml_lib.rbfn_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)]
ml_lib.rbfn_predict.restype  = None
ml_lib.free_rbfn.argtypes = [ctypes.c_void_p]
ml_lib.free_rbfn.restype  = None

# Je convertis l'image recue en tableau de pixels
# niveaux de gris 32x32 = 1024 nombres entre 0 et 1
def image_vers_pixels(image_bytes, taille=(32, 32)):
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize(taille)
    pixels = np.array(img, dtype=np.float32).flatten() / 255.0
    return pixels

# Mes 3 classes et mes modeles charges en memoire
CLASSES = ["avion", "voiture", "bateau"]
MODELES = {"pmc": {}, "rbfn": {}}

# Je charge les 6 modeles une seule fois au demarrage
# comme ca je les recharge pas a chaque requete - c est beaucoup plus rapide
def charger_modeles():
    for classe in CLASSES:
        MODELES["pmc"][classe]  = ml_lib.load_pmc(f"pmc_{classe}.bin".encode())
        MODELES["rbfn"][classe] = ml_lib.load_rbfn(f"rbfn_{classe}.bin".encode())
    print(f"Modeles charges : {len(CLASSES)} PMC + {len(CLASSES)} RBFN")

# Pour chaque image je calcule un score par classe et je prends celle avec le score le plus haut
# c est l approche OVR - One Versus Rest - la meme qu on utilise dans le notebook
def predire(type_modele, predict_fn, pixels):
    N = len(pixels)
    x_ptr = np.ctypeslib.as_ctypes(pixels)
    scores = []
    for classe in CLASSES:
        output = np.zeros(1, dtype=np.float32)
        out_ptr = np.ctypeslib.as_ctypes(output)
        predict_fn(MODELES[type_modele][classe], x_ptr, N, out_ptr)
        scores.append(float(output[0]))
    classe_predite = CLASSES[scores.index(max(scores))]
    return classe_predite, scores

# La page web du site
@app.route("/app", methods=["GET"])
def interface():
    return render_template("index.html")

# Route pour verifier que le serveur tourne
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Serveur ML operationnel", "modeles": ["pmc", "rbfn"]})

# Quand l'utilisateur envoie une image on la classifie avec le PMC
@app.route("/predict/pmc", methods=["POST"])
def predict_pmc():
    if "image" not in request.files:
        return jsonify({"erreur": "Pas d image"}), 400
    pixels = image_vers_pixels(request.files["image"].read())
    classe_predite, scores = predire("pmc", ml_lib.pmc_predict, pixels)
    return jsonify({"prediction": classe_predite, "score": max(scores), "scores": dict(zip(CLASSES, scores))})

# Meme chose mais avec le RBFN
@app.route("/predict/rbfn", methods=["POST"])
def predict_rbfn():
    if "image" not in request.files:
        return jsonify({"erreur": "Pas d image"}), 400
    pixels = image_vers_pixels(request.files["image"].read())
    classe_predite, scores = predire("rbfn", ml_lib.rbfn_predict, pixels)
    return jsonify({"prediction": classe_predite, "score": max(scores), "scores": dict(zip(CLASSES, scores))})

if __name__ == "__main__":
    charger_modeles()
    app.run(debug=True, port=5000)