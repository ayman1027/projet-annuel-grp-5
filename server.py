# Auteur : Messali Ayman
# Serveur Flask qui expose nos modeles Rust via une API

from flask import Flask, request, jsonify, render_template
import ctypes
import os
import numpy as np
from PIL import Image
import io
import math

app = Flask(__name__)

# Chargement de la lib Rust
lib_path = os.path.join(os.path.dirname(__file__), "ml_lib/target/debug/libml_lib.dylib")
ml_lib = ctypes.cdll.LoadLibrary(lib_path)

# Declaration des types
ml_lib.load_linear_model.argtypes = [ctypes.c_char_p]
ml_lib.load_linear_model.restype  = ctypes.c_void_p
ml_lib.predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
ml_lib.predict.restype  = ctypes.c_float
ml_lib.free_linear_model.argtypes = [ctypes.c_void_p]
ml_lib.free_linear_model.restype  = None

ml_lib.load_pmc.argtypes = [ctypes.c_char_p]
ml_lib.load_pmc.restype  = ctypes.c_void_p
ml_lib.pmc_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)]
ml_lib.pmc_predict.restype  = None
ml_lib.free_pmc.argtypes = [ctypes.c_void_p]
ml_lib.free_pmc.restype  = None

# Conversion image en pixels en niveaux de gris 32x32
def image_vers_pixels(image_bytes, taille=(32, 32)):
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize(taille)
    pixels = np.array(img, dtype=np.float32).flatten() / 255.0
    return pixels

# Page web principale
@app.route("/app", methods=["GET"])
def interface():
    return render_template("index.html")

# Route de test
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Serveur ML operationnel", "modeles": ["lineaire", "pmc"]})

# Route prediction avec le modele lineaire
@app.route("/predict/lineaire", methods=["POST"])
def predict_lineaire():
    if "image" not in request.files:
        return jsonify({"erreur": "Pas d image"}), 400

    image_bytes = request.files["image"].read()
    pixels = image_vers_pixels(image_bytes)
    N = len(pixels)

    model = ml_lib.load_linear_model(b"model_lineaire.bin")
    x_ptr = np.ctypeslib.as_ctypes(pixels)
    pred = ml_lib.predict(model, x_ptr, N)
    ml_lib.free_linear_model(model)

    score = float(pred)
    if math.isnan(score):
        score = 0.0
    classe = "avion" if score >= 0 else "voiture"
    return jsonify({"prediction": classe, "score": score})

# Route prediction avec le PMC
@app.route("/predict/pmc", methods=["POST"])
def predict_pmc():
    if "image" not in request.files:
        return jsonify({"erreur": "Pas d image"}), 400

    image_bytes = request.files["image"].read()
    pixels = image_vers_pixels(image_bytes)
    N = len(pixels)

    model = ml_lib.load_pmc(b"pmc_images.bin")
    x_ptr = np.ctypeslib.as_ctypes(pixels)
    output = np.zeros(1, dtype=np.float32)
    out_ptr = np.ctypeslib.as_ctypes(output)
    ml_lib.pmc_predict(model, x_ptr, N, out_ptr)
    ml_lib.free_pmc(model)

    score = float(output[0])
    if math.isnan(score):
        score = 0.0
    classe = "avion" if score >= 0.5 else "voiture"
    return jsonify({"prediction": classe, "score": score})

if __name__ == "__main__":
    app.run(debug=True, port=5000)