# Projet Annuel IABD — Classification d'Images
Groupe 5

## Problematique
Distinguer automatiquement 3 classes d'images :
- Avion
- Voiture
- Bateau

A partir d une photo, notre site predit automatiquement la classe de l image grace a des modeles de Machine Learning implementes from scratch en Rust.

## Architecture technique

```
1-L utilisateur ouvre le site web dans son navigateur
2-Il envoie une image
3-Le serveur Python recoit l'image et la convertit en pixels
4-Il appelle la bibliotheque Rust qui fait le calcul
5-La bibliotheque charge le modele entraine et predit la classe
6-Le resultat revient sur le site : Avion, Voiture ou Bateau
```

- Rust — bibliotheque de calcul compilee en .dylib
- Python — preprocessing des images, serveur Flask, notebook Jupyter
- ctypes — bridge Python vers Rust
- Flask — serveur web qui expose les modeles via une API
- Pillow — chargement et redimensionnement des images

## Structure du projet

```
Projet-Annuel-Groupe-5-2025-2026/
├── ml_lib/
│   ├── src/
│   │   ├── lib.rs
│   │   └── models/
│   │       ├── mod.rs
│   │       ├── linear_model.rs
│   │       ├── pmc.rs
│   │       └── rbfn.rs
│   ├── tests/
│   │   ├── test_01_tuyauterie_rust_python.py
│   │   ├── test_02_lineaire_points_separables.py
│   │   ├── test_03_lineaire_points_non_separables.py
│   │   └── test_05_lineaire_dataset_images.py
│   ├── data/
│   │   ├── avions/     (1025 images)
│   │   ├── voitures/   (996 images)
│   │   └── bateau/     (682 images)
│   └── Cargo.toml
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── server.py
├── [Notebook] Cas de tests (2).ipynb
├── reports/
│   └── Rapport_Jalon1_Groupe5_Classification_Images.docx
└── README.md
```

## Dataset

| Classe | Images |
|---|---|
| Avions | 1025 |
| Voitures | 996 |
| Bateaux | 682 |

## Lancer le projet

```
cd ml_lib && cargo build
python3 server.py
```

Ouvrir http://127.0.0.1:5000/app

## Membres du groupe
- Messali Ayman
- Mounkala Lois-Jeremie
- Kancel Harold

## Technologies
- Rust 1.85+ bibliotheque de calcul
- Python 3.9+ pilotage et visualisation
- Flask serveur web
- ctypes Python vers Rust
- Pillow preprocessing des images
- Jupyter notebook 
```