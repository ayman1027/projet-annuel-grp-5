# Projet Annuel IABD — Classification d'Images

## Problématique
Distinguer automatiquement 3 classes d'images :
- Avion
- Voiture  
- Bateau

À partir d'une photo, notre application prédit automatiquement la classe de l'image grâce à des modèles de Machine Learning implémentés from scratch.

## Architecture technique

```
Python (pilotage + affichage)
↕ ctypes
Rust (calculs mathématiques — libml_lib.dylib)
```

- **Rust** → bibliothèque de calcul compilée en `.dylib`
- **Python** → preprocessing des images, appel de la lib, affichage
- **Matplotlib** → visualisation des résultats

## Structure du projet

```
Projet-Annuel-Groupe-5-2025-2026/
├── ml_lib/
│   ├── src/
│   │   ├── lib.rs
│   │   └── models/
│   │       ├── mod.rs
│   │       ├── linear_model.rs
│   │       └── pmc.rs
│   ├── tests/
│   │   ├── test_01_tuyauterie_rust_python.py
│   │   ├── test_02_lineaire_points_separables.py
│   │   ├── test_03_lineaire_points_non_separables.py
│   │   ├── test_04_pmc_points_non_separables.py
│   │   ├── test_05_lineaire_dataset_images.py
│   │   └── test_06_pmc_dataset_images.py
│   ├── data/
│   │   ├── avions/     (30 images)
│   │   ├── voitures/   (30 images)
│   │   └── bateau/     (30 images)
│   ├── preprocessing.py
│   └── Cargo.toml
├── reports/
│   └── Rapport_Jalon1_Groupe5_Classification_Images.docx
└── README.md
```

## Algorithmes implémentés

| Algorithme | Statut | Description |
|---|---|---|
| Modèle Linéaire | ✅ Fait | Descente de gradient, frontière linéaire |
| PMC | ✅ Fait | Rétropropagation, frontière non-linéaire |
| RBFN | 🔄 En cours | Prévu pour Juin |
| SVM | 🔄 En cours | Prévu pour Juin |

---

## Resultats obtenus (Jalon 1 — 06/04/2026)

### Cas 1 — Points lineairement separables
- Modele Lineaire : 100% de precision
- La droite de separation est trouvee parfaitement

### Cas 2 — Points NON lineairement separables
- Modele Lineaire : 50% de precision (equivalent au hasard)
- PMC : 100% de precision
- Demontre la limite du modele lineaire face aux problemes non-lineaires

### Cas 3 — Application sur dataset reel (avion vs voiture)
- Modele Lineaire : 0% de precision
- PMC : 75% de precision (9 images sur 12 correctement classees)
- Confirme que le PMC generalise mieux sur des images complexes

## Dataset

| Classe   | Images actuelles | Objectif final |
|----------|-----------------|----------------|
| Avions   | 30              | 400            |
| Voitures | 30              | 400            |
| Bateaux  | 30              | 400            |

Repartition : 80% entrainement / 20% test
Preprocessing : redimensionnement 32x32 pixels, normalisation entre 0 et 1

## Roadmap

| Jalon | Date | Contenu |
|---|---|---|
| ✅ Jalon 1 | 06/04/2026 | Modèle Linéaire + PMC + tuyauterie + début dataset |
| 🔄 Jalon 2 | Juin 2026 | RBFN + SVM + dataset complet + API + rapport final |

## Membres du groupe
- Messali Ayman
- MOUNKALA Loïs-Jérémie 
- KANCEL Harold

## Technologies
- **Rust** 1.85+ — bibliothèque de calcul
- **Python** 3.9+ — pilotage et visualisation
- **Matplotlib** — affichage des résultats
- **ctypes** — bridge Python ↔ Rust
- **Pillow**: chargement et preprocessing des images