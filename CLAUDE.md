# CLAUDE.md

Guide pour travailler sur ce dépôt. Projet Annuel IABD — Groupe 5 (2025-2026).

## Objectif du projet

Classifier automatiquement des images en **3 classes : avion / voiture / bateau**,
en utilisant des algorithmes de Machine Learning **implémentés from scratch** (aucune
bibliothèque ML externe). Le cœur de calcul est écrit en **Rust**, piloté depuis
**Python** via `ctypes`, et exposé par une petite **API Flask** avec interface web.

Membres : Messali Ayman, Mounkala Loïs-Jérémie, Kancel Harold.

## Architecture

```
Navigateur (templates/index.html + static/)
        │  HTTP (multipart image)
        ▼
server.py  (Flask, port 5000)
        │  ctypes
        ▼
ml_lib/target/debug/libml_lib.dylib   ← bibliothèque Rust compilée (cdylib)
        │  lit / écrit
        ▼
*.bin  (poids des modèles entraînés, à la racine du dépôt)
```

- **Rust** (`ml_lib/`) = tous les calculs (forward, backprop, save/load). Compilé en
  `cdylib`, chaque fonction exposée est `#[unsafe(no_mangle)] extern "C"`.
- **Python** = orchestration : preprocessing image, appel des fonctions Rust,
  entraînement (notebook), et serveur web (`server.py`).
- Les modèles entraînés sont sérialisés en fichiers `.bin` (voir format ci-dessous).

## Structure des fichiers

| Chemin | Rôle |
|---|---|
| `ml_lib/src/lib.rs` | Point d'entrée : déclare les modules, ré-exporte, fonctions de test (`addition`, `soustraction`) |
| `ml_lib/src/models/mod.rs` | Déclare `linear_model`, `pmc`, `rbfn` |
| `ml_lib/src/models/linear_model.rs` | Modèle linéaire (descente de gradient) |
| `ml_lib/src/models/pmc.rs` | Perceptron Multi-Couches (1 couche cachée, sigmoïde, rétropropagation) |
| `ml_lib/src/models/rbfn.rs` | Radial Basis Function Network |
| `ml_lib/preprocessing.py` | Chargement dataset RGB 32×32 normalisé (`charger_dataset`) |
| `ml_lib/tests/test_0*.py` | Tests / cas de démonstration du modèle linéaire (01, 02, 03, 05) |
| `ml_lib/data/{avions,voitures,bateau}/` | Dataset images (⚠️ **git-ignoré**, ~1025/996/682 images en local) |
| `server.py` | API Flask + inférence 3 classes en OVR |
| `templates/index.html`, `static/script.js`, `static/style.css` | Interface web (upload image → prédiction) |
| `[Notebook] Cas de tests (2).ipynb` | Entraînement de tous les modèles + cas de tests (84 cellules) |
| `*.bin` (racine) | Modèles entraînés sérialisés |
| `reports/` | Rapport de jalon (.docx) |

## Modèles Rust — signatures FFI

Toutes exposées en `extern "C"`. Les pointeurs opaques (`*mut PMC`, etc.) sont
transportés côté Python comme `c_void_p`. **Chaque `create_*`/`load_*` doit être
libéré par le `free_*` correspondant.**

**Modèle linéaire** (`x` = tableau de features, 1 sortie, ±) :
`create_linear_model(n, lr)`, `train_step(m, x, n, y)`, `predict(m, x, n) -> f32`,
`save_linear_model(m, path)`, `load_linear_model(path)`, `free_linear_model(m)`.

**PMC** (input_size / hidden_size / output_size ; sigmoïde en cachée, **sortie linéaire**) :
`create_pmc(input, hidden, output, lr)`, `pmc_train_step(m, x, n, y, ny)`,
`pmc_predict(m, x, n, out_ptr)`, `save_pmc`, `load_pmc`, `free_pmc`.

**RBFN** (noyau gaussien, distance **normalisée par input_size**, sortie linéaire) :
`create_rbfn(input, hidden, output, gamma, lr)`, `rbfn_set_centers(m, centers, n)`,
`rbfn_train_step(m, x, n, y, ny)`, `rbfn_predict(m, x, n, out_ptr)`, `save_rbfn`,
`load_rbfn`, `free_rbfn`. Seuls les **poids** sont appris ; les **centres** sont fixés
(via `rbfn_set_centers`, généralement des points du dataset).

## Format des fichiers `.bin`

Sérialisation maison : un `Vec<f32>` écrit en **little-endian**, précédé d'un en-tête.

- **Linéaire** : `[n, bias, lr, w0..w(n-1)]`
- **PMC** : `[input, hidden, output, lr, w1(input×hidden), w2(hidden×output), b1(hidden), b2(output)]`
- **RBFN** : `[input, hidden, output, gamma, lr, centers(hidden×input), weights(hidden×output)]`

Modèles présents (vérifiés) :
- `pmc_{avion,voiture,bateau}.bin` : input=1024 (gris 32×32), hidden=128, output=1 — **modèles servis en prod**
- `rbfn_{avion,voiture,bateau}.bin` : input=1024, hidden=100, output=1, gamma=1.0
- `pmc_xor.bin`, `rbfn_xor.bin` : démos XOR (input=2)
- `model_lineaire.bin` : input=3072 (RGB 32×32), démo
- `pmc_images.bin` : ancien PMC RGB (input=3072, hidden=64) — non utilisé par le serveur

## Inférence 3 classes (One-vs-Rest)

Il n'y a **pas** un modèle à 3 sorties en prod. Le serveur utilise **3 modèles binaires
séparés** (un par classe), calcule un score par classe, et prend l'`argmax`.
Image d'entrée = **niveaux de gris 32×32 aplatis, normalisés /255** (1024 features).
Endpoints : `POST /predict/pmc` et `POST /predict/rbfn` (champ form `image`).

Les 6 modèles (3 PMC + 3 RBFN) sont chargés **une seule fois au démarrage** par
`charger_modeles()` et gardés en mémoire dans le dict `MODELES`. La logique OVR
commune est factorisée dans `predire(type_modele, predict_fn, pixels)`.

## Commandes

```bash
# 1. Compiler la lib Rust (À REFAIRE après toute modif de ml_lib/src/)
cd ml_lib && cargo build          # produit target/debug/libml_lib.dylib
cd ..

# 2. Lancer le serveur web  (DOIT être lancé depuis la racine du dépôt :
#    server.py charge les .bin par chemin relatif au CWD)
source .venv/bin/activate         # Python 3.9, dépendances : flask, numpy, pillow
python server.py                  # http://localhost:5000/app  (interface)
                                  # http://localhost:5000/     (health check JSON)

# 3. (Ré)entraîner les modèles
#    Ouvrir "[Notebook] Cas de tests (2).ipynb" — les cellules ~72-82 entraînent
#    les modèles images et écrivent les .bin à la racine.

# Tests / démos individuels (depuis ml_lib/) :
python tests/test_02_lineaire_points_separables.py
```

## Conventions

- Commentaires et noms de variables **en français**. Garder ce style.
- Côté Rust : tout ce qui est exposé à Python est `#[unsafe(no_mangle)] extern "C"`
  et manipule des pointeurs bruts (`unsafe { std::slice::from_raw_parts(...) }`).
- Côté Python : déclarer systématiquement `argtypes`/`restype` avant d'appeler une
  fonction ctypes, et passer les tableaux via `np.ctypeslib.as_ctypes` (dtype
  **`float32`** obligatoire).
- Le RNG Rust est un xorshift maison seedé sur l'horloge (pas de crate `rand`).

## Pièges connus

- **Rebuild obligatoire** : Python charge le `.dylib` compilé, pas les `.rs`. Après
  toute modif dans `ml_lib/src/`, relancer `cargo build` sinon les changements sont
  invisibles.
- **CWD du serveur** : `server.py` fait `load_pmc("pmc_avion.bin")` en chemin relatif
  → lancer depuis la racine, sinon crash.
- **Robustesse** : les `load_*`/`save_*` Rust font `.unwrap()` partout. Un `.bin`
  manquant/corrompu ou un chemin invalide **fait paniquer** — au démarrage du serveur
  désormais (les modèles sont chargés dans `charger_modeles()`), plus par requête.
- **Modèles jamais libérés** : les 6 modèles chargés au démarrage vivent jusqu'à
  l'arrêt du process (pas de `free_*`). C'est voulu — ils sont partagés entre requêtes.
- **Dataset git-ignoré** : `ml_lib/data/` n'est pas versionné. Le README mentionne
  « 30 images / classe » mais le local en contient ~1000 ; ne pas s'y fier.
- **Cohérence couleur** : le preprocessing d'entraînement des modèles servis et le
  serveur utilisent **niveaux de gris** (`convert("L")`, 1024 features). D'anciens
  modèles/démos (`pmc_images.bin`, `model_lineaire.bin`) sont en **RGB** (3072) et
  ne sont pas compatibles avec le pipeline du serveur.

## État / roadmap

- ✅ Modèle linéaire, PMC, RBFN implémentés ; classification 3 classes en OVR.
- ✅ Interface web fonctionnelle ; PMC ~85 % annoncé sur images (cf. dernier commit).
- 🔄 Jalon 2 (Juin 2026) : SVM, dataset complet (~400/classe visé), rapport final.
