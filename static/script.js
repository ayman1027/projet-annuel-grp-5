function afficherNom() {
    const fichier = document.getElementById("image").files[0];
    document.getElementById("nom-fichier").textContent = fichier ? fichier.name : "Aucun fichier choisi";
}

async function classifier() {
    const fichier = document.getElementById("image").files[0];
    const modele = document.getElementById("modele").value;

    if (!fichier) {
        alert("Selectionne une image d abord");
        return;
    }

    const formData = new FormData();
    formData.append("image", fichier);

    const reponse = await fetch(`/predict/${modele}`, {
        method: "POST",
        body: formData
    });

    const data = await reponse.json();
    const scores = data.scores;

    document.getElementById("apercu").src = URL.createObjectURL(fichier);
    document.getElementById("prediction").textContent = data.prediction.toUpperCase();

    document.getElementById("val-avion").textContent = scores.avion.toFixed(3);
    document.getElementById("val-voiture").textContent = scores.voiture.toFixed(3);
    document.getElementById("val-bateau").textContent = scores.bateau.toFixed(3);

    document.getElementById("score-avion").classList.remove("gagnant");
    document.getElementById("score-voiture").classList.remove("gagnant");
    document.getElementById("score-bateau").classList.remove("gagnant");
    document.getElementById("score-" + data.prediction).classList.add("gagnant");

    document.getElementById("resultat").style.display = "block";
}