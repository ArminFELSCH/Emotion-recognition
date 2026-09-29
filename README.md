# Projet RO11 - Speech Emotion Recognition (SER)

**🚀 Tester l'application en direct : \[Insérez ici le lien web de votre serveur ou de Google Colab\]**
*(Note : Si le lien de démonstration a expiré au moment de la correction, il vous suffit de cloner ce dépôt et d'exécuter `app.py` pour lancer l'interface en local).*

## Description du Projet

Ce dépôt contient le livrable technique de notre mini-projet de reconnaissance des émotions par la voix. L'objectif est de classifier des enregistrements audio parmi 8 émotions distinctes (Neutre, Calme, Heureux, Triste, Colère, Peur, Dégoût, Surprise) à partir de leurs spectrogrammes de Mel. L'application finale propose une interface web interactive développée avec Gradio.

## Jeu de Données et Prétraitement

Le projet exploite la base de données audio **RAVDESS**. Le pipeline de prétraitement (`data_prep.py`) :

1. Extrait les spectrogrammes de Mel (64 bandes, fenêtres de 25ms).

2. Harmonise la longueur des séquences avec un padding ou un rognage ciblant 301 frames (environ 3 secondes d'audio).

## Architectures & Analyse Critique

Afin de trouver la meilleure approche, nous avons entraîné et comparé trois architectures distinctes :

1. **Modèle 1 (CNN from scratch) - Le compromis :**
   Architecture légère composée de 4 blocs convolutifs (avec Batch Normalization et Max-Pooling). Ce modèle a atteint **50.3%** de précision sur le jeu de validation. Bien qu'il montre une bonne capacité à analyser le spectrogramme comme une image globale, il souffre d'un fort surapprentissage (99% en entraînement). En conditions réelles (micro PC, bruit de fond), il subit un "domain shift" important par rapport aux voix studio du dataset RAVDESS.

2. **Modèle 2 (Transfer Learning ResNet50V2) - La robustesse :**
   Utilisation d'un modèle pré-entraîné sur ImageNet. Les spectrogrammes sont dupliqués sur 3 canaux pour simuler une image RGB. La tête de classification a été ré-entraînée. Cette approche permet généralement une meilleure généralisation face à des voix inédites.

3. **Modèle 3 (LSTM Temporel) - La limite récurrente :**
   Au lieu de voir l'audio comme une image, ce réseau lit le spectrogramme chronologiquement (pas de temps par pas de temps). L'entraînement s'est arrêté prématurément (Early Stopping à l'époque 9) avec une précision plafonnant à **21%**. Cela prouve expérimentalement que traiter les tranches de temps de manière isolée capte beaucoup moins bien les motifs émotionnels globaux d'une voix qu'une convolution 2D.

## Structure du Dépôt

* `data_prep.py` : Extraction et formatage des données audio.

* `train_model1.py`, `train_model2.py`, `train_model3.py` : Scripts d'entraînement des différentes approches.

* `app.py` : Interface utilisateur web (Gradio) permettant de comparer les 3 modèles en direct via microphone. (Utilisation de l'injection de poids pour garantir la compatibilité des environnements).

* `modele1_scratch.h5`, `modele2_transfer.h5`, `modele3_lstm.h5` : Poids des modèles.

* `requirements.txt` : Liste des dépendances Python.

## Lancement en Local

Pour exécuter l'interface web sur votre propre machine :

```
pip install -r requirements.txt
python app.py

```

Ouvrez ensuite l'adresse `http://127.0.0.1:7860` dans votre navigateur.

## Équipe et Répartition du Travail

* **Armin Felsch :** \[À compléter : ex. Développement de l'interface Gradio multi-modèles, conception du CNN et du LSTM...\]

* **\[Nom du partenaire\] :** \[À compléter : ex. Prétraitement des données, gestion du Transfer Learning sur serveur GPU...\]