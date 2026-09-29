import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2
from data_prep import load_and_split_ravdess

dossier_donnees = "./" 
X_train, X_test, y_train, y_test = load_and_split_ravdess(dossier_donnees)

# Copie sur 3 canaux pour faire une image RGB pour Resnet
X_train_rgb = np.repeat(X_train[..., np.newaxis], 3, axis=-1)
X_test_rgb = np.repeat(X_test[..., np.newaxis], 3, axis=-1)

print("Construction du Modèle 2 (Transfer Learning)...")

# Backbone pré-entraîné (ImageNet), sans la tête de classification
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(64, 301, 3))
base_model.trainable = False  # On gèle les poids existants

# Ajout la tête fully connected
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(8, activation='softmax')(x)

model_2 = models.Model(inputs=base_model.input, outputs=outputs)

model_2.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

print("Lancement de l'entraînement...")
model_2.fit(
    X_train_rgb, y_train, 
    epochs=15, 
    batch_size=32, 
    validation_data=(X_test_rgb, y_test)
)
# save du modèle
model_2.save("modele2_transfer.h5")
print("Modèle 2 sauvegardé.")