import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from data_prep import load_and_split_ravdess

#Récupération des données
dossier_donnees = "./" 
X_train, X_test, y_train, y_test = load_and_split_ravdess(dossier_donnees)

#mise à la bonne dimension
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("Construction modèle CNN")

#Architecture du réseau
def build_model_1(input_shape=(64, 301, 1), num_classes=8):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    #4 blocs de convolutions : 32 -> 64 -> 128 -> 256
    for filters in [32, 64, 128, 256]:
        # 1er conv 3x3 + BN + ReLU
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # 2eme conv 3x3 + BN + ReLU
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Max-pooling 2x2
        x = layers.MaxPooling2D((2, 2))(x)
        
    # Global average pooling -> dropout -> fully connected layer
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    # 8 classes pour RAVDESS (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

model_1 = build_model_1()

# compilation et training
model_1.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

print("Lancement du training")
history = model_1.fit(
    X_train, y_train, 
    epochs=30, 
    batch_size=32, 
    validation_data=(X_test, y_test)
)

# save du modèle
model_1.save("modele1_scratch.h5")
print("Modèle sauvegardé sous 'modele1_scratch.h5'")