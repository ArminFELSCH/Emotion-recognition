import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from data_prep import load_and_split_ravdess

print("Chargement des données...")
dossier_donnees = "./" 
X_train, X_test, y_train, y_test = load_and_split_ravdess(dossier_donnees)

print("Construction du modèle Réseau Récurrent LSTM")
def build_lstm(input_shape=(64, 301, 1), num_classes=8):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Reshape((64, 301))(inputs)
    
    x = layers.Permute((2, 1))(x)
    
    #Couches LSTM pour lire l'audio chronologiquement
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)
    
    # Tête de classification
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

model_3 = build_lstm()

model_3.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# pour stop quand le modèle commence à surapprendre
early_stop = EarlyStopping(
    monitor='val_accuracy', 
    patience=20, 
    restore_best_weights=True
)

print("Lancement de l'entraînement...")
model_3.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=32, 
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

model_3.save("modele3_lstm.h5")
print("Modèle 3 sauvegardé sous 'modele3_lstm.h5'.")