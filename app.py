import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from data_prep import extract_mel_spectrogram
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2

print("Chargement des architectures et des poids...")

# --- Modèle 1 : CNN Scratch ---
def build_model_1(input_shape=(64, 301, 1), num_classes=8):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

model_1 = build_model_1()
model_1.load_weights("modele1_scratch.h5")

# --- Modèle 2 : Transfer Learning (ResNet) ---
def build_model_2(input_shape=(64, 301, 3), num_classes=8):
    base_model = ResNet50V2(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=base_model.input, outputs=outputs)

model_2 = build_model_2()
model_2.load_weights("modele2_transfer.h5")

# --- Modèle 3 : LSTM (Temporel) ---
def build_lstm(input_shape=(64, 301, 1), num_classes=8):
    inputs = layers.Input(shape=input_shape)
    x = layers.Reshape((64, 301))(inputs)
    x = layers.Permute((2, 1))(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

model_3 = build_lstm()
model_3.load_weights("modele3_lstm.h5")

print("Les 3 modèles sont prêts !")

emotions = ["Neutre", "Calme", "Heureux", "Triste", "En colère", "Peureux", "Dégoûté", "Surpris"]

def process_audio(audio_path, model_choice):
    if audio_path is None:
        return None, "Veuillez enregistrer un audio."
        
    mel_spec = extract_mel_spectrogram(audio_path)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='magma')
    plt.axis('off')
    plt.tight_layout(pad=0)
    img_path = "spectrogram.png"
    plt.savefig(img_path, bbox_inches='tight', transparent=True)
    plt.close()
    
    max_len = 301
    if mel_spec.shape[1] > max_len:
        mel_spec = mel_spec[:, :max_len]
    else:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, max_len - mel_spec.shape[1])))
    
    spec_input = mel_spec[np.newaxis, ..., np.newaxis]
    
    if "CNN Scratch" in model_choice:
        preds = model_1.predict(spec_input)[0]
    elif "Transfer Learning" in model_choice:
        spec_input_rgb = np.repeat(spec_input, 3, axis=-1)
        preds = model_2.predict(spec_input_rgb)[0]
    else:
        preds = model_3.predict(spec_input)[0]
        
    best_idx = np.argmax(preds)
    result = f"{emotions[best_idx]} (Confiance : {preds[best_idx]*100:.1f}%)"
    
    return img_path, result

demo = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Parlez ici, max 3 sec (crop avec le ciseau si nécessaire)"),
        gr.Radio(
            ["Modèle 1 (CNN Scratch)", "Modèle 2 (Transfer Learning)", "Modèle 3 (LSTM Temporel)"], 
            value="Modèle 3 (LSTM Temporel)", 
            label="Modèle d'estimation."
        )
    ],
    outputs=[
        gr.Image(label="Spectrogramme généré"),
        gr.Textbox(label="Émotion détectée")
    ],
    title="Comparaison d'algorithmes pour la détection d'émotion dans la voix.",
    description="Enregistrez votre voix et observez comment un réseau CNN classique, un modèle pré-entraîné (ResNet) et un réseau récurrent (LSTM) interprètent vos émotions."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)