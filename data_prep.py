import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

def extract_mel_spectrogram(file_path):
    """
    Extrait un spectrogramme Mel selon les spécifications du cours :
    16 kHz mono, fenêtres de 25ms, décalage de 10ms, 64 bandes Mel.
    """
    # Chargement à 16 kHz en mono
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    
    # Calcul des tailles de fenêtre et de saut (hop) en échantillons
    # 25 ms = 0.025 * 16000 = 400
    # 10 ms = 0.010 * 16000 = 160
    n_fft = int(sr * 0.025)
    hop_length = int(sr * 0.010)
    
    # Génération du spectrogramme Mel (64 bandes)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=64
    )
    
    # Conversion en décibels
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def load_and_split_ravdess(data_dir):
    """
    Parcourt le dossier RAVDESS, extrait les spectrogrammes et les labels,
    et sépare les données pour que les locuteurs de test soient invisibles à l'entraînement.
    """
    spectrograms = []
    labels = []
    actors = []
    
    for root, dirs, files in os.walk(data_dir):
            for file in files:
                # On vérifie que c'est un .wav et on ignore les fichiers cachés (comme les ._mac)
                if file.endswith(".wav") and not file.startswith("._"):
                    parts = file.split('.')[0].split('-')
                    
                    # RAVDESS a toujours exactement 7 nombres séparés par des tirets
                    if len(parts) == 7:
                        try:
                            emotion = int(parts[2]) - 1 # 0-indexé
                            actor_id = int(parts[6])
                            
                            file_path = os.path.join(root, file)
                            mel_db = extract_mel_spectrogram(file_path)
                            
                            max_len = 301
                            if mel_db.shape[1] > max_len:
                                mel_db = mel_db[:, :max_len]
                            else:
                                pad_width = max_len - mel_db.shape[1]
                                mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
                            
                            spectrograms.append(mel_db)
                            labels.append(emotion)
                            actors.append(actor_id)
                        except ValueError:
                            # Si la conversion en int échoue quand même, on passe au fichier suivant
                            continue
                    
    # --- LA SÉPARATION INDÉPENDANTE DU LOCUTEUR ---
    # On isole par exemple les acteurs 20 à 24 pour le test
    test_actors = [20, 21, 22, 23, 24]
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    for i in range(len(actors)):
        if actors[i] in test_actors:
            X_test.append(spectrograms[i])
            y_test.append(labels[i])
        else:
            X_train.append(spectrograms[i])
            y_train.append(labels[i])
            
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
if __name__ == "__main__":
    dossier_donnees = "./" 
    
    print("Extraction des spectrogrammes en cours...")
    X_train, X_test, y_train, y_test = load_and_split_ravdess(dossier_donnees)
    
    print(f"Dimensions de X_train : {X_train.shape}")
    print(f"Dimensions de X_test : {X_test.shape}")