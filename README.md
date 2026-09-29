# RO11 Project - Speech Emotion Recognition (SER)

## Project Description

This repository contains the technical deliverable for our Speech Emotion Recognition (SER) TD. The objective is to classify audio recordings into 8 distinct emotions (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised) using their Mel spectrograms. The final application features an interactive web interface developed with Gradio.

## Dataset and Preprocessing

The project utilizes the **RAVDESS** audio dataset. The preprocessing pipeline (`data_prep.py`):

1. Extracts Mel spectrograms (64 frequency bands, 25ms windows).
2. Standardizes sequence lengths using padding or cropping to reach exactly 301 frames (approximately 3 seconds of audio).

## Architectures & Critical Analysis

To determine the most effective approach, we trained and compared three distinct neural network architectures:

1. **Model 1 (CNN from scratch) - The Compromise:**
   A custom lightweight architecture consisting of 4 convolutional blocks (with Batch Normalization and Max-Pooling). This model achieved **50.3%** accuracy on the validation set. While it demonstrates a good ability to process the spectrogram as a global 2D image, it suffers from heavy overfitting (99% training accuracy). In real-world conditions (standard PC microphone, background noise), it experiences significant "domain shift" compared to the pristine studio voices of the RAVDESS dataset.

2. **Model 2 (Transfer Learning ResNet50V2) - The Robust Approach:**
   This approach leverages a model pre-trained on ImageNet. The 1D spectrograms are duplicated across 3 channels to simulate an RGB image format. The classification head was fully retrained. This methodology generally yields better generalization when encountering unseen, real-world voices.

3. **Model 3 (Temporal LSTM) - The Recurrent Limit:**
   Instead of treating the audio as a static image, this network reads the spectrogram chronologically (time step by time step). The training phase stopped prematurely (Early Stopping triggered at epoch 9), with accuracy plateauing at **21%**. This experimentally demonstrates that processing time slices in isolation captures global emotional vocal patterns much less effectively than 2D spatial convolutions.

## Repository Structure

* `data_prep.py`: Audio data extraction and formatting pipeline.
* `train_model1.py`, `train_model2.py`, `train_model3.py`: Training scripts for the three respective architectures.
* `app.py`: Web user interface (Gradio) allowing real-time comparison of the 3 models via microphone input. *(Note: We use architecture reconstruction and weight injection to ensure cross-environment compatibility without legacy Keras errors).*
* `modele1_scratch.h5`, `modele2_transfer.h5`, `modele3_lstm.h5`: Saved model weights.
* `requirements.txt`: Python dependencies required for the project.

## Local Execution

To run the interactive web interface on your local machine, clone this repository and execute the following commands:

```bash
pip install -r requirements.txt
python app.py
```

Once the script is running, open `http://127.0.0.1:7860` or `htttp://"your IPv4:7860` in your web browser.

## Team and Work Breakdown

* **Armin Felsch:** [To be completed: e.g., Multi-model Gradio interface development, CNN and LSTM architecture design...]
* **[Partner's Name]:** [To be completed: e.g., Data preprocessing pipeline, Transfer Learning implementation on GPU server...]
