import torch
import torchaudio
import librosa
import json
import os
from tqdm import tqdm
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import WhisperProcessor, Wav2Vec2Processor, WhisperForConditionalGeneration, Wav2Vec2ForCTC
from hezar.models import Model
from augmentations import *

MODELS = {'whisper-tiny': "openai/whisper-tiny",
          'hezarai': "hezarai/whisper-small-fa",
          'wav2vec2_v3_fa': "m3hrdadfi/wav2vec2-large-xlsr-persian-v3",
          'wav2vec2_fa': "masoudmzb/wav2vec2-xlsr-multilingual-53-fa"}

def get_dataframe():
    # type can be original or augmentation name
    columns = ['audio_file', 'model_name', 'type', 'latent_representation', 'latent_dim']
    return pd.DataFrame(columns=columns)

# Helper function for resampling audio
def load_and_resample_audio(path, processor):
    speech_array, sampling_rate = torchaudio.load(path)
    speech_array = speech_array.squeeze().numpy()
    return librosa.resample(np.asarray(speech_array), orig_sr=sampling_rate, target_sr=processor.feature_extractor.sampling_rate)

def get_model(model_name, device):
    assert model_name in MODELS
    if 'wav2vec2' in model_name:
        processor = Wav2Vec2Processor.from_pretrained(MODELS[model_name])
        model = Wav2Vec2ForCTC.from_pretrained(MODELS[model_name]).eval().to(device)
    elif model_name == 'whisper-tiny':
        processor = WhisperProcessor.from_pretrained(MODELS[model_name])
        model = WhisperForConditionalGeneration.from_pretrained(MODELS[model_name]).eval().to(device)
    elif model_name == 'hezarai':
        processor = WhisperProcessor.from_pretrained(MODELS['whisper-tiny'])
        model = Model.load(MODELS[model_name]).eval().to(device)
    return model, processor

def get_encoder(model):
    model_name = model.__class__.__name__
    if model_name == 'Wav2Vec2ForCTC':
        return model.wav2vec2
    elif model_name == 'WhisperForConditionalGeneration':
        return model.model.encoder
    elif model_name == 'WhisperSpeechRecognition':
        return model.whisper.model.encoder

def get_input_features(audio_array, processor, device):
    input_features = processor(audio_array, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt")
    if 'input_features' in input_features:
        input_features = input_features.input_features
    elif 'input_values' in input_features:
        input_features = input_features.input_values
    else:
        raise ValueError("Unkown processor type")
    return input_features.to(device)

def get_latent_representation(encoder, input_features):
    with torch.no_grad():
        latent = encoder(input_features).last_hidden_state
    return latent.mean(dim=1).squeeze().cpu().numpy()

def collect_latents(audio_files, dataframe, model, processor, device, model_name, augmentation_type='original'):
    encoder = get_encoder(model)
    for i, audio_file in tqdm(enumerate(audio_files)):
        audio_array = load_and_resample_audio(audio_file, processor)
        if augmentation_type != 'original':
            augmentation = get_augmentation(augmentation_type)
            audio_array = apply_augmentation(audio_array, augmentation)
        input_features = get_input_features(audio_array, processor, device)
        latent = get_latent_representation(encoder, input_features)
        sample = {'audio_file': audio_file, 'model_name': model_name, 'type': augmentation_type, 'latent_representation': latent, 'latent_dim': latent.shape[0]}
        dataframe.loc[i] = sample
    return dataframe

def main():
    # load json
    with open('configs.json') as f:
        configs = json.load(f)
    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get model
    model, processor = get_model(configs['model_name'], device)
    # get dataframe
    df = get_dataframe()
    # collect latents
    audio_files = [os.path.join(configs['audio_directory'], f) for f in os.listdir(configs['audio_directory']) if f.endswith(".wav")]
    df = collect_latents(audio_files, df, model, processor, device, configs['model_name'], configs['augmentation_type'])

    # save dataframe
    if configs['output']['save_dataframe']:
        df.to_csv(configs['output']['dataframe_path'], index=False)

if __name__ == '__main__':
    main()