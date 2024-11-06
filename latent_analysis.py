import torch
import torchaudio
import librosa
import json
from tqdm import tqdm
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import WhisperProcessor, WhisperModel, Wav2Vec2Model, Wav2Vec2Processor
from hezar.models import Model
from augmentations import *

# outline
# 1. load and resample audio
# 2. get input features
# 3. get latent representation
# 4. 
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

def get_input_features(audio_array, processor, device):
    input_features = processor(audio_array, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt")
    if 'input_features' in input_features:
        input_features = input_features.input_features
    elif 'input_values' in input_features:
        input_features = input_features.input_values
    else:
        raise ValueError("Unkown processor type")
    return input_features.to(device)

def get_latent_representation(model, input_features):
    with torch.no_grad():
        latent = model(input_features).last_hidden_state
    return latent.mean(dim=1).squeeze().cpu().numpy()

def get_model(model_name, device):
    assert model_name in MODELS
    if 'wav2vec2' in model_name:
        processor = Wav2Vec2Processor.from_pretrained(MODELS[model_name])
        model = Wav2Vec2Model.from_pretrained(MODELS[model_name]).eval().to(device)
    elif model_name == 'whisper-tiny':
        processor = WhisperProcessor.from_pretrained(MODELS[model_name])
        model = WhisperModel.from_pretrained(MODELS[model_name]).eval().get_encoder().to(device)
    elif model_name == 'hezarai':
        processor = WhisperProcessor.from_pretrained(MODELS['whisper-tiny'])
        model = Model.load(MODELS[model_name]).whisper.model.eval().get_encoder().to(device)
    return model, processor

def collect_latents(audio_files, dataframe, model, processor, device, model_name, augmentation_type='original'):
    for i, audio_file in tqdm(enumerate(audio_files)):
        audio_array = load_and_resample_audio(audio_file, processor)
        if augmentation_type != 'original':
            augmentation = get_augmentation(augmentation_type)
            audio_array = apply_augmentation(audio_array, augmentation)
        input_features = get_input_features(audio_array, processor, device)
        latent = get_latent_representation(model, input_features)
        sample = {'audio_file': audio_file, 'model_name': model_name, 'type': augmentation_type, 'latent_representation': latent, 'latent_dim': latent.shape[0]}
        dataframe.loc[i] = sample
    return dataframe

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model, processor = get_model('whisper-tiny', device)
# df = get_dataframe()
# df = collect_latents(['cvfa/cvfa_8.wav'], df, model, processor, device, 'whisper-tiny')
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
    df = collect_latents(configs['audio_directory'], df, model, processor, device, configs['model_name'], configs['augmentation_type'])

    # save dataframe
    df.to_csv(configs['output']['dataframe_path'], index=False)