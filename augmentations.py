import json
import audiomentations as AA
import torch

# Load the JSON configuration file
with open("augmentations_config.json", "r") as config_file:
    config = json.load(config_file)['augmentations']

# Map of augmentation functions to their instantiation
augmentation_functions = {
    "add_background_noise": lambda params: AA.AddBackgroundNoise(**params),
    "add_color_noise": lambda params: AA.AddColorNoise(**params),
    "add_gaussian_noise": lambda params: AA.AddGaussianNoise(**params),
    "add_gaussian_snr": lambda params: AA.AddGaussianSNR(**params),
    "air_absorption": lambda params: AA.AirAbsorption(**params),
    "apply_impulse_response": lambda params: AA.ApplyImpulseResponse(**params),
    "band_pass_filter": lambda params: AA.BandPassFilter(**params),
    "band_stop_filter": lambda params: AA.BandStopFilter(**params),
    "bit_crush": lambda params: AA.BitCrush(**params),
    "high_pass_filter": lambda params: AA.HighPassFilter(**params),
    "high_shelf_filter": lambda params: AA.HighShelfFilter(**params),
    "limiter": lambda params: AA.Limiter(**params),
    "loudness_normalization": lambda params: AA.LoudnessNormalization(**params),
    "low_pass_filter": lambda params: AA.LowPassFilter(**params),
    "low_shelf_filter": lambda params: AA.LowShelfFilter(**params),
    "mp3_compression": lambda params: AA.Mp3Compression(**params),
    "pitch_shift": lambda params: AA.PitchShift(**params),
    "resample": lambda params: AA.Resample(**params),
    "room_simulator": lambda params: AA.RoomSimulator(**params),
    "time_mask": lambda params: AA.TimeMask(**params),
    "time_stretch": lambda params: AA.TimeStretch(**params),
}

def get_augmentation(augmentation):
    augmentation_params = config[augmentation]
    return augmentation_functions[augmentation](augmentation_params)

def get_mix_augmentation(augmentation_list):
    return AA.Compose([get_augmentation(aug) for aug in augmentation_list])


def apply_augmentation(waveform, augmentation):
    waveform_aug = augmentation(samples=waveform, sample_rate=16000)
    return torch.from_numpy(waveform_aug)