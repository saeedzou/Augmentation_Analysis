# Outline:
# 1. get_augmentation(augmentation) function to define augmentations
# 2. get_mix_augmentation(augmentation_list) function to define augmentation combination
# 3. apply_augmentation(waveform, augmentation) function to apply augmentations
# 4. get_latent_representations(waveform) function to extract latent representations from Whisper's encoder
import os
import audiomentations as AA
import torch


def get_augmentation(augmentation):
    # Define augmentations
    augmentations = {
        "add_background_noise": AA.AddBackgroundNoise(sounds_path="/content/ESC-50-master/audio/1-103298-A-9.wav", p=1.0),
        "add_color_noise": AA.AddColorNoise(
            min_snr_db=5.0, 
            max_snr_db=40.0, 
            min_f_decay=-6.0, 
            max_f_decay=6.0, 
            p=1.0, 
            p_apply_a_weighting=0.0, 
            n_fft=128
        ),
        "add_gaussian_noise": AA.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
        "add_gaussian_snr": AA.AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=20, p=1.0),
        "air_absorption": AA.AirAbsorption(p=1.0),
        "apply_impulse_response": AA.ApplyImpulseResponse(ir_path="/content/ir", p=1.0),
        "band_pass_filter": AA.BandPassFilter(p=1.0),
        "band_stop_filter": AA.BandStopFilter(p=1.0),
        "bit_crush": AA.BitCrush(min_bit_depth=5, max_bit_depth=14, p=1.0),
        "high_pass_filter": AA.HighPassFilter(p=1.0),
        "high_shelf_filter": AA.HighShelfFilter(p=1.0),
        "limiter": AA.Limiter(p=1.0),
        "loudness_normalization": AA.LoudnessNormalization(p=1.0),
        "low_pass_filter": AA.LowPassFilter(p=1.0),
        "low_shelf_filter": AA.LowShelfFilter(p=1.0),
        "mp3_compression": AA.Mp3Compression(
            min_bitrate=8, 
            max_bitrate=64, 
            backend="pydub", 
            p=1.0
        ),
        "pitch_shift": AA.PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
        "resample": AA.Resample(p=1.0),
        "room_simulator": AA.RoomSimulator(p=1.0),
        "time_mask": AA.TimeMask(p=1.0),
        "time_stretch": AA.TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0),
    }
    return augmentations[augmentation]

def get_mix_augmentation(augmentation_list):
    return AA.Compose([get_augmentation(aug) for aug in augmentation_list])


def apply_augmentation(waveform, augmentation):
    waveform_aug = augmentation(samples=waveform.numpy(), sample_rate=16000)
    return torch.from_numpy(waveform_aug)