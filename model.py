#############################################################################
# YOUR ANONYMIZATION MODEL
# ---------------------
# Should be implemented in the 'anonymize' function
# !! DO NOT MODIFY THE NAME OF THE FUNCTION !!
#
# If you trained a machine learning model you can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> SAVE YOUR PARAMETERS IN THE parameters/ DIRECTORY <!>
#############################################################################

def anonymize(input_audio_path):  # <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
    """
    Anonymization algorithm

    Parameters
    ----------
    input_audio_path : str
        Path to the source audio file in ".wav" format.

    Returns
    -------
    audio : numpy.ndarray, shape (samples,), dtype=np.float32
        The anonymized audio signal as a 1D NumPy array (compatible with soundfile.write()).
    sr : int
        The sample rate of the processed audio.
    """
    
    import librosa
    import pyworld as pw
    import noisereduce as nr
    import numpy as np
    import json

    
    with open('parameters/params.json', 'r') as f:
        params = json.load(f)

    def detect_noise(y, sr):
        """Automatically detect noise using spectral analysis."""
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        S = np.abs(librosa.stft(y))
        freq_ratio = np.mean(S[30:, :]) / (np.mean(S[:30, :]) + 1e-9)
        return flatness > 0.8 or freq_ratio > 0.3

    def adaptive_denoise(y, sr):
        """Two-stage adaptive noise reduction."""
        y_clean = nr.reduce_noise(y, sr, stationary=True, prop_decrease=0.6)
        if detect_noise(y_clean, sr):
            return nr.reduce_noise(y_clean, sr, stationary=False, prop_decrease=0.8)
        return y_clean

    # Read the source audio file
    y_orig, sr = librosa.load(input_audio_path, sr=16000, mono=True)

    # Detect noise and apply adaptive denoising if needed
    is_noisy = detect_noise(y_orig, sr)
    y_denoised = adaptive_denoise(y_orig, sr) if is_noisy else y_orig.copy()

    # Choose parameter set based on detected noise level
    params_set = params['noisy'] if is_noisy else params['clean']

    # WORLD vocoder analysis
    f0, sp, ap = pw.wav2world(y_denoised.astype(np.float64), sr)

    # Spectral modification using the McAdams coefficient
    sp_shifted = np.zeros_like(sp)
    max_idx = sp.shape[1] - 1
    for i in range(sp.shape[1]):
        idx = min(int(i * params_set['mcadams']), max_idx)
        sp_shifted[:, i] = sp[:, idx]

    # Pitch shifting (scaling the F0 values)
    f0_shifted = np.clip(f0 * params_set['pitch'], 50, 500)

    # Synthesis using the WORLD vocoder
    y_anon = pw.synthesize(f0_shifted, sp_shifted, ap, sr)

    # Noise injection: add low-level Gaussian noise to further mask speaker features
    y_anon += np.random.normal(0, params_set['noise'], len(y_anon))

    # Formant shifting via double resampling:
    # Resample to a new "virtual" rate (shifting formants), then back to the original rate
    new_sr = int(sr * params_set['formant_ratio'])
    y_anon = librosa.resample(y_anon, orig_sr=sr, target_sr=new_sr)
    y_anon = librosa.resample(y_anon, orig_sr=new_sr, target_sr=sr)

    # Final output cast to float32 for compatibility
    audio = y_anon.astype(np.float32)
    return audio, sr
