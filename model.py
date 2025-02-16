#############################################################################
# YOUR ANONYMIZATION MODEL
# ---------------------
# Should be implemented in the 'anonymize' function
# !! DO NOT MODIFY THE NAME OF THE FUNCTION !!
#
# If you trained a machine learning model you can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY <!>
############################################################################

def anonymize(input_audio_path): # <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
    """
    anonymization algorithm

    Parameters
    ----------
    input_audio_path : str
        path to the source audio file in one ".wav" format.

    Returns
    -------
    audio : numpy.ndarray, shape (samples,), dtype=np.float32
        The anonymized audio signal as a 1D NumPy array of type np.float32, 
        which ensures compatibility with soundfile.write().
    sr : int
        The sample rate of the processed audio.
    """

    # Read the source audio file

    # Apply your anonymization algorithm
    
    # Output:
    import librosa
    import pyworld as pw
    import noisereduce as nr
    import numpy as np

    # Load parameters from parameters/params.json
    import json
    with open('parameters/params.json', 'r') as f:
        params = json.load(f)

    def detect_noise(y, sr):
        """Automatically detect noise using spectral analysis"""
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        S = np.abs(librosa.stft(y))
        freq_ratio = np.mean(S[30:,:]) / (np.mean(S[:30,:]) + 1e-9)
        return flatness > 0.8 and freq_ratio > 0.3


    def adaptive_denoise(y, sr):
        """Two-stage adaptive noise reduction"""
        y_clean = nr.reduce_noise(y, sr, stationary=True, prop_decrease=0.6)
        if detect_noise(y_clean, sr):
            return nr.reduce_noise(y_clean, sr, stationary=False, prop_decrease=0.8)
        return y_clean

    # Read the source audio file
    y_orig, sr = librosa.load(input_audio_path, sr=16000, mono=True)

    # Detect noise and denoise if necessary
    is_noisy = detect_noise(y_orig, sr)
    y_denoised = adaptive_denoise(y_orig, sr) if is_noisy else y_orig.copy()

    # Anonymize using WORLD vocoder
    params_set = params['noisy'] if is_noisy else params['clean']
    f0, sp, ap = pw.wav2world(y_denoised.astype(np.float64), sr)

    # Spectral modification
    sp_shifted = np.zeros_like(sp)
    max_idx = sp.shape[1] - 1
    for i in range(sp.shape[1]):
        idx = min(int(i * params_set['mcadams']), max_idx)
        sp_shifted[:, i] = sp[:, idx]

    # Pitch shifting
    f0_shifted = np.clip(f0 * params_set['pitch'], 50, 500)

    # Synthesis
    y_anon = pw.synthesize(f0_shifted, sp_shifted, ap, sr)
    y_anon += np.random.normal(0, params_set['noise'], len(y_anon))

    # Output:
    audio = y_anon.astype(np.float32)
    sr = sr
    
    return audio, sr