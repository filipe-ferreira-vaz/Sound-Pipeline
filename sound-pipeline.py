import os
import sys
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from scipy.signal import resample_poly
from pathlib import Path
from itertools import product, combinations
import warnings

def calculate_lufs_targets(L_target, SNR, N):
    """
    Calculate LUFS targets for foreground and background tracks.
    
    Parameters:
    -----------
    L_target : float
        Target final mix loudness in LUFS
    SNR : float
        Target Signal-to-Noise Ratio in dB
    N : int
        Number of foreground tracks
    
    Returns:
    --------
    tuple : (L_fg, L_bg)
        LUFS targets for individual foreground and background tracks
    """
    # Total foreground loudness
    L_fg_total = L_target - 10 * np.log10(1 + 10**(-SNR/10))
    
    # Individual foreground track loudness
    L_fg = L_fg_total - 10 * np.log10(N)
    
    # Background track loudness
    L_bg = L_fg_total - SNR
    
    return L_fg, L_bg

def ensure_mono(audio):
    """
    Convert stereo audio to mono by averaging channels.
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio signal (1D for mono, 2D for stereo)
    
    Returns:
    --------
    numpy.ndarray : Mono audio (1D)
    """
    if len(audio.shape) == 2:
        # Stereo: average channels
        return np.mean(audio, axis=1)
    return audio

def normalize_to_lufs(audio, sr, target_lufs):
    """
    Normalize audio to target LUFS using pyloudnorm.
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio signal (mono)
    sr : int
        Sample rate
    target_lufs : float
        Target LUFS value
    
    Returns:
    --------
    numpy.ndarray : Normalized audio
    """
    # Ensure mono for LUFS calculation
    audio_mono = ensure_mono(audio)
    
    # Create meter
    meter = pyln.Meter(sr)
    
    # Measure current loudness
    current_lufs = meter.integrated_loudness(audio_mono)
    
    # Calculate gain adjustment
    gain_db = target_lufs - current_lufs
    gain_linear = 10**(gain_db / 20)
    
    # Apply gain to all channels
    if len(audio.shape) == 2:
        # Stereo
        normalized = audio * gain_linear
    else:
        # Mono
        normalized = audio * gain_linear
    
    # Prevent clipping
    max_val = np.max(np.abs(normalized))
    if max_val > 1.0:
        normalized = normalized / max_val * 0.99
    
    return normalized

def trim_to_length(audio, target_length_samples):
    """
    Trim audio to target length by removing equal amounts from both sides.
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio signal
    target_length_samples : int
        Desired length in samples
    
    Returns:
    --------
    numpy.ndarray : Trimmed audio
    """
    current_length = len(audio)
    
    if current_length <= target_length_samples:
        return audio
    
    # Calculate samples to remove from each side
    excess = current_length - target_length_samples
    remove_start = excess // 2
    remove_end = excess - remove_start
    
    return audio[remove_start:current_length - remove_end]

def apply_true_peak_ceiling(audio, sr, ceiling_db=-1.0):
    """
    Apply a true-peak ceiling of ceiling_db dBFS to the audio.

    Inter-sample peaks are detected by 4x oversampling (per ITU-R BS.1770);
    if the true peak exceeds the ceiling, the whole signal is scaled down
    so the true peak sits exactly at the ceiling.

    Parameters:
    -----------
    audio : numpy.ndarray
        Audio signal (1D mono or 2D with channels on axis 1)
    sr : int
        Sample rate
    ceiling_db : float
        True-peak ceiling in dBFS

    Returns:
    --------
    numpy.ndarray : Audio with true peak <= ceiling
    """
    ceiling_linear = 10**(ceiling_db / 20)

    # 4x oversample to estimate the inter-sample (true) peak
    oversampled = resample_poly(audio, 4, 1, axis=0)
    true_peak = np.max(np.abs(oversampled))

    if true_peak > ceiling_linear:
        gain = ceiling_linear / true_peak
        return audio * gain
    return audio

def align_and_mix(background, foregrounds, sr):
    """
    Align and mix background with multiple foreground tracks.
    
    Parameters:
    -----------
    background : numpy.ndarray
        Background audio (mono)
    foregrounds : list of numpy.ndarray
        List of foreground audio tracks (mono)
    sr : int
        Sample rate
    
    Returns:
    --------
    numpy.ndarray : Mixed audio (mono)
    """
    bg_length = len(background)
    
    # Process each foreground track
    processed_foregrounds = []
    for fg in foregrounds:
        # Ensure mono
        fg_mono = ensure_mono(fg)
        
        # Trim if necessary
        if len(fg_mono) > bg_length:
            fg_mono = trim_to_length(fg_mono, bg_length)
        
        # Center align foreground
        fg_length = len(fg_mono)
        if fg_length < bg_length:
            # Pad with zeros to match background length
            pad_before = (bg_length - fg_length) // 2
            pad_after = bg_length - fg_length - pad_before
            fg_mono = np.pad(fg_mono, (pad_before, pad_after), mode='constant')
        
        processed_foregrounds.append(fg_mono)
    
    # Sum all foreground tracks
    if processed_foregrounds:
        total_foreground = sum(processed_foregrounds)
    else:
        total_foreground = np.zeros_like(background)
    
    # Mix with background (ensure both are mono)
    bg_mono = ensure_mono(background)
    mixed = bg_mono + total_foreground
    
    # Apply true-peak ceiling (-1 dBFS)
    mixed = apply_true_peak_ceiling(mixed, sr, ceiling_db=-1.0)
    
    return mixed

def process_combination(background_path, foreground_paths, SNR, L_target, output_dir):
    """
    Process a single combination of tracks.
    
    Parameters:
    -----------
    background_path : Path
        Path to background audio file
    foreground_paths : list of Path
        List of paths to foreground audio files
    SNR : float
        Signal-to-Noise Ratio in dB
    L_target : float
        Target final mix loudness in LUFS
    output_dir : Path
        Output directory
    
    Returns:
    --------
    bool : Success status
    """
    try:
        # Load background audio
        bg_audio, bg_sr = sf.read(str(background_path), always_2d=False)
        
        # Load all foreground audios
        fg_audios = []
        for fg_path in foreground_paths:
            fg_audio, fg_sr = sf.read(str(fg_path), always_2d=False)
            if fg_sr != bg_sr:
                raise ValueError(f"Sample rate mismatch: {fg_path} has {fg_sr}Hz, expected {bg_sr}Hz")
            fg_audios.append(fg_audio)
        
        # Calculate LUFS targets
        N = len(foreground_paths)
        L_fg, L_bg = calculate_lufs_targets(L_target, SNR, N)
        
        # Normalize tracks
        bg_normalized = normalize_to_lufs(bg_audio, bg_sr, L_bg)
        fg_normalized = [normalize_to_lufs(fg, bg_sr, L_fg) for fg in fg_audios]
        
        # Align and mix
        mixed = align_and_mix(bg_normalized, fg_normalized, bg_sr)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        bg_stem = background_path.stem
        fg_stems = [p.stem for p in foreground_paths]
        filename = f"{bg_stem}-{'-'.join(fg_stems)}-{SNR:.1f}.wav"
        output_path = output_dir / filename
        
        # Save as 16-bit PCM WAV (mono)
        sf.write(str(output_path), mixed, bg_sr, subtype='PCM_16')
        
        return True
        
    except Exception as e:
        print(f"Error processing combination: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_audio_files_from_folder(folder_path):
    """
    Get all audio files from a folder.
    
    Parameters:
    -----------
    folder_path : Path
        Path to folder
    
    Returns:
    --------
    list : List of Path objects to audio files
    """
    audio_extensions = ['*.wav', '*.flac', '*.mp3', '*.aiff', '*.aif', '*.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(folder_path.glob(ext))
    
    return audio_files

def batch_process_audio(L_target, SNR_list, background_folder, foreground_folders, num_foregrounds):
    """
    Main batch processing function.
    
    Parameters:
    -----------
    L_target : float
        Target final mix loudness in LUFS
    SNR_list : list of float
        List of SNR values in dB
    background_folder : str
        Path to background folder
    foreground_folders : list of str
        List of paths to foreground folders
    num_foregrounds : int
        Number of simultaneous foreground audios to use
    """
    # Convert to Path objects
    bg_folder = Path(background_folder)
    fg_folders = [Path(f) for f in foreground_folders]
    
    # Validate inputs
    if not bg_folder.exists():
        raise FileNotFoundError(f"Background folder not found: {bg_folder}")
    
    for fg_folder in fg_folders:
        if not fg_folder.exists():
            raise FileNotFoundError(f"Foreground folder not found: {fg_folder}")
    
    if num_foregrounds > len(fg_folders):
        raise ValueError(f"num_foregrounds ({num_foregrounds}) cannot be greater than number of foreground folders ({len(fg_folders)})")
    
    if num_foregrounds < 1:
        raise ValueError("num_foregrounds must be at least 1")
    
    # Get audio files
    bg_files = get_audio_files_from_folder(bg_folder)
    
    # Get files from each foreground folder
    fg_folder_files = []
    for fg_folder in fg_folders:
        fg_files = get_audio_files_from_folder(fg_folder)
        if not fg_files:
            raise ValueError(f"No audio files found in foreground folder: {fg_folder}")
        fg_folder_files.append(fg_files)
    
    if not bg_files:
        raise ValueError(f"No audio files found in background folder: {bg_folder}")
    
    # Generate all combinations of foreground folders
    folder_combinations = list(combinations(range(len(fg_folders)), num_foregrounds))
    
    # Calculate total combinations
    total_combinations = 0
    for folder_combo in folder_combinations:
        # For each folder combination, multiply the number of files in each folder
        combo_product = 1
        for folder_idx in folder_combo:
            combo_product *= len(fg_folder_files[folder_idx])
        total_combinations += len(bg_files) * combo_product * len(SNR_list)
    
    processed = 0
    
    print(f"Starting batch processing...")
    print(f"Background tracks: {len(bg_files)}")
    print(f"Foreground folders: {len(fg_folders)}")
    print(f"Simultaneous foreground audios: {num_foregrounds}")
    print(f"Foreground folder combinations: {len(folder_combinations)}")
    print(f"SNR values: {SNR_list}")
    print(f"Total combinations: {total_combinations}")
    print("-" * 50)
    
    # Create main output directory based on number of foreground audios
    main_output_dir = Path(f"Output_{num_foregrounds}")
    main_output_dir.mkdir(exist_ok=True)
    
    for bg_file in bg_files:
        for folder_combo in folder_combinations:
            # Get the file lists for this combination of folders
            combo_file_lists = [fg_folder_files[i] for i in folder_combo]
            
            # Generate all combinations of files from these folders
            fg_file_combinations = product(*combo_file_lists)
            
            for fg_file_combo in fg_file_combinations:
                for SNR in SNR_list:
                    # Create output directory structure
                    bg_stem = bg_file.stem
                    fg_stems = [p.stem for p in fg_file_combo]
                    
                    # Get folder names for this combination
                    folder_names = [fg_folders[i].name for i in folder_combo]
                    
                    # Output directory structure inside the num_foregrounds folder
                    combination_dir = main_output_dir / f"{bg_stem}-{'-'.join(fg_stems)}"
                    snr_dir = combination_dir / f"{SNR:.1f}"
                    
                    # Process this combination
                    success = process_combination(
                        bg_file, list(fg_file_combo), SNR, L_target, snr_dir
                    )
                    
                    processed += 1
                    
                    if success:
                        print(f"✓ Processed [{processed}/{total_combinations}]: "
                              f"{bg_stem} with {', '.join(fg_stems)} at SNR={SNR:.1f}dB")
                    else:
                        print(f"✗ Failed [{processed}/{total_combinations}]: "
                              f"{bg_stem} with {', '.join(fg_stems)} at SNR={SNR:.1f}dB")
    
    print("-" * 50)
    print(f"Batch processing complete. Processed {processed} combinations.")
    print(f"All outputs saved in: {main_output_dir}/")

def main():
    """
    Example usage of the batch processing script.
    Modify the parameters below for your specific use case.
    """
    # Example parameters - modify these for your use case
    L_target = -14.0  # Target LUFS for final mix
    SNR_list = [6.0, 12.0, 18.0]  # List of SNR values in dB
    num_foregrounds = 2  # Number of simultaneous foreground audios
    
    # Use absolute paths to avoid issues
    script_dir = Path(__file__).parent
    
    # List all foreground folders
    foreground_folders = [
        str(script_dir / "foregrounds" / "airplane"),
        str(script_dir / "foregrounds" / "bark"),
        str(script_dir / "foregrounds" / "car"),
        str(script_dir / "foregrounds" / "horn"),
        str(script_dir / "foregrounds" / "siren")
    ]
    
    background_folder = str(script_dir / "backgrounds")
    
    # Create directories if they don't exist
    Path(background_folder).mkdir(exist_ok=True)
    for fg_folder in foreground_folders:
        Path(fg_folder).mkdir(exist_ok=True)
    
    try:
        batch_process_audio(L_target, SNR_list, background_folder, foreground_folders, num_foregrounds)
    except Exception as e:
        print(f"Error in batch processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Suppress warnings from soundfile about format detection
    warnings.filterwarnings("ignore", message=".*format detection.*")
    
    main()