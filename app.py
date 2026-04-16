import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.io import wavfile
import soundfile as sf

def load_audio(audio_path):
    """
    Load audio file using soundfile (more reliable than scipy.io.wavfile for various formats).
    Returns (audio_array, sample_rate).
    """
    audio, sr = sf.read(audio_path)
    # If stereo, convert to mono by averaging channels
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio, sr

def analyze_audio(audio_path, title):
    """
    Takes a path to an audio file.
    Returns a matplotlib figure showing waveform + spectrogram.
    """
    if audio_path is None:
        return None
    
    # Load audio
    audio, sr = load_audio(audio_path)
    
    # Create time axis for waveform
    time = np.arange(len(audio)) / sr
    
    # Compute spectrogram using scipy
    frequencies, times, Sxx = scipy_signal.spectrogram(
        audio, 
        fs=sr,
        nperseg=1024,
        noverlap=512
    )
    
    # Convert to dB for better visualization
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Create figure with 2 plots stacked vertically
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    # Top plot: waveform (amplitude over time)
    axes[0].plot(time, audio, color='#00a8ff', linewidth=0.5)
    axes[0].set_title(f'{title} — Waveform')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_xlim(0, time[-1])
    axes[0].grid(True, alpha=0.3)
    
    # Bottom plot: spectrogram (frequencies over time)
    img = axes[1].pcolormesh(times, frequencies, Sxx_db, cmap='viridis', shading='gouraud')
    axes[1].set_title(f'{title} — Spectrogram (voice print)')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_yscale('log')
    axes[1].set_ylim(20, sr/2)
    fig.colorbar(img, ax=axes[1], label='Power (dB)')
    
    plt.tight_layout()
    return fig

def process_audio(target_voice, noisy_audio):
    """
    Day 2: Real spectrograms of uploaded audio.
    Day 3: Will add real TSE separation.
    """
    if target_voice is None or noisy_audio is None:
        return "Please upload both audio files.", None, None, None
    
    # Generate real spectrograms for both inputs
    target_fig = analyze_audio(target_voice, "Target Voice (Voice Print Source)")
    noisy_fig = analyze_audio(noisy_audio, "Noisy Environment (To Be Filtered)")
    
    status = "✅ Audio analyzed — real spectrograms generated. Day 3 will add voice isolation processing."
    
    # For now, output = input (placeholder until Day 3)
    return status, target_fig, noisy_fig, noisy_audio

# Build the Gradio interface
demo = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(label="Step 1: Upload Target Voice (10 sec sample)", type="filepath"),
        gr.Audio(label="Step 2: Upload Noisy Audio", type="filepath"),
    ],
    outputs=[
        gr.Textbox(label="Status"),
        gr.Plot(label="Target Voice Analysis"),
        gr.Plot(label="Noisy Audio Analysis"),
        gr.Audio(label="Isolated Voice (Day 3 output)"),
    ],
    title="🎯 Acoustic Spotlight — Voice Print Isolation Demo",
    description="""
    **The Cocktail Party Problem, Solved.**
    
    Traditional hearing aids amplify everything. Acoustic Spotlight isolates 
    the ONE voice you want to hear.
    
    1. Upload a 10-second sample of your target speaker
    2. Upload a noisy recording with multiple voices
    3. Click Submit — see the voice print analysis (Day 2) and isolated voice (Day 3)
    
    *This is a concept demo for the Acoustic Contact Book — 
    your personalized VIP list for your ears.*
    """,
    article="""
    **How it works:** The system extracts a mathematical voice embedding 
    (voice print) from the target speaker sample, then uses Target Speaker 
    Extraction (TSE) to isolate that voice from the noisy recording.
    
    **Market:** The hearing aid market is worth $9B+. Current devices cannot 
    isolate a specific voice in a crowded room.
    
    Built by Dragon S · Exponential Entrepreneur Bootcamp 2026
    """,
)

if __name__ == "__main__":
    demo.launch()