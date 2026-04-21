import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.io import wavfile
import soundfile as sf
import tempfile
import os
import functools

def load_audio(audio_path):
    """Load audio file, convert to mono if stereo."""
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio, sr

def analyze_audio(audio_path, title):
    """Generate waveform + spectrogram figure from audio file."""
    if audio_path is None:
        return None
    
    audio, sr = load_audio(audio_path)
    time = np.arange(len(audio)) / sr
    frequencies, times, Sxx = scipy_signal.spectrogram(audio, fs=sr, nperseg=1024, noverlap=512)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    axes[0].plot(time, audio, color='#00a8ff', linewidth=0.5)
    axes[0].set_title(f'{title} — Waveform')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_xlim(0, time[-1])
    axes[0].grid(True, alpha=0.3)
    
    img = axes[1].pcolormesh(times, frequencies, Sxx_db, cmap='viridis', shading='gouraud')
    axes[1].set_title(f'{title} — Spectrogram (voice print)')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_yscale('log')
    axes[1].set_ylim(20, sr/2)
    fig.colorbar(img, ax=axes[1], label='Power (dB)')
    
    plt.tight_layout()
    return fig

import functools

@functools.lru_cache(maxsize=1)
def load_speechbrain_model():
    """Load SpeechBrain model once, cache in memory."""
    from speechbrain.inference.speaker import EncoderClassifier
    print("Loading SpeechBrain ECAPA-TDNN model...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    print("Model loaded successfully.")
    return classifier

def extract_embedding(audio_path):
    """Extract voice embedding using SpeechBrain ECAPA-TDNN."""
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        
        # Load audio using soundfile (not torchaudio — avoids TorchCodec dependency)
        audio, sr = load_audio(audio_path)
        
        # Resample to 16kHz if needed (SpeechBrain expects 16kHz)
        if sr != 16000:
            from scipy.signal import resample
            num_samples_new = int(len(audio) * 16000 / sr)
            audio = resample(audio, num_samples_new)
            sr = 16000
            print(f"Resampled to 16kHz: {len(audio)} samples")
        
        # Trim to 30 seconds max to avoid long processing
        max_samples = 30 * sr
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            print(f"Trimmed to 30 seconds")
        
        # Convert to torch tensor with correct shape (1, num_samples)
        signal = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        print(f"Signal tensor shape: {signal.shape}")
        
        # Load model (cached after first call)
        classifier = load_speechbrain_model()
        
        # Extract embedding
        embedding = classifier.encode_batch(signal)
        print(f"Embedding extracted: {embedding.shape}")
        
        return embedding[0][0].detach().numpy(), True, ""
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"SpeechBrain error: {error_msg}")
        return None, False, error_msg

def compare_embeddings(emb1, emb2):
    """Cosine similarity between two voice embeddings. 1.0 = identical, 0.0 = completely different."""
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(cos_sim)

def process_audio(target_voice, noisy_audio):
    """
    Full pipeline:
    1. Analyze both audio files (spectrograms)
    2. Extract voice embeddings via SpeechBrain
    3. Compare embeddings (similarity score)
    """
    if target_voice is None or noisy_audio is None:
        return "Please upload both audio files.", None, None, None, None
    
    # Generate spectrograms (always works — scipy)
    target_fig = analyze_audio(target_voice, "Target Voice (Voice Print Source)")
    noisy_fig = analyze_audio(noisy_audio, "Noisy Environment (To Be Filtered)")
    
    # Attempt voice embedding extraction (SpeechBrain — may fail locally, works on HF Spaces)
    target_emb, target_ok, target_err = extract_embedding(target_voice)
    noisy_emb, noisy_ok, noisy_err = extract_embedding(noisy_audio)
    
    if target_ok and noisy_ok:
        similarity = compare_embeddings(target_emb, noisy_emb)
        
        # Create embedding visualization
        emb_fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        axes[0].bar(range(len(target_emb)), target_emb, color='#00a8ff', width=1.0)
        axes[0].set_title('Target Voice Embedding (192 dimensions)')
        axes[0].set_xlabel('Dimension')
        axes[0].set_ylabel('Value')
        axes[1].bar(range(len(noisy_emb)), noisy_emb, color='#ff6b6b', width=1.0)
        axes[1].set_title('Noisy Audio Embedding (192 dimensions)')
        axes[1].set_xlabel('Dimension')
        axes[1].set_ylabel('Value')
        plt.tight_layout()
        
        if similarity > 0.7:
            match_text = f"HIGH MATCH ({similarity:.1%}) — Same speaker detected in both recordings"
        elif similarity > 0.4:
            match_text = f"PARTIAL MATCH ({similarity:.1%}) — Target speaker may be present in noisy audio"
        else:
            match_text = f"LOW MATCH ({similarity:.1%}) — Target speaker not clearly detected"
        
        status = f"✅ Voice print analysis complete.\n\n🎯 Voice Match Score: {similarity:.1%}\n{match_text}\n\n📊 Voice embeddings: 192-dimensional vectors extracted via ECAPA-TDNN (trained on VoxCeleb).\n\nIn the full Acoustic Spotlight product, this voice print would be saved to your Acoustic Contact Book and used to isolate this speaker in real-time via Bluetooth to your hearing aids."
    else:
        emb_fig = None
        status = f"✅ Spectrograms generated.\n\n⚠️ Voice embedding extraction failed.\nTarget error: {target_err}\nNoisy error: {noisy_err}\n\nThis diagnostic info will help debug the issue."
    
    return status, target_fig, noisy_fig, emb_fig, noisy_audio

# Build the Gradio interface
demo = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(label="Step 1: Target Voice — record or upload a 10-sec sample", type="filepath", sources=["microphone", "upload"]),
        gr.Audio(label="Step 2: Noisy Audio — record or upload the noisy environment", type="filepath", sources=["microphone", "upload"]),
    ],
    outputs=[
        gr.Textbox(label="Analysis Result", lines=8),
        gr.Plot(label="Target Voice — Waveform & Spectrogram"),
        gr.Plot(label="Noisy Audio — Waveform & Spectrogram"),
        gr.Plot(label="Voice Embeddings Comparison (192-dim vectors)"),
        gr.Audio(label="Processed Audio (full isolation coming soon)"),
    ],
    title="🎯 Acoustic Spotlight — Voice Print Isolation Demo",
    description="""
    **The Cocktail Party Problem, Solved.**
    
    Traditional hearing aids amplify everything — even voices you don't want to hear.
    Acoustic Spotlight isolates the ONE voice you choose.
    
    **How to use this demo:**
    1. **Record or upload** a 10-second sample of your target speaker (their "voice print")
    2. **Record or upload** a noisy scene where the same person is speaking among other sounds
    3. **Click Submit** — the system extracts the voice print, analyzes both recordings, and shows the match
    
    *This is the Acoustic Contact Book — your personalized VIP list for your ears.
    Record someone once, tap their profile anytime, hear only them.*
    """,
    article="""
    **How it works:** The system extracts a 192-dimensional mathematical voice embedding 
    (voice print) from the target speaker using ECAPA-TDNN, a neural network trained on 
    thousands of speakers. It then compares this embedding against the noisy recording 
    to identify where the target voice appears.
    
    **The vision:** Save voice prints of the people who matter most — your partner, your boss, 
    your best friend. Walk into a noisy restaurant, tap a profile, put your phone on the table. 
    The app locks onto their voice and streams only that voice to your hearing aids via Bluetooth.
    
    **Market:** The hearing aid market is worth $9B+. 466 million people worldwide have 
    disabling hearing loss. Current devices cannot isolate a specific voice in a crowded room.
    
    Built by Dragon S · Exponential Entrepreneur Bootcamp 2026
    """,
)

if __name__ == "__main__":
    demo.launch()