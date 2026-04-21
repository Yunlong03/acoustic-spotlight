import gradio as gr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.signal import resample
import soundfile as sf
import functools
import os
import torch

# ─── COLORS ───
CORAL = '#FF6B6B'
TEAL = '#4ecdc4'

# ─── VIP PROFILES (simulated Acoustic Contact Book) ───
VIP_PROFILES = [
    {"emoji": "👩", "name": "Sarah", "relation": "Wife", "enrolled": True},
    {"emoji": "👨‍💼", "name": "James", "relation": "Boss", "enrolled": True},
    {"emoji": "🧑‍🤝‍🧑", "name": "Alex", "relation": "Best Friend", "enrolled": False},
]

def load_audio(audio_path):
    """Load audio file, convert to mono."""
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio, sr

def ensure_16k(audio, sr):
    """Resample to 16kHz if needed."""
    if sr != 16000:
        num_samples_new = int(len(audio) * 16000 / sr)
        audio = resample(audio, num_samples_new)
        sr = 16000
    return audio, sr

def make_spectrogram(audio_path, title, color=TEAL):
    """Generate waveform + spectrogram figure."""
    if audio_path is None:
        return None
    audio, sr = load_audio(audio_path)
    time_axis = np.arange(len(audio)) / sr
    frequencies, times, Sxx = scipy_signal.spectrogram(audio, fs=sr, nperseg=1024, noverlap=512)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    fig.patch.set_facecolor('#0e0e1a')
    for ax in axes:
        ax.set_facecolor('#0e0e1a')
        ax.tick_params(colors='#cccccc')
        ax.xaxis.label.set_color('#cccccc')
        ax.yaxis.label.set_color('#cccccc')
        ax.title.set_color(color)

    axes[0].plot(time_axis, audio, color=color, linewidth=0.5)
    axes[0].set_title(f'{title} — Waveform', fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_xlim(0, time_axis[-1])

    img = axes[1].pcolormesh(times, frequencies, Sxx_db, cmap='magma', shading='gouraud')
    axes[1].set_title(f'{title} — Spectrogram', fontweight='bold')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_yscale('log')
    axes[1].set_ylim(20, sr / 2)
    fig.colorbar(img, ax=axes[1], label='dB')

    plt.tight_layout()
    return fig

# ─── SPEECHBRAIN VOICE EMBEDDING ───

@functools.lru_cache(maxsize=1)
def load_speechbrain_model():
    """Load ECAPA-TDNN model once, cache in memory."""
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )

def extract_embedding(audio_path):
    """Extract 192-dim voice embedding."""
    try:
        audio, sr = load_audio(audio_path)
        audio, sr = ensure_16k(audio, sr)
        if len(audio) > 30 * sr:
            audio = audio[:30 * sr]
        signal = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        classifier = load_speechbrain_model()
        embedding = classifier.encode_batch(signal)
        return embedding[0][0].detach().numpy(), True, ""
    except Exception as e:
        return None, False, f"{type(e).__name__}: {str(e)}"

def compare_embeddings(emb1, emb2):
    """Cosine similarity between two voice embeddings."""
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

def create_embedding_plot(target_emb, noisy_emb, similarity):
    """Side-by-side embedding visualization with match score."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3), gridspec_kw={'width_ratios': [5, 5, 2]})
    fig.patch.set_facecolor('#0e0e1a')
    for ax in axes:
        ax.set_facecolor('#0e0e1a')
        ax.tick_params(colors='#cccccc')

    axes[0].bar(range(len(target_emb)), target_emb, color=TEAL, width=1.0)
    axes[0].set_title('Target Voice Print', color=TEAL, fontweight='bold')
    axes[0].set_xlabel('Dimension', color='#cccccc')

    axes[1].bar(range(len(noisy_emb)), noisy_emb, color=CORAL, width=1.0)
    axes[1].set_title('Noisy Audio Print', color=CORAL, fontweight='bold')
    axes[1].set_xlabel('Dimension', color='#cccccc')

    color = TEAL if similarity > 0.6 else '#ffd93d' if similarity > 0.35 else CORAL
    axes[2].barh([0], [similarity], color=color, height=0.5)
    axes[2].set_xlim(0, 1)
    axes[2].set_title('Match', color='#cccccc', fontweight='bold')
    axes[2].text(max(similarity + 0.05, 0.15), 0, f'{similarity:.0%}', va='center', color=color, fontweight='bold', fontsize=14)
    axes[2].set_yticks([])

    plt.tight_layout()
    return fig

# ─── SOLOSPEECH TSE API CALL ───

def extract_voice_solospeech(noisy_path, target_path):
    """Call SoloSpeech Space API for real target speaker extraction."""
    try:
        from gradio_client import Client, handle_file
        client = Client("OpenSound/SoloSpeech")
        result = client.predict(
            test_wav=handle_file(noisy_path),
            enroll_wav=handle_file(target_path),
            api_name="/process_audio"
        )
        return result, True, ""
    except Exception as e:
        return None, False, f"{type(e).__name__}: {str(e)}"

# ─── MAIN PROCESSING ───

def process_audio(noisy_audio, target_voice):
    """
    Full pipeline:
    1. Spectrograms for noisy mix and target voice
    2. Voice embedding comparison (SpeechBrain)
    3. Target speaker extraction (SoloSpeech API)
    4. Spectrogram of extracted result
    """
    if noisy_audio is None or target_voice is None:
        return "⬆️ Record or upload both audio samples, then click Submit.", None, None, None, None, None

    # 1. Spectrograms for inputs
    noisy_fig = make_spectrogram(noisy_audio, "🔊 Before — Noisy Mix", CORAL)
    target_fig = make_spectrogram(target_voice, "🎯 Target Voice (Enrollment)", TEAL)

    # 2. Voice embeddings
    target_emb, t_ok, t_err = extract_embedding(target_voice)
    noisy_emb, n_ok, n_err = extract_embedding(noisy_audio)
    
    if t_ok and n_ok:
        similarity = compare_embeddings(target_emb, noisy_emb)
        emb_fig = create_embedding_plot(target_emb, noisy_emb, similarity)
        
        if similarity > 0.6:
            match_line = f"✅ Voice Match: {similarity:.0%} — Target speaker clearly detected"
        elif similarity > 0.35:
            match_line = f"⚠️ Voice Match: {similarity:.0%} — Target speaker partially detected"
        else:
            match_line = f"❌ Voice Match: {similarity:.0%} — Target speaker not clearly detected"
    else:
        emb_fig = None
        match_line = f"⚠️ Voice embedding unavailable: {t_err or n_err}"

    # 3. Target Speaker Extraction via SoloSpeech
    status = f"🎯 ACOUSTIC SPOTLIGHT — Processing...\n\n{match_line}\n\n⏳ Extracting target voice (this may take 15-30 seconds)..."
    
    extracted_path, tse_ok, tse_err = extract_voice_solospeech(noisy_audio, target_voice)
    
    if tse_ok:
        extracted_fig = make_spectrogram(extracted_path, "🎧 After — Extracted Voice", '#4ecdc4')
        status = (
            f"🎯 ACOUSTIC SPOTLIGHT — Extraction Complete\n\n"
            f"{match_line}\n\n"
            f"🎧 Target speaker successfully extracted from noisy mix.\n"
            f"   Listen to the 'Before' and 'After' below to hear the difference.\n\n"
            f"📊 Voice print: 192-dimensional embedding via ECAPA-TDNN neural network\n"
            f"🔬 Separation: SoloSpeech cascaded generative pipeline (state-of-the-art TSE)\n\n"
            f"🔮 Product vision: Save this voice print to your Acoustic Contact Book.\n"
            f"   Next time you're in a noisy restaurant, tap their profile →\n"
            f"   phone isolates their voice → streams to your hearing aids via Bluetooth."
        )
    else:
        extracted_path = None
        extracted_fig = None
        status = (
            f"🎯 ACOUSTIC SPOTLIGHT — Partial Analysis\n\n"
            f"{match_line}\n\n"
            f"⚠️ Voice extraction unavailable: {tse_err}\n"
            f"   The SoloSpeech separation model may be loading. Try again in 30 seconds."
        )

    return status, noisy_fig, target_fig, emb_fig, extracted_fig, extracted_path


# ─── GRADIO UI ───

with gr.Blocks(title="Acoustic Spotlight", theme=gr.themes.Base(primary_hue="teal", neutral_hue="slate")) as demo:

    gr.Markdown("""
    # 🎯 Acoustic Spotlight
    ### Your Acoustic Contact Book — a VIP list for your ears

    Traditional hearing aids amplify **everything**. Acoustic Spotlight isolates
    the **one voice** you choose. Record someone once. Tap their profile anytime. Hear only them.
    """)

    # VIP Profiles
    gr.Markdown("#### 📇 Acoustic Contact Book")
    with gr.Row():
        for profile in VIP_PROFILES:
            with gr.Column(scale=1, min_width=120):
                status_icon = "✅ Enrolled" if profile["enrolled"] else "➕ Tap to enroll"
                bg = '#1a3a2a' if profile["enrolled"] else '#2a1a1a'
                border = '#4ecdc4' if profile["enrolled"] else '#444'
                status_color = '#4ecdc4' if profile["enrolled"] else '#FF6B6B'
                gr.Markdown(
                    f"<div style='text-align:center; padding:12px; border-radius:12px; "
                    f"background:{bg}; border:1px solid {border};'>"
                    f"<span style='font-size:2em'>{profile['emoji']}</span><br>"
                    f"<b style='color:white'>{profile['name']}</b><br>"
                    f"<small style='color:#999'>{profile['relation']}</small><br>"
                    f"<small style='color:{status_color}'>{status_icon}</small>"
                    f"</div>"
                )

    gr.Markdown("---")
    gr.Markdown("#### 🎙️ New Enrollment / Demo")
    gr.Markdown("*Record the noisy scene first, then the target speaker alone.*")

    with gr.Row():
        with gr.Column():
            noisy_input = gr.Audio(
                label="Step 1: Noisy Mix — the crowded room with multiple voices",
                type="filepath",
                sources=["microphone", "upload"]
            )
        with gr.Column():
            target_input = gr.Audio(
                label="Step 2: Target Voice — 10 seconds of the person you want to hear",
                type="filepath",
                sources=["microphone", "upload"]
            )

    submit_btn = gr.Button("🎯 Extract Target Voice", variant="primary", size="lg")

    status_output = gr.Textbox(label="Analysis Result", lines=10)

    gr.Markdown("#### 📊 Audio Analysis")

    with gr.Row():
        noisy_plot = gr.Plot(label="Before — Noisy Mix")
        target_plot = gr.Plot(label="Target Voice (Enrollment)")

    embedding_plot = gr.Plot(label="Voice Print Comparison (192-dimensional embeddings)")

    gr.Markdown("#### 🎧 Result — Before vs After")

    with gr.Row():
        with gr.Column():
            gr.Markdown("**Before** — *Original noisy recording*")
            # The noisy audio is already in the input above
        with gr.Column():
            gr.Markdown("**After** — *Target speaker extracted*")
            extracted_output = gr.Audio(label="Extracted Voice", type="filepath")

    extracted_plot = gr.Plot(label="After — Extracted Voice Spectrogram")

    submit_btn.click(
        fn=process_audio,
        inputs=[noisy_input, target_input],
        outputs=[status_output, noisy_plot, target_plot, embedding_plot, extracted_plot, extracted_output]
    )

    gr.Markdown("""
    ---
    #### How it works

    1. **Voice Print Extraction** — ECAPA-TDNN neural network converts a 10-second voice sample
       into a 192-dimensional mathematical fingerprint (the voice print)

    2. **Speaker Matching** — Cosine similarity compares the voice print against the noisy recording
       to confirm the target speaker is present

    3. **Target Speaker Extraction** — SoloSpeech cascaded generative pipeline isolates only
       the target speaker's voice from the mixture, removing all other voices and noise

    4. **Product Vision** — Save voice prints to your Acoustic Contact Book.
       In a noisy restaurant, tap a profile → phone extracts that voice in real-time →
       streams clean audio to your hearing aids via Bluetooth

    ---
    *The hearing aid market is worth $9B+. 466 million people worldwide have disabling hearing loss.
    Current devices cannot isolate a specific voice in a crowded room.*

    Built by YL3 · 2026
    """)

if __name__ == "__main__":
    demo.launch()