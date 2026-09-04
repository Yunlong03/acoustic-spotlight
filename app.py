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

# ─── PRE-LOADED DEMO SAMPLES ───

def generate_demo_samples():
    """
    Generate demo audio samples on startup.
    Uses two distinct synthetic 'voices' mixed together for the noisy scene,
    and one clean voice as the enrollment target.
    
    On HF Spaces, these are replaced by real LibriSpeech samples if available.
    """
    sr = 16000
    duration = 6  # seconds
    t = np.linspace(0, duration, sr * duration)

    # Speaker A: higher voice with natural-sounding modulation
    np.random.seed(42)
    pitch_a = 220 + 10 * np.sin(2 * np.pi * 0.5 * t)  # slight pitch variation
    phase_a = 2 * np.pi * np.cumsum(pitch_a) / sr
    speaker_a = (0.4 * np.sin(phase_a) +
                 0.3 * np.sin(2 * phase_a) +
                 0.15 * np.sin(3 * phase_a) +
                 0.1 * np.sin(4 * phase_a))
    # Speech-like envelope (random bursts)
    envelope_a = np.zeros_like(t)
    for start in np.arange(0, duration, 0.4):
        length = np.random.uniform(0.15, 0.35)
        mask = (t >= start) & (t < start + length)
        envelope_a[mask] = np.random.uniform(0.5, 1.0)
    envelope_a = np.convolve(envelope_a, np.ones(800)/800, mode='same')
    speaker_a = speaker_a * envelope_a

    # Speaker B: lower voice
    pitch_b = 140 + 8 * np.sin(2 * np.pi * 0.3 * t)
    phase_b = 2 * np.pi * np.cumsum(pitch_b) / sr
    speaker_b = (0.5 * np.sin(phase_b) +
                 0.25 * np.sin(2 * phase_b) +
                 0.15 * np.sin(3 * phase_b))
    envelope_b = np.zeros_like(t)
    for start in np.arange(0.2, duration, 0.5):
        length = np.random.uniform(0.2, 0.4)
        mask = (t >= start) & (t < start + length)
        envelope_b[mask] = np.random.uniform(0.4, 0.9)
    envelope_b = np.convolve(envelope_b, np.ones(800)/800, mode='same')
    speaker_b = speaker_b * envelope_b

    # Background: restaurant ambiance
    noise = np.random.normal(0, 0.06, len(t))
    # Distant chatter (filtered noise)
    from scipy.signal import butter, sosfilt
    sos = butter(4, [300, 3000], btype='band', fs=sr, output='sos')
    chatter = sosfilt(sos, np.random.normal(0, 0.1, len(t)))

    # Clean target (Speaker A only)
    clean_target = speaker_a * 0.8
    clean_target = clean_target / (np.max(np.abs(clean_target)) + 1e-10) * 0.7

    # Noisy mix (Speaker A + Speaker B + noise + chatter)
    noisy_mix = speaker_a * 0.5 + speaker_b * 0.5 + noise + chatter * 0.3
    noisy_mix = noisy_mix / (np.max(np.abs(noisy_mix)) + 1e-10) * 0.7

    # Save to temp files
    os.makedirs("demo_samples", exist_ok=True)
    target_path = "demo_samples/demo_target_voice.wav"
    noisy_path = "demo_samples/demo_noisy_scene.wav"
    sf.write(target_path, clean_target, sr)
    sf.write(noisy_path, noisy_mix, sr)

    return target_path, noisy_path

# Try to load real LibriSpeech samples, fall back to synthetic
def load_demo_samples():
    """Load demo samples. Try HF datasets first, fall back to synthetic."""
    target_path = "demo_samples/demo_target_voice.wav"
    noisy_path = "demo_samples/demo_noisy_scene.wav"
    
    if os.path.exists(target_path) and os.path.exists(noisy_path):
        return target_path, noisy_path
    
    try:
        from datasets import load_dataset
        print("Loading LibriSpeech samples for demo...")
        ds = load_dataset("openslr/librispeech_asr", "clean", split="validation", streaming=True)
        
        os.makedirs("demo_samples", exist_ok=True)
        samples = []
        speakers_seen = set()
        
        for item in ds:
            speaker = item["speaker_id"]
            if speaker not in speakers_seen and len(samples) < 2:
                audio = item["audio"]["array"]
                sr_orig = item["audio"]["sampling_rate"]
                samples.append((audio, sr_orig, speaker))
                speakers_seen.add(speaker)
            if len(samples) >= 2:
                break
        
        if len(samples) >= 2:
            # Speaker 1 = clean target
            audio1, sr1, _ = samples[0]
            sf.write(target_path, audio1, sr1)
            
            # Mix speakers for noisy scene
            audio2, sr2, _ = samples[1]
            min_len = min(len(audio1), len(audio2))
            audio1_trimmed = audio1[:min_len]
            audio2_trimmed = audio2[:min_len]
            noise = np.random.normal(0, 0.03, min_len)
            noisy = audio1_trimmed * 0.6 + audio2_trimmed * 0.4 + noise
            noisy = noisy / (np.max(np.abs(noisy)) + 1e-10) * 0.7
            sf.write(noisy_path, noisy, sr1)
            
            print("✓ LibriSpeech demo samples loaded")
            return target_path, noisy_path
    except Exception as e:
        print(f"LibriSpeech load failed ({e}), using synthetic samples")
    
    return generate_demo_samples()

# Generate on import
DEMO_TARGET, DEMO_NOISY = load_demo_samples()


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

# ─── ML: SPEECHBRAIN VOICE EMBEDDING ───

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
    """Cosine similarity. 1.0 = same person."""
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

def create_embedding_plot(target_emb, noisy_emb, similarity):
    """Side-by-side embedding visualization."""
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

# ─── SOLOSPEECH TSE API ───

def extract_voice_solospeech(noisy_path, target_path, request=None):
    """Call SoloSpeech for real target speaker extraction."""
    try:
        from gradio_client import Client, handle_file
        import os
        
        # Authenticate to get higher ZeroGPU quota
        # Method 1: Forward user's browser token (best)
        headers = {}
        if request is not None:
            ip_token = request.headers.get('x-ip-token', '')
            if ip_token:
                headers = {"x-ip-token": ip_token}
        
        # Method 2: Use Space's HF token as fallback
        hf_token = os.environ.get("HF_TOKEN", None)
        
        client = Client("OpenSound/SoloSpeech", hf_token=hf_token, headers=headers)
        result = client.predict(
            test_wav=handle_file(noisy_path),
            enroll_wav=handle_file(target_path),
            api_name="/process_audio"
        )
        return result, True, ""
    except Exception as e:
        return None, False, f"{type(e).__name__}: {str(e)}"

# ─── MAIN PROCESSING ───

def process_audio(noisy_audio, target_voice, request: gr.Request = None):
    """Full pipeline: spectrograms + embeddings + SoloSpeech extraction."""
    if noisy_audio is None or target_voice is None:
        return "⬆️ Record or upload both audio samples, then click Submit.", None, None, None, None, None, None

    # 1. Spectrograms
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
    try:
        extracted_path, tse_ok, tse_err = extract_voice_solospeech(noisy_audio, target_voice, request)
    except Exception as e:
        extracted_path, tse_ok, tse_err = None, False, f"TSE call failed: {str(e)}"
    if tse_ok:
        extracted_fig = make_spectrogram(extracted_path, "🎧 After — Extracted Voice", '#4ecdc4')
        status = (
            f"🎯 ACOUSTIC SPOTLIGHT — Extraction Complete\n\n"
            f"{match_line}\n\n"
            f"🎧 Target speaker successfully extracted from noisy mix.\n"
            f"   Compare 'Before' and 'After' below to hear the difference.\n\n"
            f"📊 Voice print: 192-dimensional embedding via ECAPA-TDNN neural network\n"
            f"🔬 Separation: SoloSpeech cascaded generative pipeline (state-of-the-art TSE)\n\n"
            f"🔮 Product vision: Save this voice print to your Acoustic Contact Book.\n"
            f"   Tap a profile → phone isolates their voice → streams to hearing aids via Bluetooth."
        )
    else:
        extracted_path = None
        extracted_fig = None
        status = (
            f"🎯 ACOUSTIC SPOTLIGHT — Partial Analysis\n\n"
            f"{match_line}\n\n"
            f"⚠️ Voice extraction unavailable: {tse_err}\n"
            f"   The SoloSpeech model may be loading (cold start ~60s). Try again shortly.\n"
            f"   If ZeroGPU daily limit is reached, extraction resumes tomorrow."
        )

    return status, target_fig, noisy_fig, emb_fig, noisy_audio, extracted_path, extracted_fig


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
    gr.Markdown("#### 🎙️ Demo — Record your own or use pre-loaded samples")
    gr.Markdown("*Click 'Try Demo' to load example audio, or record/upload your own.*")

    with gr.Row():
        with gr.Column():
            noisy_input = gr.Audio(
                label="Step 1: Noisy Scene — multiple voices mixed together",
                type="filepath",
                sources=["microphone", "upload"]
            )
        with gr.Column():
            target_input = gr.Audio(
                label="Step 2: Target Voice — the one person you want to hear",
                type="filepath",
                sources=["microphone", "upload"]
            )

    with gr.Row():
        demo_btn = gr.Button("🎧 Try Demo (pre-loaded samples)", variant="secondary", size="lg")
        submit_btn = gr.Button("🎯 Extract Target Voice", variant="primary", size="lg")

    def load_demo():
        return DEMO_NOISY, DEMO_TARGET

    demo_btn.click(
        fn=load_demo,
        outputs=[noisy_input, target_input]
    )

    status_output = gr.Textbox(label="Analysis Result", lines=10)

    gr.Markdown("#### 📊 Voice Print Analysis")
    with gr.Row():
        target_plot = gr.Plot(label="Target Voice")
        noisy_plot = gr.Plot(label="Noisy Audio")

    embedding_plot = gr.Plot(label="Voice Print Comparison (192-dimensional embeddings)")

    gr.Markdown("#### 🔊 Before vs After — Listen to the difference")
    with gr.Row():
        with gr.Column():
            gr.Markdown("**🔊 BEFORE** — *Original noisy recording (multiple voices + noise)*")
            before_audio = gr.Audio(label="Before — Noisy Mix", type="filepath", interactive=False)
        with gr.Column():
            gr.Markdown("**🎧 AFTER** — *Target speaker extracted by neural network*")
            after_audio = gr.Audio(label="After — Extracted Voice", type="filepath", interactive=False)

    extracted_plot = gr.Plot(label="After — Extracted Voice Spectrogram")

    submit_btn.click(
        fn=process_audio,
        inputs=[noisy_input, target_input],
        outputs=[status_output, target_plot, noisy_plot, embedding_plot, before_audio, after_audio, extracted_plot]
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
