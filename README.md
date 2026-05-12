title: Acoustic Spotlight
emoji: 🎯
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: "5.23.0"
app_file: app.py
pinned: false
license: mit
---
**Your Acoustic Contact Book — a VIP list for your ears.**

Traditional hearing aids amplify everything. Acoustic Spotlight isolates the one voice you choose.

[**→ Try the live demo**](https://Yunlong03-acoustic-spotlight.hf.space)

---

## The Problem

466 million people worldwide have disabling hearing loss. Current hearing aids — Signia, Oticon, Phonak — amplify the entire environment. In a busy restaurant, every voice, every plate clink, every chair scrape gets louder equally. Audiologists say emphasizing certain frequencies is enough. It isn't. All human voices occupy the same frequency range (200–4000Hz). No frequency filter can separate your wife's voice from the stranger at the next table.

## The Solution

Record someone's voice for 10 seconds. The app saves their mathematical voice print — a 192-dimensional fingerprint unique to their voice. Next time you're in a noisy room, tap their profile. A neural network extracts only their voice from the chaos and streams it to your hearing aids via Bluetooth.

We call it the **Acoustic Contact Book**: save the voices of the people who matter most. Your wife, your boss, your best friend. Tap a name, hear only them.

## What This Demo Does

1. **Voice Print Extraction** — ECAPA-TDNN neural network (SpeechBrain, trained on VoxCeleb) creates a 192-dimensional voice embedding from a 10-second sample
2. **Speaker Matching** — Cosine similarity confirms the target speaker is present in the noisy recording
3. **Target Speaker Extraction** — SoloSpeech (Johns Hopkins, state-of-the-art TSE) isolates only the target voice, removing all other speakers and background noise
4. **Before/After Comparison** — Spectrograms and audio playback let you hear and see the difference

## Quick Start

Visit the [live demo](https://YOUR_USERNAME-acoustic-spotlight.hf.space) and click **"Try Demo"** to hear pre-loaded examples, or record your own voice samples directly from your phone.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Gradio (Python) |
| Audio analysis | scipy, soundfile, matplotlib |
| Voice embeddings | SpeechBrain ECAPA-TDNN |
| Speaker extraction | SoloSpeech (via API) |
| ML engine | PyTorch |
| Hosting | Hugging Face Spaces |

## Market Opportunity

- Global hearing aid market: **$9B+**
- Target users: **50-80M** active hearing aid users dissatisfied with noisy environments
- Expansion: normal-hearing earbuds users wanting voice focus
- No existing product offers persistent, audio-only, multi-person voice profiles

## Competitive Landscape

| Solution | Approach | Limitation |
|----------|----------|-----------|
| Signia / Oticon / Phonak | Hardware beamforming | Amplifies whatever is in front — even strangers |
| HeardThat (app) | General noise reduction | Cannot separate Speaker A from Speaker B |
| Look Once to Hear (UW) | Visual enrollment via camera | One speaker, requires camera |
| **Acoustic Spotlight** | **Audio enrollment + persistent profiles** | **Demo stage — real-time processing in development** |

## Status

This is a **pitch demo prototype**, not a finished product. Built in 5 days as part of an exponential entrepreneur bootcamp. The voice extraction works (you can hear it), but processing takes 15-70 seconds — real-time on-device inference is the next engineering milestone.

## What's Needed

- **Technical cofounder**: Audio ML engineer (model optimization for real-time mobile inference)
- **Seed funding**: $100-175K for 12-month prototype → pilot → launch
- **Contact**: [Your contact info]

## License

Demo code: MIT. SoloSpeech model: CC-BY-NC-4.0 (non-commercial). Production path: SpeechBrain SepFormer (MIT license) or commercial license negotiation.

---

Built by YL3 · 2026
