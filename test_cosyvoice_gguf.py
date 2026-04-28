import os
import sys
import time
import tempfile
import torch
import torchaudio

# Paths
COSYVOICE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(COSYVOICE_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B")
GGUF_PATH = os.path.join(COSYVOICE_DIR, "cosyvoice_llm_f32.gguf")
REF_AUDIO = os.path.join(COSYVOICE_DIR, "ref_voice.wav")
OUTPUT_DIR = os.path.join(COSYVOICE_DIR, "test_output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Model dir : {MODEL_DIR}")
print(f"GGUF path : {GGUF_PATH}")
print(f"GGUF exists: {os.path.exists(GGUF_PATH)}")
print(f"Model exists: {os.path.exists(MODEL_DIR)}")
print()

sys.path.insert(0, COSYVOICE_DIR)

from cosyvoice.cli.cosyvoice import AutoModel

print("Loading model with llama.cpp backend...")
t0 = time.time()

cosyvoice = AutoModel(
    model_dir=MODEL_DIR,
    load_llama_cpp=True,
    gguf_model_path=GGUF_PATH,
)

print(f"Model loaded in {time.time() - t0:.1f}s")
print()

# Reference audio for zero-shot voice cloning
if not os.path.exists(REF_AUDIO):
    print(f"ERROR: reference audio not found: {REF_AUDIO}")
    print("Place a WAV file (3-10s, clean speech) and set REF_AUDIO path.")
    sys.exit(1)

# Trim reference audio to 30s max (CosyVoice hard limit)
MAX_REF_SEC = 30
ref_wav, ref_sr = torchaudio.load(REF_AUDIO)
ref_dur = ref_wav.shape[1] / ref_sr
if ref_dur > MAX_REF_SEC:
    print(f"Trimming ref audio {ref_dur:.1f}s -> {MAX_REF_SEC}s")
    ref_wav = ref_wav[:, : int(ref_sr * MAX_REF_SEC)]
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(tmp.name, ref_wav, ref_sr)
    REF_AUDIO = tmp.name

ref_text = "Даже если семья, даже если стоят за него, как стены, все знакомые, друзья — не прощайте ему измен, никогда не прощайте измены. Даже самым любимым женам, пусть тяжело, пусть даже словом попробует или делом она оправдываться снова — не прощайте вы ей измены, никогда не прощайте измены ни любимым друзьям, ни любимым супругам — только разлука избавит."

# Test texts
test_cases = [
    ("ru", "Первые лучи рассвета коснулись шпилей Восточного Сектора, где располагались покои учеников высшего ранга. Эти мгновения всегда были особенными – когда грань между ночью и днём размывается, а мир замирает в предвкушении нового дня. Белоснежные дворцы, парящие среди облаков, постепенно проступали из утренней дымки, словно небесные острова в море тумана."),
]

for lang, text in test_cases:
    print(f"Synthesizing [{lang}]: {text[:50]}...")
    t1 = time.time()

    output_path = os.path.join(OUTPUT_DIR, f"test_{lang}.wav")
    audio_chunks = []

    for output in cosyvoice.inference_zero_shot(
        tts_text=text,
        prompt_text=ref_text,
        prompt_wav=REF_AUDIO,
    ):
        audio_chunks.append(output["tts_speech"])

    if audio_chunks:
        # Crossfade between chunks to avoid boundary artifacts
        crossfade_ms = 30  # 30ms crossfade
        crossfade_samples = int(22050 * crossfade_ms / 1000)
        merged = [audio_chunks[0]]
        for chunk in audio_chunks[1:]:
            prev = merged[-1]
            if prev.shape[-1] >= crossfade_samples and chunk.shape[-1] >= crossfade_samples:
                fade_out = torch.linspace(1.0, 0.0, crossfade_samples)
                fade_in = torch.linspace(0.0, 1.0, crossfade_samples)
                overlap = prev[..., -crossfade_samples:] * fade_out + chunk[..., :crossfade_samples] * fade_in
                merged[-1] = torch.cat([prev[..., :-crossfade_samples], overlap], dim=-1)
                merged.append(chunk[..., crossfade_samples:])
            else:
                merged.append(chunk)
        audio = torch.cat(merged, dim=-1)
        torchaudio.save(output_path, audio, 22050)
        elapsed = time.time() - t1
        duration = audio.shape[-1] / 22050
        rtf = elapsed / duration
        print(f"  Done in {elapsed:.2f}s | audio {duration:.1f}s | RTF: {rtf:.3f}")
        print(f"  Saved: {output_path}")
    else:
        print("  No audio generated!")
    print()

print("All tests done.")
