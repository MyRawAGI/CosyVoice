"""
CosyVoice Novel TTS Pipeline
Markdown → Opus audiobooks with llama.cpp backend, progress tracking, and resume support.
"""

import os
import gc
import re
import json
import argparse
import subprocess
import tempfile
import time
import sys
import torch
import torchaudio
from markdown import markdown
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_REF_TEXT = (
    "Даже если семья, даже если стоят за него, как стены, все знакомые, друзья — "
    "не прощайте ему измен, никогда не прощайте измены. Даже самым любимым женам, "
    "пусть тяжело, пусть даже словом попробует или делом она оправдываться снова — "
    "не прощайте вы ей измены, никогда не прощайте измены ни любимым друзьям, "
    "ни любимым супругам — только разлука избавит."
)


def numbers_to_russian_words(text):
    """Replace standalone numbers with Russian words for correct TTS reading."""
    from num2words import num2words

    def replace_number(m):
        num = int(m.group(0))
        try:
            return num2words(num, lang='ru')
        except:
            return m.group(0)

    # Replace standalone numbers (not part of other words)
    text = re.sub(r'\b(\d+)\b', replace_number, text)
    return text


def md_to_clean_text(md_content):
    html = markdown(md_content)
    text = re.sub(r'<p>|<br\s*/?>|<li>', '\n', html)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\*#_~`]', '', text)
    text = re.sub(r'[«»<<>>""'']', '"', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_progress(output_dir):
    path = os.path.join(output_dir, 'progress.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_progress(output_dir, progress):
    path = os.path.join(output_dir, 'progress.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def trim_ref_audio(ref_wav_path, max_sec=30):
    wav, sr = torchaudio.load(ref_wav_path)
    dur = wav.shape[1] / sr
    if dur > max_sec:
        wav = wav[:, :int(sr * max_sec)]
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        torchaudio.save(tmp.name, wav, sr)
        return tmp.name
    return ref_wav_path


def crossfade_merge(chunks, sr=22050, crossfade_ms=30):
    if len(chunks) <= 1:
        return chunks[0] if chunks else torch.zeros(1, 0)
    crossfade_samples = int(sr * crossfade_ms / 1000)
    merged = [chunks[0]]
    for chunk in chunks[1:]:
        prev = merged[-1]
        if prev.shape[-1] >= crossfade_samples and chunk.shape[-1] >= crossfade_samples:
            fade_out = torch.linspace(1.0, 0.0, crossfade_samples)
            fade_in = torch.linspace(0.0, 1.0, crossfade_samples)
            overlap = prev[..., -crossfade_samples:] * fade_out + chunk[..., :crossfade_samples] * fade_in
            merged[-1] = torch.cat([prev[..., :-crossfade_samples], overlap], dim=-1)
            merged.append(chunk[..., crossfade_samples:])
        else:
            merged.append(chunk)
    return torch.cat(merged, dim=-1)


def concat_wavs_ffmpeg(wav_files, output_path):
    list_path = output_path + '.list.txt'
    with open(list_path, 'w', encoding='utf-8') as f:
        for wf in wav_files:
            f.write(f"file '{os.path.abspath(wf)}'\n")
    cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_path, '-c', 'copy', output_path]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.remove(list_path)


def wav_to_opus(wav_path, opus_path, bitrate=48):
    cmd = [
        'ffmpeg', '-y', '-i', wav_path,
        '-c:a', 'libopus', '-b:a', f'{bitrate}k',
        '-ac', '1', '-application', 'voip',
        opus_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def synthesize_chunk(cosyvoice, text, ref_text, ref_wav, max_retries=3):
    for attempt in range(max_retries):
        try:
            audio_chunks = []
            for output in cosyvoice.inference_zero_shot(
                tts_text=text,
                prompt_text=ref_text,
                prompt_wav=ref_wav,
            ):
                audio_chunks.append(output['tts_speech'])

            if audio_chunks:
                return crossfade_merge(audio_chunks)
            return None
        except torch.cuda.OutOfMemoryError:
            print(f"    CUDA OOM, retry {attempt+1}/{max_retries}")
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)
        except Exception as e:
            print(f"    Error: {e}, retry {attempt+1}/{max_retries}")
            time.sleep(1)
    return None


def process_file(cosyvoice, md_path, output_dir, ref_text, ref_wav, bitrate, progress, file_key):
    basename = os.path.splitext(os.path.basename(md_path))[0]
    temp_dir = os.path.join(output_dir, 'temp', basename)
    os.makedirs(temp_dir, exist_ok=True)

    final_wav = os.path.join(output_dir, f"{basename}.wav")
    final_opus = os.path.join(output_dir, f"{basename}.opus")

    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    clean_text = md_to_clean_text(content)
    clean_text = numbers_to_russian_words(clean_text)
    if not clean_text:
        print(f"  Empty text, skipping")
        return

    # Split text by sentences — CosyVoice frontend does its own splitting,
    # but we pre-split to keep chunks small enough for reliable generation
    sentences = re.split(r'(?<=[.!?…])\s+', clean_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Group sentences into chunks of ~200-300 chars (safe for CosyVoice)
    chunks = []
    current = ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > 300:
            chunks.append(current)
            current = s
        else:
            current = (current + " " + s).strip() if current else s
    if current:
        chunks.append(current)

    print(f"  Text split into {len(chunks)} chunks")

    # Resume: check which chunks are already done
    file_state = progress.get(file_key, {})
    start_chunk = file_state.get('next_chunk', 0)

    if start_chunk > 0:
        print(f"  Resuming from chunk {start_chunk}/{len(chunks)}")

    # Update progress to in_progress
    progress[file_key] = {'status': 'in_progress', 'next_chunk': start_chunk, 'total': len(chunks)}
    save_progress(output_dir, progress)

    # Generate chunks
    temp_wavs = []
    # Collect existing temp files from previous run
    for i in range(start_chunk):
        tp = os.path.join(temp_dir, f"chunk_{i:05d}.wav")
        if os.path.exists(tp):
            temp_wavs.append(tp)

    failed_chunks = []
    for i in range(start_chunk, len(chunks)):
        chunk_text = chunks[i]
        temp_path = os.path.join(temp_dir, f"chunk_{i:05d}.wav")

        print(f"  Chunk {i+1}/{len(chunks)} ({len(chunk_text)} chars): {chunk_text[:60]}...")
        t0 = time.time()

        audio = synthesize_chunk(cosyvoice, chunk_text, ref_text, ref_wav)
        if audio is not None and audio.shape[-1] > 0:
            torchaudio.save(temp_path, audio, cosyvoice.sample_rate)
            temp_wavs.append(temp_path)
            dur = audio.shape[-1] / cosyvoice.sample_rate
            print(f"    Done: {dur:.1f}s in {time.time()-t0:.1f}s")
        else:
            failed_chunks.append(i)
            print(f"    FAILED — no audio generated")

        # Update progress after each chunk
        progress[file_key]['next_chunk'] = i + 1
        save_progress(output_dir, progress)

        torch.cuda.empty_cache()
        gc.collect()

    if not temp_wavs:
        print(f"  No audio generated for {basename}")
        progress[file_key] = {'status': 'failed', 'next_chunk': len(chunks), 'total': len(chunks)}
        save_progress(output_dir, progress)
        return

    # Concatenate all chunks
    print(f"  Concatenating {len(temp_wavs)} chunks...")
    if len(temp_wavs) == 1:
        import shutil
        shutil.copy2(temp_wavs[0], final_wav)
    else:
        concat_wavs_ffmpeg(temp_wavs, final_wav)

    # Convert to opus
    print(f"  Converting to opus ({bitrate}kbps)...")
    wav_to_opus(final_wav, final_opus, bitrate)

    # Stats
    wav_mb = os.path.getsize(final_wav) / (1024 * 1024)
    opus_mb = os.path.getsize(final_opus) / (1024 * 1024)
    print(f"  Result: WAV {wav_mb:.1f}MB → Opus {opus_mb:.1f}MB")

    # Cleanup temp
    for f in temp_wavs:
        try:
            os.remove(f)
        except:
            pass
    try:
        os.rmdir(temp_dir)
    except:
        pass

    # Remove intermediate WAV
    try:
        os.remove(final_wav)
    except:
        pass

    if failed_chunks:
        print(f"  WARNING: {len(failed_chunks)} chunks failed: {failed_chunks}")

    progress[file_key] = {'status': 'done', 'next_chunk': len(chunks), 'total': len(chunks)}
    save_progress(output_dir, progress)


def main():
    parser = argparse.ArgumentParser(description='CosyVoice Novel TTS — MD to Opus')
    parser.add_argument('--novel-dir', default=os.path.join(SCRIPT_DIR, 'novel'),
                        help='Folder with .md files (default: novel/)')
    parser.add_argument('--output-dir', default=os.path.join(SCRIPT_DIR, 'audio'),
                        help='Output folder for .opus files (default: audio/)')
    parser.add_argument('--gguf', default=os.path.join(SCRIPT_DIR, 'cosyvoice_llm_f32.gguf'),
                        help='Path to GGUF model')
    parser.add_argument('--model-dir', default=os.path.join(SCRIPT_DIR, 'pretrained_models', 'Fun-CosyVoice3-0.5B'),
                        help='Path to CosyVoice model dir')
    parser.add_argument('--ref-wav', default=os.path.join(SCRIPT_DIR, 'ref_voice.wav'),
                        help='Reference voice WAV for zero-shot')
    parser.add_argument('--ref-text', default=DEFAULT_REF_TEXT,
                        help='Reference text (transcription of ref_wav)')
    parser.add_argument('--bitrate', type=int, default=48,
                        help='Opus bitrate in kbps (default: 48)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Ignore progress file, start fresh')
    args = parser.parse_args()

    # Check ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except:
        print("ERROR: ffmpeg not found. Install ffmpeg and add to PATH.")
        return

    # Check inputs
    if not os.path.exists(args.novel_dir):
        print(f"ERROR: novel folder not found: {args.novel_dir}")
        print("Create a 'novel/' folder with .md files.")
        return

    md_files = sorted([f for f in os.listdir(args.novel_dir) if f.endswith('.md')])
    if not md_files:
        print(f"No .md files found in {args.novel_dir}")
        return

    # Load or reset progress
    os.makedirs(args.output_dir, exist_ok=True)
    progress = {} if args.no_resume else load_progress(args.output_dir)

    # Prepare ref audio
    print(f"Preparing ref audio: {args.ref_wav}")
    ref_wav = trim_ref_audio(args.ref_wav)

    # Load model
    print(f"Loading CosyVoice model...")
    sys.path.insert(0, SCRIPT_DIR)
    from cosyvoice.cli.cosyvoice import AutoModel

    t0 = time.time()
    cosyvoice = AutoModel(
        model_dir=args.model_dir,
        load_llama_cpp=True,
        gguf_model_path=args.gguf,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    # Process files
    print(f"Found {len(md_files)} .md files\n")
    for md_file in md_files:
        file_key = md_file
        file_state = progress.get(file_key, {})

        if file_state.get('status') == 'done' and not args.no_resume:
            print(f"[SKIP] {md_file} — already done")
            continue

        print(f"{'='*60}")
        print(f"Processing: {md_file}")
        md_path = os.path.join(args.novel_dir, md_file)

        try:
            process_file(
                cosyvoice=cosyvoice,
                md_path=md_path,
                output_dir=args.output_dir,
                ref_text=args.ref_text,
                ref_wav=ref_wav,
                bitrate=args.bitrate,
                progress=progress,
                file_key=file_key,
            )
        except Exception as e:
            print(f"  FATAL: {e}")
            import traceback
            traceback.print_exc()

        print()

    print("All done.")


if __name__ == '__main__':
    main()
