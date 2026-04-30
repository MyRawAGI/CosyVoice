"""
CosyVoice Novel TTS Pipeline
Markdown → Opus audiobooks with llama.cpp backend, progress tracking, and resume support.
"""

import os
import sys

# Windows: add PyTorch lib dir to PATH before any other imports so that
# onnxruntime-gpu can find cuDNN/CUBLAS DLLs that ship with torch.
if sys.platform == 'win32':
    try:
        import torch as _torch_pre
        _torch_lib = os.path.join(os.path.dirname(_torch_pre.__file__), 'lib')
        if os.path.isdir(_torch_lib):
            os.add_dll_directory(_torch_lib)
            os.environ['PATH'] = _torch_lib + os.pathsep + os.environ.get('PATH', '')
    except Exception:
        pass

import gc
import re
import json
import argparse
import subprocess
import tempfile
import time
import shutil
import wave
import torch
import torchaudio
from markdown import markdown
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_REF_TEXT = "Даже если семья даже если стоят за него как стены все знакомые друзья не прощайте ему измен никогда не прощайте измены даже самым любимым женам пусть тяжело пусть даже словом попробует или делом она оправдываться снова не прощайте вы ей измены"


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
    # Collapse spaces/tabs within lines but preserve newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    return text.strip()


def load_progress(output_dir):
    path = os.path.join(output_dir, 'progress.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_progress(output_dir, progress):
    path = os.path.join(output_dir, 'progress.json')
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


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


def load_and_merge_wavs(wav_files, output_path, sr=22050, crossfade_ms=30):
    """Merge WAV files with crossfade, streaming output to avoid loading all into RAM."""
    crossfade_samples = int(sr * crossfade_ms / 1000)
    with wave.open(output_path, 'wb') as out_wav:
        out_wav.setnchannels(1)
        out_wav.setsampwidth(2)
        out_wav.setframerate(sr)
        pending = None
        for wf in wav_files:
            wav, file_sr = torchaudio.load(wf)
            if file_sr != sr:
                wav = torchaudio.functional.resample(wav, file_sr, sr)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if pending is None:
                pending = wav
                continue
            if pending.shape[-1] >= crossfade_samples and wav.shape[-1] >= crossfade_samples:
                fade_out = torch.linspace(1.0, 0.0, crossfade_samples)
                fade_in  = torch.linspace(0.0, 1.0, crossfade_samples)
                overlap  = pending[..., -crossfade_samples:] * fade_out + wav[..., :crossfade_samples] * fade_in
                write_part = torch.cat([pending[..., :-crossfade_samples], overlap], dim=-1)
                pending = wav[..., crossfade_samples:]
            else:
                write_part = pending
                pending = wav
            pcm = (write_part.squeeze(0) * 32767).clamp(-32768, 32767).to(torch.int16)
            out_wav.writeframes(pcm.numpy().tobytes())
        if pending is not None and pending.shape[-1] > 0:
            pcm = (pending.squeeze(0) * 32767).clamp(-32768, 32767).to(torch.int16)
            out_wav.writeframes(pcm.numpy().tobytes())


def wav_to_opus(wav_path, opus_path, bitrate=48):
    cmd = [
        'ffmpeg', '-y', '-i', wav_path,
        '-c:a', 'libopus', '-b:a', f'{bitrate}k',
        '-ac', '1', '-application', 'audio',
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
                # Move to CPU immediately to free VRAM before next yield
                audio_chunks.append(output['tts_speech'].cpu())
                torch.cuda.empty_cache()

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
    print(f"    All {max_retries} retries exhausted, skipping chunk")
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

    # Hierarchical text splitting: paragraph → sentence → ; → comma+conjunction → comma
    def _subsplit(segment):
        """Sub-split an oversized segment using progressively weaker boundaries."""
        if len(segment) <= 500:
            return [segment]
        if ';' in segment:
            parts = [p.strip() for p in segment.split(';') if p.strip()]
            if len(parts) > 1:
                result = []
                for p in parts:
                    result.extend(_subsplit(p))
                return result
        parts = re.split(r',\s*(?=(?:а|но|и|или|что|как|когда|где|если)\b)', segment)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            result = []
            for p in parts:
                result.extend(_subsplit(p))
            return result
        if ',' in segment:
            parts = [p.strip() for p in segment.split(',') if p.strip()]
            if len(parts) > 1:
                return parts
        return [segment]

    paragraphs = re.split(r'\n\n+', clean_text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?…])\s+', para)
        sentences = [s.strip() for s in sentences if s.strip()]
        current = ""
        for s in sentences:
            subs = _subsplit(s) if len(s) > 500 else [s]
            for sub in subs:
                if current and len(current) + len(sub) + 1 > 500:
                    chunks.append(current)
                    current = sub
                else:
                    current = (current + " " + sub).strip() if current else sub
        if current:
            chunks.append(current)

    # Merge chunks shorter than 150 chars with neighbour
    merged = []
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < 150:
            if i + 1 < len(chunks):
                chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                i += 1
            elif merged:
                merged[-1] = merged[-1] + " " + chunks[i]
                i += 1
            else:
                merged.append(chunks[i])
                i += 1
        else:
            merged.append(chunks[i])
            i += 1
    chunks = merged

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
    for i in tqdm(range(start_chunk, len(chunks)), initial=start_chunk, total=len(chunks), desc="  Chunks", unit="chunk"):
        chunk_text = chunks[i]
        temp_path = os.path.join(temp_dir, f"chunk_{i:05d}.wav")

        tqdm.write(f"  Chunk {i+1}/{len(chunks)} ({len(chunk_text)} chars): {chunk_text[:60]}...")
        t0 = time.time()

        audio = synthesize_chunk(cosyvoice, chunk_text, ref_text, ref_wav)
        if audio is not None and audio.shape[-1] > 0:
            torchaudio.save(temp_path, audio, cosyvoice.sample_rate)
            temp_wavs.append(temp_path)
            dur = audio.shape[-1] / cosyvoice.sample_rate
            tqdm.write(f"    Done: {dur:.1f}s in {time.time()-t0:.1f}s")
        else:
            failed_chunks.append(i)
            tqdm.write(f"    FAILED — no audio generated")

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
        shutil.copy2(temp_wavs[0], final_wav)
    else:
        load_and_merge_wavs(temp_wavs, final_wav, sr=cosyvoice.sample_rate)

    # Convert to opus
    print(f"  Converting to opus ({bitrate}kbps)...")
    wav_to_opus(final_wav, final_opus, bitrate)

    # Stats
    wav_mb = os.path.getsize(final_wav) / (1024 * 1024)
    if os.path.exists(final_opus) and os.path.getsize(final_opus) > 0:
        opus_mb = os.path.getsize(final_opus) / (1024 * 1024)
        print(f"  Result: WAV {wav_mb:.1f}MB → Opus {opus_mb:.1f}MB")
    else:
        print(f"  WARNING: Opus conversion failed, keeping WAV: {final_wav}")

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

    # Remove intermediate WAV only after confirmed successful opus conversion
    if os.path.exists(final_opus) and os.path.getsize(final_opus) > 0:
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
    ref_wav = trim_ref_audio(args.ref_wav, max_sec=22)
    ref_wav_is_temp = (ref_wav != args.ref_wav)

    try:
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
    finally:
        if ref_wav_is_temp:
            try:
                os.remove(ref_wav)
            except:
                pass

    print("All done.")


if __name__ == '__main__':
    main()
