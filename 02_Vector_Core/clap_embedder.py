"""CLAP (laion/larger_clap_music) audio & text embedder + zero-shot musical tagger."""

import os
import sys
import threading

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config

CLAP_DIM = 512

TAG_BANK = [
    "classic rock with electric guitars",
    "hard rock heavy guitar riffs",
    "alternative indie rock band",
    "mainstream pop vocal music",
    "80s synthpop new wave",
    "hip hop rap beat",
    "melodic trap hip hop",
    "smooth r&b soul vocals",
    "funk disco bassline groove",
    "classic vocal jazz with brass and swing",
    "big band swing jazz",
    "bossa nova acoustic jazz",
    "japanese city pop funk",
    "j-pop energetic rock",
    "bollywood hindi film music",
    "classical orchestral strings",
    "solo classical piano piece",
    "acoustic folk singer songwriter",
    "electronic synth dance music",
    "house club dance beat",
    "ambient chill lo-fi beats",
    "latin merengue salsa percussion",
    "reggaeton latin urban",
    "country acoustic guitar",
    "blues guitar",
    "fast punk rock",
    "heavy metal distortion",
    "slow emotional ballad",
    "cinematic orchestral score",
    "dark moody atmospheric",
]

_TAG_DISPLAY_LABELS = {
    "classic rock with electric guitars": "classic rock",
    "hard rock heavy guitar riffs": "hard rock",
    "alternative indie rock band": "indie rock",
    "mainstream pop vocal music": "pop",
    "80s synthpop new wave": "synthpop",
    "hip hop rap beat": "hip hop",
    "melodic trap hip hop": "trap",
    "smooth r&b soul vocals": "r&b / soul",
    "funk disco bassline groove": "funk / disco",
    "classic vocal jazz with brass and swing": "vocal jazz",
    "big band swing jazz": "swing / big band",
    "bossa nova acoustic jazz": "bossa nova",
    "japanese city pop funk": "city pop",
    "j-pop energetic rock": "j-pop / j-rock",
    "bollywood hindi film music": "bollywood / hindi",
    "classical orchestral strings": "classical",
    "solo classical piano piece": "piano instrumental",
    "acoustic folk singer songwriter": "folk / acoustic",
    "electronic synth dance music": "electronic",
    "house club dance beat": "house",
    "ambient chill lo-fi beats": "chill / lo-fi",
    "latin merengue salsa percussion": "merengue / salsa",
    "reggaeton latin urban": "latin pop",
    "country acoustic guitar": "country",
    "blues guitar": "blues",
    "fast punk rock": "punk",
    "heavy metal distortion": "metal",
    "slow emotional ballad": "ballad",
    "cinematic orchestral score": "soundtrack",
    "dark moody atmospheric": "dark atmospheric",
}

_CLAP_LOCK = threading.Lock()
_CACHED_CLAP = None
_CACHED_TAG_MATRIX = None


def _extract_feature_tensor(output):
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "audio_embeds") and output.audio_embeds is not None:
        return output.audio_embeds
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"Unsupported CLAP feature output type: {type(output)}")


def get_clap_model(model_id=None):
    """Load and cache CLAP processor and model (CUDA if available, else CPU)."""
    global _CACHED_CLAP
    import torch
    from transformers import ClapModel, ClapProcessor

    target_id = model_id or getattr(config, "CLAP_MODEL_ID", "laion/clap-htsat-unfused")
    with _CLAP_LOCK:
        if _CACHED_CLAP is not None and _CACHED_CLAP[0] == target_id:
            return _CACHED_CLAP[1], _CACHED_CLAP[2], _CACHED_CLAP[3]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[CLAP] Loading {target_id} on {device.upper()}...")
        processor = ClapProcessor.from_pretrained(target_id)
        model = ClapModel.from_pretrained(target_id)
        model.to(device)
        model.eval()
        model.is_clap = True
        model._clap_device = device
        model.is_static_npu = False
        _CACHED_CLAP = (target_id, processor, model, device)
        return processor, model, device


def _slice_audio_windows(waveform_48k, sr=48000, window_sec=10.0, max_windows=3):
    wav = np.asarray(waveform_48k, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=0)
    win_samples = int(sr * window_sec)
    n = len(wav)
    if n <= win_samples:
        return [wav]

    total_possible = max(1, n // win_samples)
    if total_possible <= max_windows:
        starts = [i * win_samples for i in range(total_possible)]
        if n - starts[-1] > int(sr * 2.0) and len(starts) < max_windows and (n - win_samples) > starts[-1]:
            starts.append(n - win_samples)
    else:
        starts = np.linspace(0, n - win_samples, num=max_windows, dtype=int).tolist()

    return [wav[s : s + win_samples] for s in starts]


def embed_audio_waveform(waveform_48k, processor, model, device="cpu", max_windows=3, sr=48000):
    """Embed a 48kHz mono waveform by averaging 10s CLAP windows into one L2-normalized 512-D vector."""
    import torch

    chunks = _slice_audio_windows(waveform_48k, sr=sr, window_sec=10.0, max_windows=max_windows)
    inputs = processor(audio=chunks, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        raw = model.get_audio_features(**inputs)
        feats = _extract_feature_tensor(raw)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        mean_vec = feats.mean(dim=0, keepdim=True)
        mean_vec = torch.nn.functional.normalize(mean_vec, p=2, dim=1)
    return mean_vec[0].detach().cpu().numpy().astype(np.float32)


def embed_texts_clap(texts, processor, model, device=None, batch_size=32):
    """Embed text strings into the 512-D CLAP space."""
    import torch

    if not texts:
        return np.empty((0, CLAP_DIM), dtype=np.float32)

    dev = device or getattr(model, "_clap_device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
    chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [str(t or "") for t in texts[i : i + batch_size]]
            inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            raw = model.get_text_features(**inputs)
            feats = _extract_feature_tensor(raw)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            chunks.append(feats.detach().cpu().numpy().astype(np.float32))
    return np.vstack(chunks)


def tag_audio_embeddings(audio_vecs, processor, model, device=None, top_k=3, min_score=0.18):
    """Zero-shot tag 512-D CLAP audio vectors against TAG_BANK."""
    global _CACHED_TAG_MATRIX
    if audio_vecs is None or len(audio_vecs) == 0:
        return []

    mat = np.asarray(audio_vecs, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)

    with _CLAP_LOCK:
        if _CACHED_TAG_MATRIX is None:
            _CACHED_TAG_MATRIX = embed_texts_clap(TAG_BANK, processor, model, device=device)

    sims = mat @ _CACHED_TAG_MATRIX.T
    results = []
    for row in sims:
        ranked = np.argsort(row)[::-1][:top_k]
        tags = []
        for idx in ranked:
            if float(row[idx]) >= min_score or not tags:
                raw_tag = TAG_BANK[int(idx)]
                tags.append(_TAG_DISPLAY_LABELS.get(raw_tag, raw_tag))
        results.append(tags)
    return results
