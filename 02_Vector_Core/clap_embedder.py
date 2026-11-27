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

# Universal 26-family global genre hierarchy (synthesized from AllMusic, Apple Music, Spotify, and Deezer).
# Maps real-world provider/editorial tags -> macro-genres, then scopes CLAP acoustic subgenre tagging
# inside the track's macro family (promoting coarse Deezer 'Pop'/'Dance' into Dance-Pop, Traditional Pop, etc.).
MACRO_GENRES = (
    "pop",
    "dance pop",
    "traditional pop",
    "rock",
    "alternative / indie",
    "metal",
    "punk",
    "hip hop",
    "r&b",
    "soul / funk",
    "electronic / edm",
    "ambient / lo-fi",
    "jazz",
    "blues",
    "classical",
    "soundtrack",
    "folk / singer-songwriter",
    "country",
    "latin",
    "reggae / caribbean",
    "k-pop",
    "j-pop / city pop",
    "afrobeats / african",
    "indian / bollywood",
    "world / traditional",
    "comedy / novelty",
)

# Ordered specific-first keyword rules for mapping arbitrary provider/editorial tags -> macro-genre.
MACRO_GENRE_KEYWORDS = (
    ("comedy / novelty", ("comedy", "parody", "novelty", "spoken word", "spoken", "humour", "humor")),
    ("metal", (
        "metal", "metalcore", "deathcore", "grindcore", "thrash", "doom", "sludge",
        "djent", "nu-metal", "black metal", "death metal", "power metal",
    )),
    ("punk", (
        "punk", "hardcore", "emo", "screamo", "post-punk", "goth rock", "skate punk",
    )),
    ("hip hop", (
        "hip hop", "hip-hop", "rap", "trap", "drill", "grime", "boom bap", "phonk",
        "gangsta", "conscious hip", "g-funk", "crunk", "cloud rap",
    )),
    ("traditional pop", (
        "traditional pop", "adult standards", "crooner", "easy listening", "lounge",
        "doo-wop", "doo wop", "tin pan alley", "torch song",
    )),
    ("jazz", (
        "jazz", "swing", "big band", "bossa nova", "bebop", "hard bop", "cool jazz",
        "dixieland", "ragtime", "manouche", "gypsy jazz", "jazz fusion",
    )),
    ("blues", (
        "blues", "delta blues", "chicago blues", "electric blues", "boogie-woogie",
    )),
    ("classical", (
        "classical", "orchestra", "symphony", "symphonic", "baroque", "opera", "chamber",
        "sonata", "concerto", "choral", "choir", "piano instrumental", "solo piano",
        "romantic era", "neoclassical", "contemporary classical",
    )),
    ("soundtrack", (
        "soundtrack", "film score", "films/games", "movie score", "ost", "video game",
        "vgm", "musical", "broadway", "show tunes", "disney", "anime", "kids",
    )),
    ("ambient / lo-fi", (
        "ambient", "lo-fi", "lofi", "chillout", "chillstep", "new age", "drone",
        "meditation", "downtempo", "trip-hop", "trip hop", "sleep",
    )),
    ("dance pop", (
        "dance pop", "dance-pop", "synthpop", "synth-pop", "electropop", "europop",
        "eurodance", "new wave", "hyperpop", "bubblegum", "disco pop",
    )),
    ("electronic / edm", (
        "electronic", "electro", "edm", "house", "techno", "trance", "dubstep",
        "drum and bass", "drum & bass", "dnb", "jungle", "uk garage", "synthwave",
        "retrowave", "vaporwave", "dance", "club", "idm", "breakbeat", "hardstyle",
        "future bass", "glitch",
    )),
    ("soul / funk", (
        "soul", "neo-soul", "neo soul", "motown", "funk", "disco", "nu-disco",
        "boogie", "p-funk", "gospel", "christian",
    )),
    ("r&b", (
        "r&b", "rnb", "rhythm and blues", "quiet storm", "new jack swing", "contemporary r&b",
    )),
    ("reggae / caribbean", (
        "reggae", "dub", "ska", "rocksteady", "dancehall", "soca", "calypso", "caribbean",
    )),
    ("latin", (
        "latin", "reggaeton", "salsa", "bachata", "merengue", "cumbia", "mariachi",
        "norteno", "norteño", "banda", "ranchera", "corrido", "mexicano", "regional mexican",
        "tango", "flamenco", "bolero", "urbano", "tejano", "vallenato", "mpb",
        "samba", "pagode", "sertanejo", "brazilian", "forro", "forró", "choro",
    )),
    ("k-pop", ("k-pop", "kpop", "korean pop", "korean")),
    ("j-pop / city pop", ("j-pop", "jpop", "j-rock", "jrock", "city pop", "japanese", "shibuya-kei")),
    ("afrobeats / african", ("afrobeat", "afropop", "amapiano", "highlife", "soukous", "african")),
    ("indian / bollywood", ("bollywood", "hindi", "indian", "punjabi", "bhangra", "carnatic", "hindustani", "qawwali")),
    ("world / traditional", (
        "c-pop", "mandopop", "cantopop", "asian", "arabic", "middle eastern", "turkish",
        "persian", "celtic", "irish", "klezmer", "balkan", "nordic", "world",
        "traditional", "fado", "chanson",
    )),
    ("country", ("country", "bluegrass", "outlaw country", "honky tonk", "alt-country", "nashville")),
    ("folk / singer-songwriter", (
        "folk", "singer-songwriter", "singer songwriter", "singer & songwriter",
        "americana", "acoustic", "appalachian", "roots",
    )),
    ("alternative / indie", (
        "alternative", "indie", "shoegaze", "dream pop", "britpop", "grunge",
        "math rock", "post-rock", "noise pop", "chamber pop", "art rock",
    )),
    ("rock", (
        "rock", "classic rock", "hard rock", "psychedelic", "prog", "progressive rock",
        "glam rock", "southern rock", "garage rock", "surf rock", "rockabilly",
        "rock & roll", "rock and roll", "soft rock", "arena rock", "heartland rock", "yacht rock",
    )),
    ("pop", (
        "pop", "teen pop", "oldies", "ballad", "schlager", "art pop", "baroque pop",
    )),
)

# When a coarse editorial tag (e.g. Deezer 'Pop' or 'Dance') arrives, allow CLAP to score
# across closely related sibling macro-genres so tracks can promote cleanly to their true home.
COARSE_MACRO_EXPANSION = {
    "pop": ("pop", "dance pop", "traditional pop", "soul / funk"),
    "electronic / edm": ("electronic / edm", "dance pop", "ambient / lo-fi"),
    "alternative / indie": ("alternative / indie", "punk", "rock"),
    "soundtrack": ("soundtrack", "classical", "ambient / lo-fi"),
    "r&b": ("r&b", "soul / funk"),
}

# Concrete acoustic leaf subgenres grouped by universal macro-genre.
SUBGENRE_HIERARCHY = {
    "pop": [
        ("modern pop", "contemporary mainstream pop with polished vocals and catchy chorus"),
        ("pop ballad", "slow emotional vocal pop ballad with piano and sweeping strings"),
        ("art / chamber pop", "theatrical art pop with orchestral strings and quirky arrangement"),
        ("acoustic pop", "stripped acoustic pop with warm guitar and intimate vocals"),
    ],
    "dance pop": [
        ("dance pop", "high energy radio dance pop with four-on-the-floor beat and bright hook"),
        ("synthpop", "1980s synthpop and new wave with retro analog synthesizers and drum machine"),
        ("europop / disco pop", "upbeat catchy europop and disco pop with glossy keyboards and chorus"),
        ("electropop", "modern sleek electropop with pulsing synth bass and crisp electronic drums"),
    ],
    "traditional pop": [
        ("crooner standards", "vintage 1950s crooner standards with lush string orchestra"),
        ("doo-wop / oldies", "1950s doo-wop a cappella vocal group harmony"),
        ("easy listening", "classic lounge easy listening orchestral standard with brass"),
    ],
    "rock": [
        ("classic rock", "1970s classic rock with electric guitar riffs and live drums"),
        ("hard rock", "hard rock with heavy overdriven guitar riffs and driving drums"),
        ("rock & roll", "vintage 1950s rock and roll and rockabilly with twangy electric guitar"),
        ("psychedelic rock", "psychedelic rock with dreamy reverb guitars and swirling organ"),
        ("progressive rock", "progressive rock with complex time signatures and keyboards"),
        ("soft / yacht rock", "smooth 1970s soft rock with warm electric piano and clean guitar"),
    ],
    "alternative / indie": [
        ("indie rock", "alternative indie rock band with jangly guitars and steady drums"),
        ("indie pop", "soft dreamy indie bedroom pop with gentle guitars and synths"),
        ("grunge", "raw 1990s grunge rock with fuzzy distorted guitars"),
        ("shoegaze / dream pop", "hazy shoegaze dream pop with wall of reverb guitars"),
        ("britpop", "anthemic 1990s britpop guitar rock with bright melodies"),
    ],
    "metal": [
        ("heavy metal", "classic heavy metal with distorted guitar riffs and double bass drums"),
        ("thrash / extreme metal", "fast aggressive thrash metal with rapid palm-muted guitars"),
        ("metalcore / modern metal", "heavy modern metalcore with breakdown riffs and intense vocals"),
        ("doom / stoner metal", "slow heavy doom metal with thick fuzzy distorted guitar"),
    ],
    "punk": [
        ("punk rock", "fast raw punk rock with energetic power chords and shouted vocals"),
        ("pop punk", "upbeat melodic pop punk with driving guitars and youthful chorus"),
        ("post-punk", "dark angular post-punk with prominent bassline and brooding baritone"),
        ("emo / alt punk", "emotional emo punk rock with dynamic quiet-loud guitars"),
    ],
    "hip hop": [
        ("classic hip hop", "1990s boom bap hip hop with sampled drums and rhythmic rap"),
        ("trap", "modern trap hip hop with deep 808 sub bass and rapid hi-hats"),
        ("melodic rap", "melodic rap hip hop with sung hooks and smooth beat"),
        ("uk rap / grime", "british uk rap and grime with gritty rhythmic cadence and bass"),
        ("alt hip hop", "experimental alternative hip hop with eclectic jazz or synth samples"),
        ("pop rap", "upbeat party rap hip hop with bright brass or synth chorus"),
    ],
    "r&b": [
        ("contemporary r&b", "smooth modern r&b with silky vocals and slow electronic groove"),
        ("r&b ballad", "slow romantic r&b love ballad with lush vocal harmonies"),
        ("alt r&b", "moody atmospheric alternative r&b with deep bass and falsetto"),
    ],
    "soul / funk": [
        ("classic soul", "vintage 1960s motown soul with warm horns and passionate vocals"),
        ("neo-soul", "warm laid-back neo-soul with electric rhodes piano and pocket drums"),
        ("classic funk", "1970s funk with slap electric bass, tight drums, and brass stabs"),
        ("disco", "upbeat 1970s disco dance track with sweeping strings and four-on-the-floor"),
        ("gospel / choral", "uplifting gospel soul with powerful choir harmonies and organ"),
    ],
    "electronic / edm": [
        ("house", "club house dance music with steady four-on-the-floor kick and groove"),
        ("eurodance / edm", "high energy festival edm with bright synth leads and drop"),
        ("techno / trance", "driving hypnotic techno and trance with pulsing bass and arpeggios"),
        ("drum & bass", "fast breakbeat drum and bass with rapid snare patterns and sub bass"),
        ("dubstep / bass", "heavy bass electronic dubstep with wobble synths and half-time beat"),
        ("synthwave", "retro 1980s analog synthwave electronic with nostalgic neon synths"),
    ],
    "ambient / lo-fi": [
        ("chill / lo-fi", "relaxing lo-fi hip hop instrumental beats with dusty vinyl crackle"),
        ("ambient", "calm atmospheric ambient drone and soft floating soundscapes"),
        ("downtempo / trip-hop", "moody downtempo trip-hop with slow breakbeats and atmospheric pads"),
    ],
    "jazz": [
        ("vocal jazz", "classic vocal jazz standard with warm crooning vocals and upright bass"),
        ("swing / big band", "upbeat big band swing jazz with brass horn section and ride cymbal"),
        ("bossa nova", "soft brazilian bossa nova jazz with nylon acoustic guitar and gentle brushes"),
        ("bebop / hard bop", "instrumental jazz quartet with piano, saxophone, trumpet, and fast swing"),
        ("smooth jazz", "mellow smooth jazz with clean saxophone or hollow-body guitar"),
        ("jazz fusion", "electric jazz fusion with intricate keyboards, bass guitar, and drums"),
    ],
    "blues": [
        ("electric blues", "chicago electric blues with expressive bent guitar solos and shuffle beat"),
        ("acoustic delta blues", "raw acoustic delta blues with slide guitar and gritty vocals"),
        ("blues rock", "driving blues rock with crunchy electric guitar riffs"),
        ("soul blues", "warm soul blues with organ, horns, and heartfelt vocals"),
    ],
    "classical": [
        ("orchestral", "symphonic classical orchestra with sweeping strings, woodwinds, and brass"),
        ("solo piano", "solo acoustic classical piano instrumental piece"),
        ("chamber / strings", "intimate classical chamber music with violin, viola, and cello"),
        ("opera / choral", "dramatic classical opera with operatic vocals and choir"),
        ("baroque", "baroque classical music with harpsichord and crisp string ensemble"),
    ],
    "soundtrack": [
        ("film score", "cinematic orchestral movie soundtrack score with dramatic strings"),
        ("video game ost", "melodic video game soundtrack with synth chiptune or orchestral motifs"),
        ("musical theater", "theatrical broadway musical show tune with expressive stage vocals"),
        ("dark cinematic", "dark moody atmospheric cinematic score with brooding bass"),
    ],
    "folk / singer-songwriter": [
        ("indie folk", "intimate acoustic singer-songwriter folk with fingerpicked guitar"),
        ("americana", "warm americana folk with acoustic guitar, harmonica, and storytelling"),
        ("acoustic ballad", "stripped acoustic guitar vocal ballad"),
    ],
    "country": [
        ("traditional country", "classic country music with acoustic guitar, pedal steel, and twang"),
        ("modern country", "polished modern country rock with crisp drums and storytelling vocals"),
        ("bluegrass", "fast acoustic bluegrass with banjo, mandolin, and fiddle"),
    ],
    "latin": [
        ("reggaeton", "reggaeton latin urban track with syncopated dembow drum rhythm"),
        ("salsa / merengue", "upbeat latin salsa and merengue with brass horns and timbales"),
        ("bachata / cumbia", "rhythmic bachata or cumbia with bright guitar and tropical percussion"),
        ("latin pop", "melodic spanish latin pop with acoustic guitar and dance groove"),
        ("brazilian / samba", "brazilian samba and mpb with acoustic guitar and surdo percussion"),
        ("regional mexican", "mexican mariachi or norteno with accordion, trumpets, and acoustic bass"),
    ],
    "reggae / caribbean": [
        ("roots reggae", "laid-back jamaican roots reggae with offbeat guitar skank and deep bass"),
        ("dub", "spacious dub reggae with heavy echo reverb and hypnotic bassline"),
        ("dancehall", "energetic caribbean dancehall with digital riddim and rhythmic toasting"),
        ("ska / rocksteady", "upbeat ska and rocksteady with brass horns and choppy offbeat guitar"),
    ],
    "k-pop": [
        ("k-pop dance", "glossy korean pop dance track with sharp electronic production"),
        ("k-pop ballad", "emotional korean pop vocal ballad with piano and strings"),
    ],
    "j-pop / city pop": [
        ("city pop", "japanese 1980s city pop with funky slap bass and glossy urban synths"),
        ("j-pop / j-rock", "energetic japanese rock and anime pop with fast melodic electric guitars"),
    ],
    "afrobeats / african": [
        ("afrobeats", "groovy west african afrobeats with syncopated percussion and smooth vocals"),
        ("amapiano", "south african amapiano house with deep log drum bassline"),
    ],
    "indian / bollywood": [
        ("bollywood", "indian bollywood film music with expressive vocals, tabla, and strings"),
        ("punjabi / bhangra", "upbeat punjabi bhangra with dhol drums and energetic vocals"),
    ],
    "world / traditional": [
        ("celtic / european folk", "traditional celtic or european folk with fiddle, accordion, and flute"),
        ("middle eastern / arabic", "middle eastern arabic music with oud, darbuka percussion, and ornate melody"),
        ("chanson / fado", "intimate european chanson or fado with acoustic guitar and accordion"),
    ],
    "comedy / novelty": [
        ("comedy rap / parody", "playful comedy rap or musical parody with witty spoken vocals"),
        ("musical comedy", "theatrical comedic song with humorous lyrics and upbeat accompaniment"),
    ],
}

# Flat exports preserved for backward compatibility
TAG_BANK = [prompt for subs in SUBGENRE_HIERARCHY.values() for _, prompt in subs]
_TAG_DISPLAY_LABELS = {prompt: label for subs in SUBGENRE_HIERARCHY.values() for label, prompt in subs}
_SUBGENRE_TO_MACRO = {label: macro for macro, subs in SUBGENRE_HIERARCHY.items() for label, _ in subs}

_CLAP_LOCK = threading.Lock()
_CACHED_CLAP = None
_CACHED_TAG_MATRIX = None
_CACHED_SUB_MATRICES = None


def map_tags_to_macro_genres(raw_tags):
    """Map arbitrary provider/editorial genre strings to ordered unique universal macro-genres."""
    if not raw_tags:
        return []
    if isinstance(raw_tags, str):
        items = [t.strip() for t in raw_tags.split(",") if t.strip()]
    else:
        items = [str(t or "").strip() for t in raw_tags if str(t or "").strip()]

    matched = []
    seen = set()
    for tag in items:
        low = tag.lower()
        if low in MACRO_GENRES and low not in seen:
            seen.add(low)
            matched.append(low)
            continue
        if low in _SUBGENRE_TO_MACRO:
            parent = _SUBGENRE_TO_MACRO[low]
            if parent not in seen:
                seen.add(parent)
                matched.append(parent)
            continue
        for macro, keywords in MACRO_GENRE_KEYWORDS:
            if any(kw in low for kw in keywords):
                if macro not in seen:
                    seen.add(macro)
                    matched.append(macro)
                break
    return matched


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


def _ensure_tag_matrices(processor, model, device=None):
    global _CACHED_TAG_MATRIX, _CACHED_SUB_MATRICES
    with _CLAP_LOCK:
        if _CACHED_TAG_MATRIX is None or _CACHED_SUB_MATRICES is None:
            full_mat = embed_texts_clap(TAG_BANK, processor, model, device=device)
            _CACHED_TAG_MATRIX = full_mat
            sub_map = {}
            cursor = 0
            for macro, subs in SUBGENRE_HIERARCHY.items():
                n_sub = len(subs)
                sub_map[macro] = (
                    [label for label, _ in subs],
                    full_mat[cursor : cursor + n_sub],
                )
                cursor += n_sub
            _CACHED_SUB_MATRICES = sub_map
        return _CACHED_TAG_MATRIX, _CACHED_SUB_MATRICES


def tag_audio_embeddings(audio_vecs, processor, model, device=None, top_k=3, min_score=0.18, macro_hints=None):
    """
    Zero-shot tag 512-D CLAP audio vectors using the 2-level global genre hierarchy.
    When macro_hints[i] is provided (e.g. from Spotify or Deezer editorial genres),
    CLAP scores ONLY subgenres inside those macro-genre families to prevent cross-family leakage.
    Otherwise, it falls back to bottom-up rollup from all leaf subgenres -> parent macro-genre.
    """
    if audio_vecs is None or len(audio_vecs) == 0:
        return []

    mat = np.asarray(audio_vecs, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)

    full_mat, sub_map = _ensure_tag_matrices(processor, model, device=device)
    all_leaf_labels = [_TAG_DISPLAY_LABELS[p] for p in TAG_BANK]

    results = []
    for i, vec in enumerate(mat):
        hints = []
        if macro_hints is not None and i < len(macro_hints) and macro_hints[i]:
            raw_h = macro_hints[i]
            hints = [m for m in (raw_h if isinstance(raw_h, (list, tuple)) else [raw_h]) if m in sub_map]

        if hints:
            # Expand coarse umbrella hints (e.g. Deezer 'pop' -> pop + dance pop + traditional pop)
            # so CLAP promotes each track into its true acoustic macro-genre without cross-family leaks.
            search_macros = []
            seen_m = set()
            for m in hints[:2]:
                for exp_m in COARSE_MACRO_EXPANSION.get(m, (m,)):
                    if exp_m in sub_map and exp_m not in seen_m:
                        seen_m.add(exp_m)
                        search_macros.append(exp_m)

            scored_subs = []
            for macro in search_macros:
                s_labels, s_mat = sub_map[macro]
                s_sims = vec @ s_mat.T
                for j, sc in enumerate(s_sims):
                    scored_subs.append((float(sc), s_labels[j], macro))
            scored_subs.sort(key=lambda x: x[0], reverse=True)

            winning_macro = scored_subs[0][2] if scored_subs else hints[0]
            tags = [winning_macro]
            for raw_sc, sl, _ in scored_subs:
                if sl not in tags and (raw_sc >= min_score or len(tags) < 2):
                    tags.append(sl)
                if len(tags) >= top_k:
                    break
            results.append(tags[:top_k])
        else:
            # Bottom-up fallback: find strongest concrete leaf subgenre, roll up to its macro parent
            sims = vec @ full_mat.T
            best_idx = int(np.argmax(sims))
            best_sub = all_leaf_labels[best_idx]
            parent_macro = _SUBGENRE_TO_MACRO.get(best_sub, "")
            tags = [parent_macro, best_sub] if parent_macro and parent_macro != best_sub else [best_sub]
            if parent_macro in sub_map and len(tags) < top_k:
                s_labels, s_mat = sub_map[parent_macro]
                s_sims = vec @ s_mat.T
                for j in np.argsort(s_sims)[::-1]:
                    sl = s_labels[int(j)]
                    if sl not in tags and float(s_sims[int(j)]) >= min_score:
                        tags.append(sl)
                    if len(tags) >= top_k:
                        break
            results.append(tags[:top_k])

    return results
