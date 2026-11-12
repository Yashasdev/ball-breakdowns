"""The breakdown narrator: Chatterbox (Resemble AI, MIT license) speaking with hype, in the voice of a reference clip.

    python -m breakdowns.voice crazy-01 OUT_DIR

The reference clip
(assets/voice/ref_onyx.wav) was spoken by Kokoro's am_onyx voice (Apache 2.0), so no person's voice is copied. Writes one WAV per
beat (line00.wav ...) and meta.json with each line's length.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ITEMS = ROOT / "items.json"
REF = ROOT / "assets" / "voice" / "ref_onyx.wav"
EXAGGERATION, CFG = 1.0, 0.3   # chosen by Yashas on Oct 31, 2026 ("E ... am_onyx")
# Items marked "voice": "yashas" in items.json / cards.json are narrated in a clone of Yashas's own voice, on his Mac
# only (breakdowns/yashas.py; his recordings never enter this public repo). GitHub's build can't voice them, so it
# never builds them and leaves the published videos alone. CLONED: what their published json and index entry say
# ("voice"), so the poster adds the caption credit.
CLONED = {"yashas": "yashas-omnivoice"}


def local_only(item: dict) -> bool:
    """Voiced on Yashas's Mac, never by the GitHub build."""
    return item.get("voice") in CLONED


def main() -> None:
    import torch
    import torchaudio
    from chatterbox.tts import ChatterboxTTS

    item_id, out = sys.argv[1], Path(sys.argv[2])
    ref = str(REF)
    item = next(i for i in json.loads(ITEMS.read_text()) if i["id"] == item_id)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(item_id.split("-")[-1]))  # the same item always sounds the same
    model = ChatterboxTTS.from_pretrained(device="cpu")
    meta = []
    for k, beat in enumerate(item["beats"]):
        wav = model.generate(beat["say"], audio_prompt_path=ref, exaggeration=EXAGGERATION, cfg_weight=CFG)
        x = wav[0]
        loud = (x.abs() > 0.02).nonzero()
        if len(loud):  # trim silence at both ends, keep a hair of air
            a, b = max(0, int(loud[0]) - 800), min(len(x), int(loud[-1]) + 2400)
            x = x[a:b]
        path = out / f"line{k:02d}.wav"
        torchaudio.save(str(path), x.unsqueeze(0), model.sr)
        meta.append({"file": path.name, "seconds": round(len(x) / model.sr, 3)})
        print(f"{item_id} line {k}: {meta[-1]['seconds']}s", flush=True)
    (out / "meta.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    main()
