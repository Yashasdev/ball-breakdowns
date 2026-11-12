"""Narration in Yashas's own cloned voice. Local only: it runs on his Mac, never on GitHub.

    $BK_OMNIVOICE_PYTHON -m breakdowns.yashas --lines lines.json --out DIR [--strict] [--seed N]

build.py runs this for `python -m breakdowns.build --voice yashas --only <ids>`. The engine is OmniVoice (k2-fsa;
code Apache-2.0, but the model weights are CC-BY-NC, so non-commercial use only), installed in its own Python. His
recordings must never be committed or uploaded (this repo is public), so the reference clip and its transcript come
from environment variables that point outside the repo:

    BK_VOICE_REF         a 3-10 s WAV of him speaking, outside this repo
    BK_VOICE_REF_TEXT    its exact transcript
    BK_VOICE_SIMILAR     optional: a longer recording of him (not the reference), used to keep the take that sounds
                         most like him; without it the first take that passes the word check is kept
    BK_OMNIVOICE_PYTHON  the Python that has omnivoice, faster-whisper and transformers (build.py uses it)

lines.json: [{"say": "...", "check": "..."}] ("check" is what the transcript should read when "say" has OmniVoice
pronunciation spans like [K AY1 M]; it defaults to "say"; "strict": true makes every word count for that line, to
confirm a hard word was said right). Every line gets TAKES takes; the kept one passes the transcript check
(heard_ok, and every word for --strict card lines), preferring takes where every word was heard right, then the one
most similar to him. If none passes, more takes up to MAX_TAKES. Writes lineNN.wav and meta.json like build.voice(),
plus takes.json (every take: seed, transcript, similarity, seconds, words a minute). Only the finished lines leave this
folder, inside a video.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from breakdowns.build import heard_ok, words

TAKES, MAX_TAKES = 3, 6
VOICE_ID = "yashas-omnivoice"  # what the published json and index say ("voice"), so the poster adds the credit
CREDIT = "AI voice: a clone of my voice (OmniVoice)."


def ref_from_env() -> tuple[str, str]:
    wav, text = os.environ.get("BK_VOICE_REF", ""), os.environ.get("BK_VOICE_REF_TEXT", "").strip()
    if not wav or not Path(wav).is_file() or not text:
        raise SystemExit("set BK_VOICE_REF (a WAV outside this repo) and BK_VOICE_REF_TEXT (its exact transcript)")
    if Path(wav).resolve().is_relative_to(Path(__file__).resolve().parent.parent):
        raise SystemExit("BK_VOICE_REF must be outside this repo: the repo is public")
    return wav, text


def plain(say: str) -> str:
    """The words of a line without OmniVoice's [CMU PHONEME] spans."""
    return re.sub(r"\s+", " ", re.sub(r"\[[^\]]*\]", "", say)).strip()


class Similarity:
    """Speaker similarity (WavLM x-vectors, cosine) to the mean of windows of a longer recording of the same person."""

    def __init__(self, recording: str):
        import numpy as np
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.np, self.torch = np, torch
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.m = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
        x = self.load16(recording)
        win = [x[a:a + 8 * 16000] for a in range(0, len(x) - 4 * 16000, 8 * 16000)]
        win = [w for w in win if np.abs(w).mean() > 0.01]  # skip near-silent stretches
        c = np.mean([self.emb(w) for w in win], 0)
        self.centroid = c / np.linalg.norm(c)

    def load16(self, path: str):
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32")
        x = x if x.ndim == 1 else x.mean(1)
        return self.np.interp(self.np.arange(0, len(x), sr / 16000), self.np.arange(len(x)), x).astype(self.np.float32)

    def emb(self, x):
        with self.torch.no_grad():
            e = self.m(**self.fe(x, sampling_rate=16000, return_tensors="pt")).embeddings[0]
        return self.torch.nn.functional.normalize(e, dim=-1).numpy()

    def __call__(self, path: str) -> float:
        return float(self.emb(self.load16(path)) @ self.centroid)


def expressiveness(path: str) -> dict:
    """How lively a take sounds: pitch range (standard deviation of the voiced pitch, in semitones) and loudness
    variation (standard deviation of the voiced frames' level, in dB); "score" = semitones + dB / 3."""
    import librosa
    import numpy as np
    y, sr = librosa.load(path, sr=16000)
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr, frame_length=1024, hop_length=160)
    db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=1024, hop_length=160)[0] + 1e-6)
    n = min(len(f0), len(db))
    f0, voiced, db = f0[:n], voiced[:n], db[:n]
    p = f0[voiced & ~np.isnan(f0)]
    st = float(np.std(12 * np.log2(p / np.median(p)))) if len(p) > 5 else 0.0
    loud = float(np.std(db[voiced])) if voiced.sum() > 5 else 0.0
    return {"pitch_st": round(st, 2), "loud_db": round(loud, 2), "score": round(st + loud / 3, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strict", action="store_true", help="every word must match (short card lines)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--takes", type=int, default=TAKES)
    ap.add_argument("--speed", type=float, default=None, help="OmniVoice speaking speed (default: its own estimate)")
    ap.add_argument("--pick", choices=["similar", "expressive"], default="similar",
                    help="expressive: among takes heard right and within --window of the most similar one, keep the "
                         "liveliest (expressiveness())")
    ap.add_argument("--window", type=float, default=0.01)
    a = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import numpy as np
    import soundfile as sf
    import torch
    from faster_whisper import WhisperModel
    from omnivoice import OmniVoice

    ref_wav, ref_text = ref_from_env()
    lines = json.loads(Path(a.lines).read_text())
    out = Path(a.out)
    (out / "takes").mkdir(parents=True, exist_ok=True)
    whisper = WhisperModel("small", device="cpu", compute_type="int8")
    similar = Similarity(os.environ["BK_VOICE_SIMILAR"]) if os.environ.get("BK_VOICE_SIMILAR") else None
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=dev,
                                      dtype=torch.float16 if dev != "cpu" else torch.float32)
    prompt = model.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=ref_text)  # in memory only, never saved
    sr = model.sampling_rate

    meta, log = [], []
    for k, line in enumerate(lines):
        say, check = line["say"], line.get("check") or plain(line["say"])
        takes = []
        for t in range(MAX_TAKES):
            if t >= a.takes and any(x["ok"] for x in takes):
                break
            seed = a.seed * 1000 + k * 10 + t
            torch.manual_seed(seed)
            t0 = time.time()
            x = np.asarray(model.generate(text=say, language="en", voice_clone_prompt=prompt, speed=a.speed)[0],
                           np.float32)
            gen_s = time.time() - t0
            loud = np.flatnonzero(np.abs(x) > 0.02)
            if len(loud):  # the same trim as build.voice(): silence off both ends, a hair of air kept
                x = x[max(0, loud[0] - 800): min(len(x), loud[-1] + 2400)]
            path = out / "takes" / f"line{k:02d}_t{t}.wav"
            sf.write(path, x, sr, subtype="PCM_16")
            x16 = np.interp(np.arange(0, len(x), sr / 16000), np.arange(len(x)), x).astype(np.float32)
            segs, _ = whisper.transcribe(x16, language="en")
            heard = " ".join(s.text for s in segs).strip()
            ok, why = heard_ok(check, heard)
            if (a.strict or line.get("strict")) and ok and words(check) != words(heard):
                ok, why = False, "a word differs"
            sim = similar(str(path)) if similar else None
            expr = expressiveness(str(path)) if a.pick == "expressive" else None
            dur = len(x) / sr
            takes.append({"take": t, "seed": seed, "ok": ok, "exact": words(check) == words(heard), "heard": heard,
                          "why": why, "similarity": sim, "expressive": expr,
                          "seconds": round(dur, 3), "wpm": round(len(check.split()) / dur * 60), "gen_s": round(gen_s, 1)})
            print(f"  line {k} take {t + 1}: {'ok' if ok else 'redo'} sim {sim if sim is None else round(sim, 4)} "
                  + (f"expr {expr['score']} ({expr['pitch_st']} st, {expr['loud_db']} dB) " if expr else "")
                  + f"{dur:.2f}s | {heard}" + (f" | {why}" if why else ""), flush=True)
        passed = [x for x in takes if x["ok"]]
        # a take whose every word was heard right beats one that only passes the letter match ("before few feel the
        # burn" for "before you feel the burn"); then the one that sounds most like him
        best = max(passed, key=lambda x: (x["exact"], x["similarity"] or 0)) if passed else None
        if best and a.pick == "expressive":  # the liveliest of the takes that sound about as much like him
            close = [x for x in passed if x["exact"] == best["exact"]
                     and (x["similarity"] or 0) >= (best["similarity"] or 0) - a.window]
            best = max(close, key=lambda x: x["expressive"]["score"])
        log.append({"say": say, "check": check, "kept": None if best is None else best["take"], "takes": takes})
        (out / "takes.json").write_text(json.dumps(log, indent=1))
        if best is None:
            print(f"::error::line {k} never matched the script in {len(takes)} takes: {say}")
            sys.exit(2)
        keep = out / f"line{k:02d}.wav"
        keep.write_bytes((out / "takes" / f"line{k:02d}_t{best['take']}.wav").read_bytes())
        meta.append({"file": keep.name, "seconds": best["seconds"]})
        print(f"  line {k}: kept take {best['take'] + 1} of {len(takes)}", flush=True)
    (out / "meta.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    main()
