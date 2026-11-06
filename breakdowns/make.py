"""How Crazy Was It? Ball Knowledge's breakdowns of famous games.

    python -m breakdowns.make crazy-01 --voice-dir DIR     # render, using lines from breakdowns.voice

One focal thing on screen at a time (a giant scoreboard, a stat, a checklist, a play diagram, a debate card), a
TV-style score bug, and bold captions timed to the narrator's actual pauses. The page (template.html) is drawn by
headless Chrome through Playwright, one screenshot per frame; FFmpeg joins them. The music is original
(bk/music.py, composed by code, nothing sampled) and every line starts on a beat. Stories are in items.json, every
fact checked against a box score or play-by-play (each item lists its sources).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from bk import music, sound

ROOT = Path(__file__).resolve().parent.parent
ITEMS = ROOT / "items.json"
TEMPLATE = Path(__file__).resolve().parent / "template.html"
FPS, W, H, SR = 30, 1080, 1920, sound.SR
LEAD_IN, GAP, TAIL, COVER_TAIL = 0.25, 0.08, 1.2, 0.5
STYLES_BY_SPORT = {"nfl": ["stadium", "drumline"], "cfb": ["drumline", "stadium"], "nba": ["phonk", "drill", "trap"]}


def track_for(it: dict) -> str:
    """A different original track for each story, cycling through its sport's styles (bk/music.py, 5 tracks each)."""
    styles = STYLES_BY_SPORT.get(it["sport"], ["stadium"])
    options = [f"{s}-{k}" for k in range(1, 6) for s in styles]
    return options[int(it["id"].split("-")[-1]) % len(options)]


def item(item_id: str) -> dict:
    return next(i for i in json.loads(ITEMS.read_text()) if i["id"] == item_id)


def ffmpeg() -> str:
    """The system's FFmpeg, or the copy bundled with imageio-ffmpeg."""
    return shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()


def load(path: Path) -> np.ndarray:
    v, sr = sf.read(path, dtype="float32")
    v = v if v.ndim == 1 else v.mean(axis=1)
    if sr != SR:
        v = np.interp(np.arange(0, len(v), sr / SR), np.arange(len(v)), v).astype(np.float32)
    return v


def words(cap: str, start: float, voice: np.ndarray) -> list[dict]:
    """Caption words timed to the speech: each word lands at the same share of the *voiced* time as its share of
    the caption's letters, so pauses ("he just... keeps... running") hold the captions too."""
    toks = re.findall(r"\*[^*]+\*|\S+", cap)
    out = []
    for tok in toks:
        acc = tok.startswith("*")
        for w in (tok.strip("*").split() if acc else [tok]):
            out.append({"w": w, "acc": acc})
    frame = int(SR * 0.02)
    env = np.array([np.abs(voice[i:i + frame]).mean() for i in range(0, len(voice), frame)])
    voiced = env > max(0.01, env.max() * 0.08)
    cum = np.cumsum(voiced) / max(1, voiced.sum())
    weights = np.array([len(w["w"]) + 1 for w in out], float)
    edges = np.concatenate([[0], np.cumsum(weights) / weights.sum()])[:-1]
    for w, e in zip(out, edges):
        k = int(np.searchsorted(cum, e * 0.97))
        w["t"] = round(start + k * 0.02, 3)
    return out


def timeline(it: dict, voices: list[np.ndarray], track: str) -> dict:
    half = 60 / music.bpm(track) / 2
    t, beats = LEAD_IN, []
    for b, v in zip(it["beats"], voices):
        start = float(np.ceil(t / half - 1e-6) * half)
        end = start + len(v) / SR
        beats.append({**b, "start": round(start, 3), "end": round(end, 3), "words": words(b["cap"], start, v)})
        t = end + GAP
    return {"kicker": it["kicker"], "home": it["home"], "away": it["away"], "beats": beats,
            "total": round(beats[-1]["end"] + TAIL, 3)}


def soundtrack(tl: dict, voices: list[np.ndarray], track: str, path: Path) -> None:
    total = tl["total"] + COVER_TAIL
    n = int(total * SR) + SR
    buf = np.zeros(n, np.float32)
    voiced = np.zeros(n, bool)
    rng = np.random.default_rng(3)

    def add(x, at, gain):
        x = x if x.ndim == 1 else x.mean(axis=1)
        i = int(at * SR)
        k = min(len(x), n - i)
        if k > 0:
            buf[i:i + k] += x[:k] * gain

    for b, v in zip(tl["beats"], voices):
        add(v / (np.abs(v).max() or 1) * 0.9, b["start"], 1.0)
        voiced[int(b["start"] * SR): int(b["start"] * SR) + len(v)] = True
        add(sound.whoosh(rng, 0.3), max(0, b["start"] - 0.12), 0.22)   # every scene change whooshes in
        if b.get("flash"):
            add(sound.impact(rng), b["start"], 0.6)
            add(sound.bass_drop(), b["start"], 0.5)
        if b["scene"].get("stamp"):
            add(sound.impact(rng), b["start"] + 0.6, 0.5)
    prev = None
    for b in tl["beats"]:
        for k, s in enumerate(b.get("steps") or ([b["score"]] if b.get("score") else [])):
            if s != prev:
                add(sound.pop(1.0 + 0.1 * k), b["start"] + k * (b["end"] - b["start"]) / max(1, len(b.get("steps") or [1])) + 0.15, 0.35)
            prev = s
    bed = music.render(track, total)[:n].mean(axis=1)
    duck = np.convolve(np.where(voiced[: len(bed)], 0.30, 1.0), np.ones(4410) / 4410, mode="same")
    buf[: len(bed)] += (bed * duck * 0.26).astype(np.float32)
    buf = buf[: int(total * SR)]
    sf.write(path, buf / (np.abs(buf).max() or 1) * 0.9, SR)


def frames(tl: dict, out_dir: Path, cover_at: float) -> None:
    from playwright.sync_api import sync_playwright
    n = int(tl["total"] * FPS)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": W, "height": H})
        page.goto(TEMPLATE.as_uri())
        page.evaluate("document.fonts.ready")
        page.evaluate("tl => setup(tl)", tl)
        for i in range(n):
            page.evaluate(f"render({i / FPS})")
            page.screenshot(path=str(out_dir / f"{i:05d}.jpg"), type="jpeg", quality=92)
        page.evaluate(f"render({cover_at})")
        page.screenshot(path=str(out_dir / "cover.jpg"), type="jpeg", quality=95)
        browser.close()
    for j in range(int(COVER_TAIL * FPS)):
        shutil.copy(out_dir / "cover.jpg", out_dir / f"{n + j:05d}.jpg")


def render(item_id: str, voice_dir: Path, out: Path, track: str | None = None) -> dict:
    it = item(item_id)
    track = track or track_for(it)
    meta = json.loads((voice_dir / "meta.json").read_text())
    voices = [load(voice_dir / m["file"]) for m in meta]
    tl = timeline(it, voices, track)
    work = Path(tempfile.mkdtemp(prefix="crazy-"))
    (work / "frames").mkdir()
    frames(tl, work / "frames", cover_at=tl["beats"][0]["end"] - 0.05)  # the cover: the hook's big scoreboard
    soundtrack(tl, voices, track, work / "sound.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(work / "frames" / "%05d.jpg"),
                    "-i", str(work / "sound.wav"), "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:v", "libx264", "-crf", "19",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest",
                    "-movflags", "+faststart", str(out)], check=True)
    shutil.copy(work / "frames" / "cover.jpg", out.with_suffix(".jpg"))
    shutil.rmtree(work, ignore_errors=True)
    return {"video": str(out), "seconds": round(tl["total"] + COVER_TAIL, 2), "music": track}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--voice-dir", required=True)
    ap.add_argument("--music")
    a = ap.parse_args()
    print(json.dumps(render(a.id, Path(a.voice_dir), ROOT / "out" / f"{a.id}.mp4", a.music)))


if __name__ == "__main__":
    main()
