"""Build every story in items.json that isn't published yet: voice it, check the voice, render it, publish it.

    python -m breakdowns.build [--max 12] [--dry]

Published videos live on this repo's GitHub release "videos" (public URLs, no keys): <id>.mp4, <id>.jpg (cover)
and <id>.json (what the poster needs), plus index.json listing everything published, in items.json order. The Ball
Knowledge poster in Yashasdev/act-reels reads index.json and posts the next video it hasn't posted.

The narrator sometimes slurs or drops a word, so every line is transcribed (faster-whisper) and compared with the
script; a line that doesn't match is spoken again with another seed, up to four times, or the story is skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ITEMS = ROOT / "items.json"
OUT = ROOT / "out"
RELEASE = "videos"
REPO = "Yashasdev/ball-breakdowns"
NUMBER_WORDS = set("""zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen
sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand point first
second third fourth fifth sixth seventh eighth ninth tenth""".split())


def words(text: str) -> list[str]:
    """Words to compare: lowercase, no punctuation, and no numbers (the transcriber writes 28 or 3rd for the words)."""
    out = re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("-", " ").replace("'", "")).split()
    return [w for w in out if w not in NUMBER_WORDS and not re.fullmatch(r"\d+(st|nd|rd|th|s)?", w)]


def heard_ok(say: str, heard: str) -> tuple[bool, str]:
    """Does the transcript match the script? Compared as letters, not words, so the transcriber's spelling choices
    ("back up" for backup, "28-3" for twenty-eight to three) don't count, while a dropped phrase, a stray sound before
    the line or a cut-off ending does."""
    from difflib import SequenceMatcher
    want, got = "".join(words(say)), "".join(words(heard))
    if not want:
        return True, ""
    ratio = lambda x, y: SequenceMatcher(None, x, y).ratio()  # noqa: E731
    whole, end = ratio(want, got), ratio(want[-8:], got[-8:])
    # where the line's opening letters turn up in the transcript: later than 1 letter in means a stray sound before it
    m = SequenceMatcher(None, want[:8], got[:24]).find_longest_match(0, 8, 0, 24)
    lead = m.b - m.a if m.size >= 3 else 99
    ok = whole >= 0.8 and lead <= 1 and end >= 0.6
    return ok, "" if ok else f"match {whole:.2f}, stray start {lead}, end {end:.2f}"


def released() -> set[str]:
    r = subprocess.run(["gh", "release", "view", RELEASE, "-R", REPO, "--json", "assets", "--jq", ".assets[].name"],
                       capture_output=True, text=True)
    if r.returncode:
        subprocess.run(["gh", "release", "create", RELEASE, "-R", REPO, "--title", "Published breakdown videos",
                        "--notes", "Rendered by the build workflow. index.json lists them in posting order."], check=True)
        return set()
    return set(r.stdout.split())


BUILD = 2  # raise to rebuild every published video (2: a different music track per story)


def story_hash(item: dict) -> str:
    """Changes whenever a story's script, captions or scenes change (or BUILD), so a fixed story is built again."""
    return hashlib.sha1(json.dumps([BUILD, item], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def published_index() -> dict[str, dict]:
    r = subprocess.run(["gh", "release", "download", RELEASE, "-R", REPO, "-p", "index.json", "-O", "-"],
                       capture_output=True, text=True)
    return {v["id"]: v for v in json.loads(r.stdout)} if r.returncode == 0 and r.stdout.strip() else {}


def voice(model, whisper, item: dict, out: Path) -> bool:
    import torch
    import torchaudio
    from breakdowns.voice import CFG, EXAGGERATION, REF
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for k, beat in enumerate(item["beats"]):
        for attempt in range(4):
            torch.manual_seed(int(item["id"].split("-")[-1]) * 1000 + k * 10 + attempt)
            x = model.generate(beat["say"], audio_prompt_path=str(REF), exaggeration=EXAGGERATION, cfg_weight=CFG)[0]
            loud = (x.abs() > 0.02).nonzero()
            if len(loud):
                x = x[max(0, int(loud[0]) - 800): min(len(x), int(loud[-1]) + 2400)]
            path = out / f"line{k:02d}.wav"
            torchaudio.save(str(path), x.unsqueeze(0), model.sr)
            segs, _ = whisper.transcribe(str(path), language="en")
            heard = " ".join(s.text for s in segs).strip()
            ok, why = heard_ok(beat["say"], heard)
            print(f"  line {k} try {attempt + 1}: {'ok' if ok else 'redo'} | {heard}" + (f" | {why}" if why else ""), flush=True)
            if ok:
                break
        else:
            print(f"::error::{item['id']} line {k} never matched the script; skipping this story")
            return False
        meta.append({"file": path.name, "seconds": round(len(x) / model.sr, 3)})
    (out / "meta.json").write_text(json.dumps(meta))
    return True


def post_info(item: dict, result: dict) -> dict:
    """What the poster needs: the caption's hook line and comment question, the length and the music."""
    return {"id": item["id"], "title": item["title"], "kicker": item["kicker"], "sport": item["sport"],
            "hook": item["beats"][0]["cap"].replace("*", ""), "ask": item["ask"],
            "seconds": result["seconds"], "music": result["music"], "sources": item["sources"], "hash": story_hash(item)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--dry", action="store_true", help="render only; publish nothing")
    a = ap.parse_args()
    items = json.loads(ITEMS.read_text())
    done = set() if a.dry else released()
    index = {} if a.dry else published_index()
    # new stories, and published ones whose script was fixed since (their hash changed)
    todo = [i for i in items if f"{i['id']}.mp4" not in done or index.get(i["id"], {}).get("hash") != story_hash(i)][: a.max]
    print(f"{len(items)} stories, {len(index)} published, building {len(todo)}: {[i['id'] for i in todo]}")
    if not todo:
        return
    from chatterbox.tts import ChatterboxTTS
    from faster_whisper import WhisperModel

    from breakdowns import make
    model = ChatterboxTTS.from_pretrained(device="cpu")
    whisper = WhisperModel("small", device="cpu", compute_type="int8")
    OUT.mkdir(exist_ok=True)
    built = []
    for it in todo:
        print(f"== {it['id']} {it['title']}", flush=True)
        vdir = Path(tempfile.mkdtemp()) / it["id"]
        if not voice(model, whisper, it, vdir):
            continue
        result = make.render(it["id"], vdir, OUT / f"{it['id']}.mp4")
        (OUT / f"{it['id']}.json").write_text(json.dumps(post_info(it, result), indent=1, ensure_ascii=False))
        if result["seconds"] > 59:
            print(f"::error::{it['id']} is {result['seconds']} s; Shorts must be under 60. Shorten its script.")
            continue
        print(f"   rendered {result['seconds']} s with {result['music']}", flush=True)
        built.append(it["id"])
        if not a.dry:
            subprocess.run(["gh", "release", "upload", RELEASE, "-R", REPO, "--clobber",
                            *(str(OUT / f"{it['id']}{ext}") for ext in (".mp4", ".jpg", ".json"))], check=True)
            index[it["id"]] = post_info(it, result)
            publish_index(items, index)  # after every video, so a run cut short keeps what it finished
    print(f"built {built}" + ("" if a.dry else f"; the index lists {len(index)} videos"))
    if not built:  # every story failed: something is broken, so fail the run (one bad story only warns)
        raise SystemExit("no story could be built")


def publish_index(items: list[dict], index: dict[str, dict]) -> None:
    """index.json: every published video, in items.json order (the order Ball Knowledge posts them)."""
    listing = [index[i["id"]] for i in items if i["id"] in index]
    (OUT / "index.json").write_text(json.dumps(listing, indent=1, ensure_ascii=False))
    subprocess.run(["gh", "release", "upload", RELEASE, "-R", REPO, "--clobber", str(OUT / "index.json")], check=True)


if __name__ == "__main__":
    main()
