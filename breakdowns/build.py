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
sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand and a""".split())


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("-", " ").replace("'", "")).split()


def heard_ok(say: str, heard: str) -> tuple[bool, str]:
    """Does the transcript match the script? Numbers are skipped (the transcriber writes 28 for twenty-eight)."""
    want = [w for w in words(say) if w not in NUMBER_WORDS and not w.isdigit()]
    got = [w for w in words(heard) if w not in NUMBER_WORDS and not w.isdigit()]
    if not want:
        return True, ""
    missing = [w for w in want if w not in got]
    extra = [w for w in got if w not in want]
    first_ok = bool(got) and got[0][:3] == want[0][:3]      # no stray sound before the line ("one ka, Houston...")
    last_ok = bool(got) and got[-1][:3] == want[-1][:3]      # nothing cut off at the end
    ok = len(missing) <= 1 and len(extra) <= 1 and first_ok and last_ok
    return ok, f"missing {missing}, extra {extra}" if not ok else ""


def released() -> set[str]:
    r = subprocess.run(["gh", "release", "view", RELEASE, "-R", REPO, "--json", "assets", "--jq", ".assets[].name"],
                       capture_output=True, text=True)
    if r.returncode:
        subprocess.run(["gh", "release", "create", RELEASE, "-R", REPO, "--title", "Published breakdown videos",
                        "--notes", "Rendered by the build workflow. index.json lists them in posting order."], check=True)
        return set()
    return set(r.stdout.split())


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
            "seconds": result["seconds"], "music": result["music"], "sources": item["sources"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--dry", action="store_true", help="render only; publish nothing")
    a = ap.parse_args()
    items = json.loads(ITEMS.read_text())
    done = set() if a.dry else released()
    todo = [i for i in items if f"{i['id']}.mp4" not in done][: a.max]
    print(f"{len(items)} stories, {sum(n.endswith('.mp4') for n in done)} published, building {len(todo)}: {[i['id'] for i in todo]}")
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
        (OUT / f"{it['id']}.json").write_text(json.dumps(post_info(it, result), indent=1))
        print(f"   rendered {result['seconds']} s with {result['music']}", flush=True)
        built.append(it["id"])
        if not a.dry:
            subprocess.run(["gh", "release", "upload", RELEASE, "-R", REPO, "--clobber",
                            *(str(OUT / f"{it['id']}{ext}") for ext in (".mp4", ".jpg", ".json"))], check=True)
    if a.dry or not built:
        return
    # index.json: everything published, in items.json order (the order Ball Knowledge posts them)
    have = released()
    index = []
    for it in items:
        if f"{it['id']}.mp4" in have:
            local = OUT / f"{it['id']}.json"
            if not local.exists():
                subprocess.run(["gh", "release", "download", RELEASE, "-R", REPO, "-p", f"{it['id']}.json", "-D", str(OUT),
                                "--clobber"], check=True)
            index.append(json.loads(local.read_text()))
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    subprocess.run(["gh", "release", "upload", RELEASE, "-R", REPO, "--clobber", str(OUT / "index.json")], check=True)
    print(f"published {built}; index lists {len(index)} videos")


if __name__ == "__main__":
    main()
