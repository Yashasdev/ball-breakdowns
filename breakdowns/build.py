"""Build every story in items.json and every card in cards.json that isn't published yet, and publish it.

    python -m breakdowns.build [--max 12] [--dry]

Stories ("How Crazy Was It?", format "crazy") are voiced, checked and rendered; cards (formats "guess", "hilo" and
"rank", breakdowns/cards.py) have no voice, so they skip Chatterbox and whisper and only render. Cards marked
"placeholder": true are never built or published.

Published videos live on this repo's GitHub release "videos" (public URLs, no keys): <id>.mp4, <id>.jpg (cover)
and <id>.json (what the poster needs), plus index.json listing everything published (stories in items.json order,
then cards in cards.json order). The Ball Knowledge poster in Yashasdev/act-reels reads index.json and posts the next
video it hasn't posted; each entry's "format" says which kind it is ("crazy", "guess", "hilo", "rank").

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


BUILD = 3  # raise to rebuild every published video (3: layout inside the phone's safe area)


# Fields only the poster's caption uses: they're left out of the hash, so adding or fixing one doesn't rebuild a video
# (refresh_info() updates the published index instead). "send" is the caption's first line, aimed at the losing team's
# fans ("Send this to a Falcons fan."); "who" is who that is. The video's first second doesn't show them.
CAPTION_ONLY = ("send", "who")


def story_hash(item: dict) -> str:
    """Changes whenever a story's script, captions or scenes change (or BUILD), so a fixed story is built again."""
    item = {k: v for k, v in item.items() if k not in CAPTION_ONLY}
    return hashlib.sha1(json.dumps([BUILD, item], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def published_index() -> dict[str, dict]:
    r = subprocess.run(["gh", "release", "download", RELEASE, "-R", REPO, "-p", "index.json", "-O", "-"],
                       capture_output=True, text=True)
    return {v["id"]: v for v in json.loads(r.stdout)} if r.returncode == 0 and r.stdout.strip() else {}


def voice(model, whisper, item: dict, out: Path, exaggeration: float | None = None) -> bool:
    import torch
    import torchaudio
    from breakdowns.voice import CFG, EXAGGERATION, REF
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for k, beat in enumerate(item["beats"]):
        for attempt in range(4):
            torch.manual_seed(int(item["id"].split("-")[-1]) * 1000 + k * 10 + attempt)
            x = model.generate(beat["say"], audio_prompt_path=str(REF), cfg_weight=CFG,
                               exaggeration=EXAGGERATION if exaggeration is None else exaggeration)[0]
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
    return {"id": item["id"], "format": "crazy", "title": item["title"], "kicker": item["kicker"], "sport": item["sport"],
            "hook": item["beats"][0]["cap"].replace("*", ""), "ask": item["ask"],
            "seconds": result["seconds"], "music": result["music"], "sources": item["sources"], "hash": story_hash(item),
            **{k: item[k] for k in CAPTION_ONLY if item.get(k)}}


def wanted(stories: list[dict], card_items: list[dict]) -> list[tuple[dict, str]]:
    """Everything that should be published, with its hash: every story, then every real card that passes its checks
    (placeholders never; a card with a problem is skipped with an error in the log)."""
    from breakdowns import cards
    out = [(i, story_hash(i)) for i in stories]
    for c in cards.publishable(card_items):
        if cards.problems(c):
            print(f"::error::{c['id']} isn't ready, skipping it: {'; '.join(cards.problems(c))}")
        else:
            out.append((c, cards.card_hash(c)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--dry", action="store_true", help="render only; publish nothing")
    a = ap.parse_args()
    from breakdowns import cards
    want = wanted(json.loads(ITEMS.read_text()), cards.load())
    items = [i for i, _ in want]
    card_items = [i for i in items if "beats" not in i]
    done = set() if a.dry else released()
    index = {} if a.dry else published_index()
    fixed = [] if a.dry else backfill_format(index) + refresh_info(json.loads(ITEMS.read_text()), index)
    # new videos, and published ones whose script or card was fixed since (their hash changed)
    todo = [(i, h) for i, h in want if f"{i['id']}.mp4" not in done or index.get(i["id"], {}).get("hash") != h][: a.max]
    print(f"{len(items) - len(card_items)} stories, {len(card_items)} cards, {len(index)} published, "
          f"building {len(todo)}: {[i['id'] for i, _ in todo]}")
    if not todo:
        if fixed:
            publish_index(items, index)
        return
    from breakdowns import make
    if any("beats" in i or cards.voiced(i) for i, _ in todo):  # the narrator loads only when something is voiced
        from chatterbox.tts import ChatterboxTTS
        from faster_whisper import WhisperModel
        model = ChatterboxTTS.from_pretrained(device="cpu")
        whisper = WhisperModel("small", device="cpu", compute_type="int8")
    OUT.mkdir(exist_ok=True)
    built = []
    for it, h in todo:
        card = "beats" not in it
        print(f"== {it['id']} {cards.FORMATS[it['format']] if card else it['title']}", flush=True)
        if card:
            line = None
            if cards.voiced(it):  # the pain card's deadpan opening line, checked like every story line
                vdir = Path(tempfile.mkdtemp()) / it["id"]
                if voice(model, whisper, {"id": it["id"], "beats": [{"say": cards.opening(it)}]}, vdir, cards.DEADPAN):
                    line = vdir / "line00.wav"
                else:
                    print(f"::warning::{it['id']}: the opening line never matched; building it without the voice")
            result = cards.render(it, OUT / f"{it['id']}.mp4", voice_wav=line)
            info = cards.post_info(it, result, h)
        else:
            vdir = Path(tempfile.mkdtemp()) / it["id"]
            if not voice(model, whisper, it, vdir):
                continue
            result = make.render(it["id"], vdir, OUT / f"{it['id']}.mp4")
            info = post_info(it, result)
        (OUT / f"{it['id']}.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
        if result["seconds"] > 59:
            print(f"::error::{it['id']} is {result['seconds']} s; Shorts must be under 60. Shorten its script.")
            continue
        print(f"   rendered {result['seconds']} s with {result['music']}", flush=True)
        built.append(it["id"])
        if not a.dry:
            subprocess.run(["gh", "release", "upload", RELEASE, "-R", REPO, "--clobber",
                            *(str(OUT / f"{it['id']}{ext}") for ext in (".mp4", ".jpg", ".json"))], check=True)
            index[it["id"]] = info
            publish_index(items, index)  # after every video, so a run cut short keeps what it finished
    print(f"built {built}" + ("" if a.dry else f"; the index lists {len(index)} videos"))
    if not built:  # every video failed: something is broken, so fail the run (one bad video only warns)
        raise SystemExit("no video could be built")


def backfill_format(index: dict[str, dict]) -> list[str]:
    """Stories published before cards existed have no "format": mark them "crazy" in the index and in their <id>.json
    (re-uploaded from the index entry; the video isn't rebuilt)."""
    fixed = [k for k, v in index.items() if "format" not in v and k.startswith("crazy-")]
    if not fixed:
        return []
    OUT.mkdir(exist_ok=True)
    for k in fixed:
        index[k] = {"id": k, "format": "crazy", **{f: v for f, v in index[k].items() if f != "id"}}
        (OUT / f"{k}.json").write_text(json.dumps(index[k], indent=1, ensure_ascii=False))
    subprocess.run(["gh", "release", "upload", RELEASE, "-R", REPO, "--clobber", *(str(OUT / f"{k}.json") for k in fixed)],
                   check=True)
    print(f"marked {len(fixed)} published stories as format crazy")
    return fixed


def refresh_info(stories: list[dict], index: dict[str, dict]) -> list[str]:
    """Published stories whose caption-only fields (CAPTION_ONLY, e.g. a new send line) differ from items.json: update
    their index entry and <id>.json without rebuilding the video (their hash doesn't change)."""
    fixed = []
    for it in stories:
        v = index.get(it["id"])
        if v and v.get("hash") == story_hash(it) and any(v.get(k) != it.get(k) for k in CAPTION_ONLY):
            index[it["id"]] = {**{k: x for k, x in v.items() if k not in CAPTION_ONLY},
                               **{k: it[k] for k in CAPTION_ONLY if it.get(k)}}
            fixed.append(it["id"])
    if fixed:
        OUT.mkdir(exist_ok=True)
        for k in fixed:
            (OUT / f"{k}.json").write_text(json.dumps(index[k], indent=1, ensure_ascii=False))
        subprocess.run(["gh", "release", "upload", RELEASE, "-R", REPO, "--clobber",
                        *(str(OUT / f"{k}.json") for k in fixed)], check=True)
        print(f"updated the send line of {len(fixed)} published stories (no rebuild)")
    return fixed


def publish_index(items: list[dict], index: dict[str, dict]) -> None:
    """index.json: every published video, stories in items.json order, then cards in cards.json order."""
    listing = [index[i["id"]] for i in items if i["id"] in index]
    (OUT / "index.json").write_text(json.dumps(listing, indent=1, ensure_ascii=False))
    subprocess.run(["gh", "release", "upload", RELEASE, "-R", REPO, "--clobber", str(OUT / "index.json")], check=True)


if __name__ == "__main__":
    main()
