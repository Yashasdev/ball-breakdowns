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

Stories and pain cards marked "voice": "yashas" are narrated in a clone of Yashas's own voice, and only on his Mac,
because his recordings never enter this public repo (breakdowns/yashas.py has the setup):

    python -m breakdowns.build --voice yashas --only crazy-04,pain-02      # voice, check and render into out-yashas/
    python -m breakdowns.build --upload-local crazy-04,pain-02            # after review: publish those files

GitHub's build never builds those items (todo()), so it can't overwrite them with the Chatterbox narrator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import os
import shutil
import tempfile
from pathlib import Path

from breakdowns import voice as narrator

ROOT = Path(__file__).resolve().parent.parent
ITEMS = ROOT / "items.json"
OUT = ROOT / "out"
OUT_LOCAL = ROOT / "out-yashas"  # videos voiced on Yashas's Mac (--voice yashas), kept out of git
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


def voice(model, whisper, item: dict, out: Path, exaggeration: float | None = None, strict: bool = False) -> bool:
    """strict: every word must match (short card lines, where the letter match lets "Warrior Spam" pass for
    "Warriors fan")."""
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
            if strict and ok and words(beat["say"]) != words(heard):
                ok, why = False, "a word differs"
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
            **{k: item[k] for k in CAPTION_ONLY if item.get(k)},
            **({"voice": narrator.CLONED[item["voice"]]} if narrator.local_only(item) else {})}


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


def todo(want: list[tuple[dict, str]], done: set[str], index: dict[str, dict], limit: int) -> list[tuple[dict, str]]:
    """What GitHub's build makes: new videos, and published ones whose script or card was fixed since (their hash
    changed). Never an item voiced on Yashas's Mac ("voice": "yashas"): the build can't voice it, and rebuilding it
    with the Chatterbox narrator would overwrite his version."""
    out = []
    for i, h in want:
        if f"{i['id']}.mp4" in done and index.get(i["id"], {}).get("hash") == h:
            continue
        if narrator.local_only(i):
            print(f"::notice::{i['id']} is voiced on Yashas's Mac and isn't built here; to publish a change, run "
                  f"python -m breakdowns.build --voice yashas --only {i['id']} there, then --upload-local {i['id']}")
            continue
        out.append((i, h))
    return out[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--dry", action="store_true", help="render only; publish nothing")
    ap.add_argument("--voice", choices=["yashas"], help="voice --only these items locally in Yashas's cloned voice "
                    "into out-yashas/ (never uploads)")
    ap.add_argument("--only", help="comma-separated ids, for --voice")
    ap.add_argument("--upload-local", metavar="IDS", help="publish these out-yashas/ videos to the release and index")
    a = ap.parse_args()
    if a.voice:
        return build_local([x for x in (a.only or "").split(",") if x])
    if a.upload_local:
        return upload_local([x for x in a.upload_local.split(",") if x])
    from breakdowns import cards
    want = wanted(json.loads(ITEMS.read_text()), cards.load())
    items = [i for i, _ in want]
    card_items = [i for i in items if "beats" not in i]
    done = set() if a.dry else released()
    index = {} if a.dry else published_index()
    fixed = [] if a.dry else backfill_format(index) + refresh_info(json.loads(ITEMS.read_text()), index)
    build = todo(want, done, index, a.max)
    print(f"{len(items) - len(card_items)} stories, {len(card_items)} cards, {len(index)} published, "
          f"building {len(build)}: {[i['id'] for i, _ in build]}")
    if not build:
        if fixed:
            publish_index(items, index)
        return
    from breakdowns import make
    if any("beats" in i or cards.voiced(i) for i, _ in build):  # the narrator loads only when something is voiced
        from chatterbox.tts import ChatterboxTTS
        from faster_whisper import WhisperModel
        model = ChatterboxTTS.from_pretrained(device="cpu")
        whisper = WhisperModel("small", device="cpu", compute_type="int8")
    OUT.mkdir(exist_ok=True)
    built = []
    for it, h in build:
        card = "beats" not in it
        print(f"== {it['id']} {cards.FORMATS[it['format']] if card else it['title']}", flush=True)
        if card:
            line = None
            if cards.voiced(it):  # the pain card's deadpan opening line, checked like every story line
                vdir = Path(tempfile.mkdtemp()) / it["id"]
                if voice(model, whisper, {"id": it["id"], "beats": [{"say": cards.opening(it)}]}, vdir, cards.DEADPAN, strict=True):
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


def build_local(ids: list[str]) -> None:
    """Voice (breakdowns/yashas.py, in its own Python: BK_OMNIVOICE_PYTHON), check and render items marked "voice":
    "yashas" into out-yashas/: <id>.mp4, <id>.jpg, <id>.json (with "voice") and <id>.takes.json (every take's
    transcript and similarity). One at a time; nothing is uploaded."""
    from breakdowns import cards, make
    every = {i["id"]: i for i in json.loads(ITEMS.read_text()) + cards.load()}
    if not ids or any(i not in every for i in ids):
        raise SystemExit(f"--only needs ids from items.json or cards.json; unknown: {[i for i in ids if i not in every]}")
    if [i for i in ids if not narrator.local_only(every[i])]:
        raise SystemExit(f"mark these \"voice\": \"yashas\" first, so GitHub never rebuilds them: "
                         f"{[i for i in ids if not narrator.local_only(every[i])]}")
    python = os.environ.get("BK_OMNIVOICE_PYTHON")
    if not python:
        raise SystemExit("set BK_OMNIVOICE_PYTHON (and BK_VOICE_REF, BK_VOICE_REF_TEXT: see breakdowns/yashas.py)")
    OUT_LOCAL.mkdir(exist_ok=True)
    for k in ids:
        it = every[k]
        card = "beats" not in it
        if card and not cards.voiced(it):
            raise SystemExit(f"{k}: only stories and pain cards have a voice")
        print(f"== {k} {cards.FORMATS[it['format']] if card else it['title']}", flush=True)
        work = Path(tempfile.mkdtemp(prefix=f"{k}-"))
        lines = [{"say": cards.opening(it)}] if card else [{"say": b["say"]} for b in it["beats"]]
        (work / "lines.json").write_text(json.dumps(lines))
        vdir = work / "voice"
        r = subprocess.run([python, "-m", "breakdowns.yashas", "--lines", str(work / "lines.json"), "--out", str(vdir),
                            "--seed", k.split("-")[-1], *(["--strict"] if card else [])], cwd=ROOT)
        if r.returncode:
            shutil.rmtree(work, ignore_errors=True)
            raise SystemExit(f"{k}: a line never matched its script; nothing was rendered")
        out = OUT_LOCAL / f"{k}.mp4"
        if card:
            result = cards.render(it, out, voice_wav=vdir / "line00.wav")
            info = cards.post_info(it, result, cards.card_hash(it))
        else:
            result = make.render(k, vdir, out)
            info = post_info(it, result)
        (OUT_LOCAL / f"{k}.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
        shutil.copy(vdir / "takes.json", OUT_LOCAL / f"{k}.takes.json")
        shutil.rmtree(work, ignore_errors=True)
        if result["seconds"] > 59:
            print(f"::error::{k} is {result['seconds']} s; Shorts must be under 60. Shorten its script.")
        print(f"   rendered {result['seconds']} s with {result['music']} -> {out}", flush=True)


def upload_local(ids: list[str]) -> None:
    """Publish videos made with --voice yashas (after review): upload out-yashas/<id>.mp4/.jpg/.json to the release
    and list them in index.json, in place of whatever was published under that id."""
    from breakdowns import cards
    want = wanted(json.loads(ITEMS.read_text()), cards.load())
    hashes = {i["id"]: h for i, h in want}
    index = published_index()
    for k in ids:
        files = [OUT_LOCAL / f"{k}{ext}" for ext in (".mp4", ".jpg", ".json")]
        if not all(f.exists() for f in files):
            raise SystemExit(f"{k}: render it first (--voice yashas --only {k})")
        info = json.loads(files[2].read_text())
        if info.get("hash") != hashes.get(k) or not info.get("voice"):
            raise SystemExit(f"{k}: its item changed since this video was made; render it again")
        subprocess.run(["gh", "release", "upload", RELEASE, "-R", REPO, "--clobber", *map(str, files)], check=True)
        index[k] = info
        print(f"uploaded {k} ({info['seconds']} s, voice {info['voice']})")
    OUT.mkdir(exist_ok=True)
    publish_index([i for i, _ in want], index)


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
