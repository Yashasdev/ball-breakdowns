"""Ball Knowledge's quick card formats: Guess the Player, Higher or Lower, Rank These.

    python -m breakdowns.cards guess-01                       # render one item from cards.json into out/
    python -m breakdowns.cards guess-01 --items FILE --out X  # from another items file (a draft), to X.mp4
    python -m breakdowns.cards --check [FILE]                 # schema and text-fit check, renders nothing

No narration: on-screen text, motion, an original track (bk/music.py) and synthesized sounds (bk/sound.py), 12-18 s,
built to loop. Each format has its own page in breakdowns/cards/ (drawn frame by frame by headless Chrome, like the
breakdowns) and its own layout and motion, so the channel never posts look-alikes:

  guess  a scouting file: clues type in one at a time, hardest first, with a "clues used" meter; the name slams in.
  hilo   a split-screen duel: the left value is shown, the right is a "?"; a 3-2-1 ring; the number counts up and a
         HIGHER / LOWER stamp lands.
  rank   a 2x2 grid of lettered tiles; a 3-2-1; the tiles fold into a leaderboard and slide into the right order.

The last 0.5 s is the cover: the question, never the answer. Items are in cards.json; "placeholder": true items are
for building and testing the pages only and are never published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from bk import music, sound
from breakdowns import make

ROOT = Path(__file__).resolve().parent.parent
CARDS = ROOT / "cards.json"
PAGES = Path(__file__).resolve().parent / "cards"
FORMATS = {"guess": "Guess the player", "hilo": "Higher or lower", "rank": "Rank these"}
SPORTS = {"nba": "NBA", "nfl": "NFL", "cfb": "COLLEGE FOOTBALL", "cbb": "COLLEGE HOOPS", "wnba": "WNBA", "mlb": "MLB",
          "nhl": "NHL", "soccer": "SOCCER", "f1": "FORMULA 1", "golf": "GOLF", "tennis": "TENNIS", "boxing": "BOXING",
          "ufc": "UFC", "olympics": "OLYMPICS"}
# a sport's track styles (bk/music.py), as the breakdowns use them, plus the sports breakdowns don't cover
STYLES = {**make.STYLES_BY_SPORT, "cbb": ["phonk", "drill", "trap"], "wnba": ["phonk", "trap", "drill"],
          "mlb": ["stadium", "drumline"], "nhl": ["drill", "phonk", "stadium"], "soccer": ["house", "stadium", "phonk"],
          "f1": ["house", "phonk"]}
OTHER_STYLES = ["stadium", "phonk"]
COVER_TAIL = make.COVER_TAIL
CARD_BUILD = 5  # raise to rebuild every published card (5: layout inside the phone's safe area)
# Yashas (Nov 8, 2026): the cards' background music should be a quick, stimulating rap beat, like the old card beat
# then (same day) asked for more variety: Atlanta trap, melodic and drill alongside the quick beat
CARD_STYLES = ["rapquick", "atltrap", "melodic", "drill"]


def load(path: Path = CARDS) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def publishable(items: list[dict]) -> list[dict]:
    """Real items only: placeholders exist to build and test the pages and are never published."""
    return [i for i in items if not i.get("placeholder")]


# --- values ------------------------------------------------------------------------------------------------------
def number(v) -> float:
    """A stat's value as a number: 38652, 27.1, "38,652", "41.2%", "$1.2"."""
    if isinstance(v, bool):
        raise ValueError(f"not a stat value: {v!r}")
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d[\d,]*(\.\d+)?|-?\.\d+", str(v))
    if not m:
        raise ValueError(f"no number in {v!r}")
    return float(m.group(0).replace(",", ""))


def display(v) -> str:
    """How a value is shown: numbers get thousands commas (38,652); strings are shown as written."""
    if isinstance(v, str):
        return v
    if isinstance(v, int) or float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:,}"


def card_hash(it: dict) -> str:
    """Changes whenever a card's item changes (or CARD_BUILD), so a fixed card is built again."""
    return hashlib.sha1(json.dumps([CARD_BUILD, it], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


# --- what the poster needs ----------------------------------------------------------------------------------------
def hook(it: dict) -> str:
    """The first on-screen line (the caption's first line)."""
    if it.get("hook"):
        return it["hook"]
    if it["format"] == "guess":
        return f"Guess the player in {len(it['clues'])} clues"
    if it["format"] == "hilo":
        return f"Higher or lower? {it['stat']}"
    return it["question"]


def letters(it: dict) -> str:
    """Rank These: the answer as tile letters, #1 first ("BADC"), the way people comment it."""
    return "".join("ABCD"[k] for k in it["order"])


def answer(it: dict) -> str:
    if it["format"] == "guess":
        return it["answer"]
    if it["format"] == "hilo":
        return f"{it['answer'].upper()}: {display(it['right']['value'])} vs {display(it['left']['value'])}"
    return " ".join(f"{r + 1}. {it['items'][k]['name']}" for r, k in enumerate(it["order"]))


def why(it: dict) -> str:
    return it.get("why") or it["payoff"]


def post_info(it: dict, result: dict, item_hash: str) -> dict:
    info = {"id": it["id"], "format": it["format"], "title": FORMATS[it["format"]], "sport": it["sport"],
            "hook": hook(it), "ask": it["ask"], "answer": answer(it), "why": why(it), "seconds": result["seconds"],
            "music": result["music"], "sources": it["sources"], "hash": item_hash}
    if it["format"] == "rank":
        info["letters"] = letters(it)
    return info


# --- checks ------------------------------------------------------------------------------------------------------
def problems(it: dict) -> list[str]:
    """What's wrong with an item (empty when it's fine). Text lengths here; the pages check the fit on screen."""
    out = []
    need = lambda cond, msg: cond or out.append(msg)  # noqa: E731
    f = it.get("format")
    if f not in FORMATS:
        return [f"format must be one of {list(FORMATS)}"]
    need(re.fullmatch(rf"{f}-\d{{2,3}}", str(it.get("id", ""))), f"id must look like {f}-01")
    need(isinstance(it.get("sport"), str) and re.fullmatch(r"[a-z0-9]+", it["sport"]), "sport: a short lowercase code")
    need(isinstance(it.get("ask"), str) and 10 <= len(it["ask"]) < 80, "ask: the caption's question, under 80 characters")
    need(isinstance(it.get("payoff"), str) and 5 <= len(it["payoff"]) <= 90, "payoff: one short line (90 characters max)")
    src = it.get("sources")
    if not it.get("placeholder"):
        need(isinstance(src, list) and src and all(isinstance(s, str) and s.startswith("https://") for s in src),
             "sources: a list of https:// links")
    if out:
        return out
    if f == "guess":
        clues = it.get("clues")
        need(isinstance(it.get("answer"), str) and 3 <= len(it["answer"]) <= 30, "answer: the player's name")
        need(isinstance(clues, list) and 4 <= len(clues) <= 5, "clues: 4 or 5")
        if out:
            return out
        for c in clues:
            need(isinstance(c, str) and 5 <= len(c) < 60, f"clue under 60 characters: {c!r}")
        last = it["answer"].split()[-1].lower()
        for c in clues:
            need(last not in str(c).lower(), f"a clue gives away the name: {c!r}")
    elif f == "hilo":
        need(isinstance(it.get("stat"), str) and 3 <= len(it["stat"]) < 40, "stat: a label under 40 characters")
        need(isinstance(it.get("as_of"), str) and 2 <= len(it["as_of"]) <= 32, "as_of: e.g. career, through 2024-25")
        need(it.get("answer") in ("higher", "lower"), 'answer: "higher" or "lower" (right vs left)')
        for side in ("left", "right"):
            x = it.get(side)
            need(isinstance(x, dict) and isinstance(x.get("name"), str) and 3 <= len(x["name"]) <= 32,
                 f"{side}: a name (32 characters max)")
            if not out:
                try:
                    number(x["value"])
                except (KeyError, ValueError):
                    out.append(f"{side}: value must be a number")
                need(len(display(x.get("value", ""))) <= 9, f"{side}: value too long to show")
        if not out:
            lv, rv = number(it["left"]["value"]), number(it["right"]["value"])
            need(lv != rv, "the values are equal: neither higher nor lower")
            need(it["answer"] == ("higher" if rv > lv else "lower"), "answer doesn't match the values")
    elif f == "rank":
        rows, order = it.get("items"), it.get("order")
        need(isinstance(it.get("question"), str) and 10 <= len(it["question"]) < 60, "question: under 60 characters")
        need(isinstance(it.get("stat_label"), str) and 3 <= len(it["stat_label"]) <= 40, "stat_label: 40 characters max")
        need(isinstance(rows, list) and len(rows) == 4, "items: exactly 4")
        need(isinstance(order, list) and sorted(order) == [0, 1, 2, 3], "order: the 4 item indexes, #1 first")
        if out:
            return out
        for x in rows:
            need(isinstance(x, dict) and isinstance(x.get("name"), str) and 3 <= len(x["name"]) <= 32,
                 "every item has a name (32 characters max)")
            try:
                number(x["value"])
            except (KeyError, ValueError, TypeError):
                out.append(f"value must be a number: {x}")
            need(len(display(x.get("value", ""))) <= 9, f"value too long to show: {x}")
        if out:
            return out
        vals = [number(rows[k]["value"]) for k in order]
        best_low = bool(it.get("lowest_first"))  # e.g. rank by ERA: #1 is the lowest
        ok = all(a < b for a, b in zip(vals, vals[1:])) if best_low else all(a > b for a, b in zip(vals, vals[1:]))
        need(ok, "order doesn't match the values (and ties aren't allowed)")
        need(order != [0, 1, 2, 3], "the items are shown already in order: shuffle them")
    return out


# --- the timeline -------------------------------------------------------------------------------------------------
def track_for(it: dict) -> str:
    """A different original track per item: its sport's styles, offset by format so the same day's cards differ."""
    if it.get("music"):
        return it["music"]
    styles = CARD_STYLES or STYLES.get(it["sport"], OTHER_STYLES)
    options = [f"{s}-{k}" for k in range(1, 6) for s in styles]
    n = int(it["id"].split("-")[-1])
    return options[(n + 2 * list(FORMATS).index(it["format"])) % len(options)]


def timeline(it: dict, track: str) -> dict:
    """When everything happens (seconds), snapped to the track's half-beats so hits land on the music."""
    half = 60 / music.bpm(track) / 2
    snap = lambda t: round(round(t / half) * half, 3)  # noqa: E731
    f = it["format"]
    data = {"format": f, "title": FORMATS[f], "sport": SPORTS.get(it["sport"], it["sport"].upper()), "hook": hook(it),
            "payoff": it["payoff"]}
    if f == "guess":
        n = len(it["clues"])
        step = 2.5 if n == 5 else 3.0
        clues = [snap(0.9 + k * step) for k in range(n)]
        reveal = snap(0.9 + n * step)
        t = {"clues": clues, "reveal": reveal, "payoff": round(reveal + 0.45, 3), "end": round(reveal + 3.0, 3)}
        data.update(clues=it["clues"], answer=it["answer"], t=t)
        cover = reveal - 0.8
    elif f == "hilo":
        cd = snap(4.2)
        reveal = round(cd + 3.0, 3)
        stamp = round(reveal + 1.6, 3)
        t = {"right": 0.45, "prompt": 1.2, "countdown": cd, "reveal": reveal, "stamp": stamp,
             "payoff": round(stamp + 0.35, 3), "end": round(stamp + 3.4, 3)}
        data.update(stat=it["stat"], as_of=it["as_of"], answer=it["answer"], t=t,
                    left={"name": it["left"]["name"], "value": display(it["left"]["value"]), "n": number(it["left"]["value"])},
                    right={"name": it["right"]["name"], "value": display(it["right"]["value"]), "n": number(it["right"]["value"])})
        cover = cd - 0.05
    else:
        cd = snap(5.4)
        reveal = round(cd + 3.0, 3)
        fold, slide = reveal, round(reveal + 0.6, 3)
        ranks = [round(reveal + 1.7 + 0.6 * j, 3) for j in range(4)]  # #4 first, #1 last
        t = {"tiles": [round(0.7 + 0.22 * k, 3) for k in range(4)], "prompt": 2.0, "countdown": cd, "reveal": reveal,
             "fold": fold, "slide": slide, "ranks": ranks, "payoff": round(ranks[-1] + 0.5, 3),
             "end": round(ranks[-1] + 3.2, 3)}
        example = "BDAC" if letters(it) != "BDAC" else "CADB"
        data.update(question=it["question"], stat_label=it["stat_label"], order=it["order"], example=example, t=t,
                    items=[{"name": x["name"], "value": display(x["value"]), "n": number(x["value"])} for x in it["items"]])
        cover = cd - 0.05
    data["total"] = t["end"]
    data["cover"] = round(cover, 3)
    return data


# --- sound -------------------------------------------------------------------------------------------------------
def soundtrack(tl: dict, track: str, path: Path) -> None:
    total = tl["total"] + COVER_TAIL
    n = int(total * sound.SR) + sound.SR
    buf = np.zeros(n, np.float32)
    duck = np.ones(n, np.float32)
    rng = np.random.default_rng(int(tl.get("seed", 7)))
    SR = sound.SR

    def add(x, at, gain):
        x = x if x.ndim == 1 else x.mean(axis=1)
        i = int(max(0, at) * SR)
        k = min(len(x), n - i)
        if k > 0:
            buf[i:i + k] += x[:k] * gain

    def dip(at, dur, level=0.55):
        duck[int(at * SR): int((at + dur) * SR)] = level

    t = tl["t"]
    add(sound.whoosh(rng, 0.35), 0, 0.35)          # the title slams in
    add(sound.impact(rng), 0.05, 0.6)
    if tl["format"] == "guess":
        for k, c in enumerate(t["clues"]):
            add(sound.whoosh(rng, 0.25), c - 0.1, 0.18)
            add(sound.pop(1.0 + 0.12 * k), c, 0.5)
            for j in range(6):                      # the clue types in: soft key clicks
                add(sound.tick(), c + 0.06 + j * 0.07, 0.12)
        add(sound.riser(rng, 1.3), t["reveal"] - 1.3, 0.4)
        add(sound.impact(rng), t["reveal"], 0.9)
        add(sound.bass_drop(), t["reveal"], 0.6)
        add(sound.bell(), t["reveal"] + 0.05, 0.5)
        add(sound.crowd(rng, 2.4), t["reveal"] + 0.05, 0.22)
        dip(t["reveal"] - 0.1, 1.4)
    else:
        if tl["format"] == "hilo":
            add(sound.whoosh(rng, 0.3), t["right"] - 0.1, 0.3)
        else:
            for k, c in enumerate(t["tiles"]):
                add(sound.pop(1.0 + 0.1 * k), c, 0.4)
        for k in range(3):                           # 3-2-1
            add(sound.tick(high=True), t["countdown"] + k, 0.55)
            add(sound.tick(), t["countdown"] + k + 0.5, 0.3)
        add(sound.riser(rng, 1.2), t["reveal"] - 1.2, 0.35)
        add(sound.whoosh(rng, 0.3), t["reveal"] - 0.05, 0.4)
        if tl["format"] == "hilo":
            k, at = 0, t["reveal"]
            while at < t["stamp"] - 0.05:            # the number spins up
                add(sound.tick(high=k % 2 == 0), at, 0.25)
                at += 0.06 + 0.1 * ((at - t["reveal"]) / (t["stamp"] - t["reveal"])) ** 2
                k += 1
            add(sound.impact(rng), t["stamp"], 1.0)
            add(sound.bell(), t["stamp"] + 0.03, 0.55)
            add(sound.crowd(rng, 2.0), t["stamp"] + 0.05, 0.18)
            dip(t["stamp"] - 0.1, 1.2)
        else:
            add(sound.whoosh(rng, 0.45), t["slide"] - 0.05, 0.35)
            for j, r in enumerate(t["ranks"]):
                add(sound.pop(0.9 + 0.15 * j), r, 0.5)
                add(sound.impact(rng), r, 0.35 if j < 3 else 0.95)
            add(sound.bass_drop(), t["ranks"][-1], 0.55)
            add(sound.bell(), t["ranks"][-1] + 0.03, 0.5)
            add(sound.crowd(rng, 2.0), t["ranks"][-1] + 0.05, 0.2)
            dip(t["ranks"][-1] - 0.1, 1.2)
    bed = music.render(track, total)[:n].mean(axis=1)
    sm = np.convolve(duck[: len(bed)], np.ones(2205) / 2205, mode="same")
    buf[: len(bed)] += (bed * sm * 0.7).astype(np.float32)  # no voice on cards, so the beat sits up front
    buf = buf[: int(total * SR)]
    fade = int(0.3 * SR)
    buf[-fade:] *= np.linspace(1, 0, fade)        # a soft end, so the loop back to the start is clean
    sf.write(path, buf / (np.abs(buf).max() or 1) * 0.9, SR)


# --- render ------------------------------------------------------------------------------------------------------
def render(it: dict, out: Path, track: str | None = None) -> dict:
    track = track or track_for(it)
    tl = timeline(it, track)
    tl["seed"] = int(it["id"].split("-")[-1]) + 100 * list(FORMATS).index(it["format"])
    work = Path(tempfile.mkdtemp(prefix=f"{it['format']}-"))
    (work / "frames").mkdir()
    make.frames(tl, work / "frames", cover_at=tl["cover"], template=PAGES / f"{it['format']}.html")
    soundtrack(tl, track, work / "sound.wav")
    make.encode(work, out)
    return {"video": str(out), "seconds": round(tl["total"] + COVER_TAIL, 2), "music": track}


def overflow(items: list[dict]) -> dict[str, list[str]]:
    """Text that doesn't fit its box, and key elements outside the phone's safe area (make.SAFE), per item id, on the
    cover (the question) and at the end (the answer); empty lists when everything fits. Needs Chromium."""
    from playwright.sync_api import sync_playwright
    found = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": make.W, "height": make.H})
        for it in items:
            page.goto((PAGES / f"{it['format']}.html").as_uri())
            page.evaluate("document.fonts.ready")
            tl = timeline(it, track_for(it))
            page.evaluate("tl => setup(tl)", tl)
            bad = []
            for at in (tl["cover"], tl["total"] - 0.05):  # the question, and the answer
                page.evaluate(f"render({at})")
                bad += page.evaluate("overflowing()")
                bad += [f"outside the safe area: {e}" for e in page.evaluate("L => unsafe(L)", make.SAFE)]
            found[it["id"]] = sorted(set(bad))
        browser.close()
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("id", nargs="?")
    ap.add_argument("--items", type=Path, default=CARDS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--music")
    ap.add_argument("--check", action="store_true", help="check the items file (schema and fit); render nothing")
    a = ap.parse_args()
    items = load(a.items)
    if a.check:
        bad = 0
        fit = overflow([i for i in items if not problems(i)])
        for i in items:
            msgs = problems(i) + [e if e.startswith("outside") else f"doesn't fit on screen: {e}" for e in fit.get(i["id"], [])]
            bad += bool(msgs)
            print(f"{i.get('id')}: {'ok' if not msgs else '; '.join(msgs)}")
        raise SystemExit(1 if bad else 0)
    it = next(i for i in items if i["id"] == a.id)
    if problems(it):
        raise SystemExit(f"{a.id}: {problems(it)}")
    print(json.dumps(render(it, a.out or ROOT / "out" / f"{a.id}.mp4", a.music)))


if __name__ == "__main__":
    main()
