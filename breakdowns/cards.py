"""Ball Knowledge's quick card formats: Guess the Player, Higher or Lower, Rank These, and the share-first formats
Know Ball or Cap, The Pain Scoreboard and This Box Score Is Real.

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

Share-first formats (Nov 2026, ~/virality-research/sports.md recipes 5, 1 and 4; docs/SHARE_PLAN.md in act-reels): each
item names who it would be sent to ("who") and carries the send line ("send") shown at the end and used as the
caption's first line.

  cap      Know Ball or Cap: "Send this to the friend who says he knows ball." Four players with a one-line career,
           one of them invented (never a real, named person); a 5-second timer; FAKE is stamped; a real one is proved.
  pain     The Pain Scoreboard: "Send this to a Falcons fan." full screen (the cover; spoken when built), then one
           scoreboard: the lead, the collapse one score at a time with a hit, a draining lead bar and a jab that rubs
           it in ("Still up 19. Relax." ... "Tied. Oh no."; 2-3 of them spoken), the final and one record line, a
           spoken stinger ("28-3. Say it to a Falcons fan and watch their face."), and an end card with the send line
           and a comment question. Jabs roast teams and results only: never a player's character, injuries, family or
           life off the field; nothing hateful, no profanity; anything that sounds like a fact is true and sourced.
  boxscore This Box Score Is Real: a black card with one absurd stat line in huge type, context lines one at a time,
           a twist line, then the send line and a comment question. No voice.

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
from breakdowns import voice as narrator

ROOT = Path(__file__).resolve().parent.parent
CARDS = ROOT / "cards.json"
PAGES = Path(__file__).resolve().parent / "cards"
FORMATS = {"guess": "Guess the player", "hilo": "Higher or lower", "rank": "Rank these", "cap": "Know Ball or Cap",
           "pain": "The Pain Scoreboard", "boxscore": "This Box Score Is Real"}
SHARE = ("cap", "pain", "boxscore")  # the share-first formats: each item has "who", "send", "reason" and "hook"
LENGTHS = {"cap": (12, 16), "pain": (15, 20.5), "boxscore": (8, 12)}  # seconds; the other formats run 12-18
REASONS = ("tease", "disbelief")  # why someone sends it (SHARE_PLAN: tease a rival or a friend; "you won't believe this")
SPORTS = {"nba": "NBA", "nfl": "NFL", "cfb": "COLLEGE FOOTBALL", "cbb": "COLLEGE HOOPS", "wnba": "WNBA", "mlb": "MLB",
          "nhl": "NHL", "soccer": "SOCCER", "f1": "FORMULA 1", "golf": "GOLF", "tennis": "TENNIS", "boxing": "BOXING",
          "ufc": "UFC", "olympics": "OLYMPICS"}
# a sport's track styles (bk/music.py), as the breakdowns use them, plus the sports breakdowns don't cover
STYLES = {**make.STYLES_BY_SPORT, "cbb": ["phonk", "drill", "trap"], "wnba": ["phonk", "trap", "drill"],
          "mlb": ["stadium", "drumline"], "nhl": ["drill", "phonk", "stadium"], "soccer": ["house", "stadium", "phonk"],
          "f1": ["house", "phonk"]}
OTHER_STYLES = ["stadium", "phonk"]
COVER_TAIL = make.COVER_TAIL
VOICE_AT = 0.12  # the pain card's spoken opening line starts here
DEADPAN = 0.35  # Chatterbox exaggeration for the pain card's lines: flat and dry (the stories use 1.0, full hype)
JAB_MAX, STING_MAX = 28, 70  # characters: a jab sits under the scoreboard, the stinger in its own box
WORDS_PER_SECOND = 2.9  # for timing an unvoiced card (layout checks) as if it were spoken


def voiced(it: dict) -> bool:
    """Cards with spoken lines: only the pain card (its opening, 2-3 jabs and the stinger: pain_lines())."""
    return it.get("format") == "pain"


def spoken(text: str) -> str:
    """A line as the narrator should read it: "0-7" -> "oh and 7", "28-3" -> "28 to 3"."""
    text = re.sub(r"\b0-(\d+)\b", r"oh and \1", text)
    return re.sub(r"\b(\d+)-(\d+)\b", r"\1 to \2", text)


def pain_lines(it: dict) -> list[dict]:
    """The pain card's spoken lines, in order: the opening (every word must be heard right), the spoken jabs, the
    stinger. Each {"key", "say"[, "strict"]}; build.py voices them and render() places them."""
    return ([{"key": "open", "say": opening(it), "strict": True}]
            + [{"key": f"jab{k}", "say": spoken(it["jabs"][k])} for k in sorted(it["spoken"])]
            + [{"key": "sting", "say": it.get("stinger_say") or spoken(it["stinger"])}])


def jab_problems(it: dict) -> list[str]:
    """Jabs that state the score must match it: "Up 19" is the lead at that step, "Down 3" the deficit, "Tied" a tie,
    and "0-3" / "3-1" the score shown."""
    out = []
    for jab, step in zip(it["jabs"], it["steps"]):
        a, b = step["score"]
        low = jab.lower()
        for m in re.finditer(r"\bup (\d+)\b", low):
            if int(m.group(1)) != a - b:
                out.append(f"{jab!r}: the lead is {a - b}")
        for m in re.finditer(r"\bdown (\d+)\b", low):
            if int(m.group(1)) != b - a:
                out.append(f"{jab!r}: the deficit is {b - a}")
        if re.search(r"\btied\b", low) and a != b:
            out.append(f"{jab!r}: it isn't tied ({a}-{b})")
        for m in re.finditer(r"\b(\d+)-(\d+)\b", jab):
            if [int(m.group(1)), int(m.group(2))] != [a, b]:
                out.append(f"{jab!r}: the score is {a}-{b}")
    return out
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
def opening(it: dict) -> str:
    """The share formats' target line, shown full screen first (pain: also the voiced line)."""
    return f"Send this to {it['who']}."


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
    """The answer for the bottom of the caption; empty for pain and boxscore, which have nothing to guess."""
    if it["format"] == "cap":
        return f"{'ABCD'[it['fake']]}. {it['names'][it['fake']]['name']} never existed"
    if it["format"] in ("pain", "boxscore"):
        return ""
    if it["format"] == "guess":
        return it["answer"]
    if it["format"] == "hilo":
        return f"{it['answer'].upper()}: {display(it['right']['value'])} vs {display(it['left']['value'])}"
    return " ".join(f"{r + 1}. {it['items'][k]['name']}" for r, k in enumerate(it["order"]))


def why(it: dict) -> str:
    return it.get("why") or it.get("payoff") or it.get("proof") or it.get("record") or it.get("twist")


def post_info(it: dict, result: dict, item_hash: str) -> dict:
    info = {"id": it["id"], "format": it["format"], "title": FORMATS[it["format"]], "sport": it["sport"],
            "hook": hook(it), "ask": it["ask"], "answer": answer(it), "why": why(it), "seconds": result["seconds"],
            "music": result["music"], "sources": it["sources"], "hash": item_hash}
    if it["format"] == "rank":
        info["letters"] = letters(it)
    if it["format"] in SHARE:  # the poster's caption opens with the send line
        info.update(send=it["send"], who=it["who"])
    if narrator.local_only(it):  # narrated in Yashas's cloned voice: the poster adds the credit line
        info["voice"] = narrator.CLONED[it["voice"]]
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
    text = lambda k, lo, hi: isinstance(it.get(k), str) and lo <= len(it[k]) <= hi  # noqa: E731
    if f in SHARE:
        need(text("who", 5, 50), "who: who they'd send it to, e.g. a Falcons fan (50 characters max)")
        need(text("hook", 10, 90), "hook: the caption's and YouTube title's line (90 characters max)")
        need(it.get("reason") in REASONS, f"reason: one of {REASONS}")
        need(text("send", 15, 70) and it["send"].startswith("Send this to "), 'send: "Send this to ...", 70 characters max')
        if not out:  # the send line names the person: "a Falcons fan" -> "Falcons fan" is in it
            core = re.sub(r"^(a|an|the|your)\s+", "", it["who"], flags=re.I).lower()
            need(core in it["send"].lower(), f"send must name who it's for ({core!r})")
            need(not re.search(r"\b(bet|bets|betting|parlay|odds|wager)\b", " ".join(
                str(v) for v in it.values()).lower()), "no gambling words")
    else:
        need(text("payoff", 5, 90), "payoff: one short line (90 characters max)")
    src = it.get("sources")
    if not it.get("placeholder"):
        need(isinstance(src, list) and src and all(isinstance(s, str) and s.startswith("https://") for s in src),
             "sources: a list of https:// links")
    if out:
        return out
    if f == "cap":
        names = it.get("names")
        need(text("group", 5, 34), "group: who the four are, e.g. COWBOYS QBs AFTER AIKMAN (34 characters max)")
        need(text("proof", 10, 80), "proof: one line proving a real one is real (80 characters max)")
        need(isinstance(names, list) and len(names) == 4, "names: exactly 4 (3 real, 1 invented)")
        need(it.get("fake") in (0, 1, 2, 3), "fake: the index (0-3) of the invented player")
        if out:
            return out
        for x in names:
            need(isinstance(x, dict) and isinstance(x.get("name"), str) and 5 <= len(x["name"]) <= 22,
                 f"every name 5-22 characters: {x}")
            need(isinstance(x, dict) and isinstance(x.get("line"), str) and 5 <= len(x["line"]) <= 30,
                 f"every career line 5-30 characters: {x}")
        if not out:
            need(len({x["name"] for x in names}) == 4, "four different names")
            fake = names[it["fake"]]["name"]
            need(fake.split()[-1].lower() not in it["proof"].lower(), "the proof line is about a real one, not the fake")
    elif f == "pain":
        teams, lead, steps, final = it.get("teams"), it.get("lead"), it.get("steps"), it.get("final")
        need(text("kicker", 5, 34), "kicker: the game, e.g. SUPER BOWL LI · FEB 5, 2017 (34 characters max)")
        need(text("record", 10, 70), "record: one line under the final (70 characters max)")
        need(text("meter", 5, 24), "meter: the draining bar's label (24 characters max)")
        need(isinstance(teams, list) and len(teams) == 2 and all(
            isinstance(x, dict) and isinstance(x.get("abbr"), str) and 1 <= len(x["abbr"]) <= 6
            and re.fullmatch(r"#[0-9a-fA-F]{6}", str(x.get("color", ""))) for x in teams),
             "teams: 2, the collapsing team first, each {abbr (6 max), color #rrggbb}")
        score = lambda x: isinstance(x, dict) and isinstance(x.get("score"), list) and len(x["score"]) == 2 and all(  # noqa: E731
            isinstance(v, int) and v >= 0 for v in x["score"]) and isinstance(x.get("clock"), str) and 2 <= len(x["clock"]) <= 18
        need(score(lead) and isinstance(lead.get("line"), str) and 5 <= len(lead["line"]) <= 34,
             "lead: {score, clock (18 max), line (34 max)}")
        need(isinstance(steps, list) and 2 <= len(steps) <= 6 and all(score(x) for x in steps), "steps: 2-6 {score, clock}")
        need(score(final), "final: {score, clock}")
        if out:
            return out
        seq = [lead["score"]] + [x["score"] for x in steps] + [final["score"]]
        need(all(b[0] >= a[0] and b[1] >= a[1] and b != a for a, b in zip(seq, seq[1:])),
             "every score changes and never goes down")
        need(final["score"][0] < final["score"][1], "the final: the team in pain loses")
        need(len(it["who"]) <= 24, "who: short (24 characters max), so the spoken opening fits the first 2 seconds")
        need(lead["score"][0] >= lead["score"][1], "the lead: the team in pain is ahead (or level)")
        jabs, said = it.get("jabs"), it.get("spoken")
        need(isinstance(jabs, list) and len(jabs) == len(steps) and all(
            isinstance(j, str) and 4 <= len(j) <= JAB_MAX for j in jabs),
             f"jabs: one mocking line per step ({JAB_MAX} characters max)")
        need(isinstance(said, list) and 2 <= len(said) <= 3 and len(set(said)) == len(said) and all(
            isinstance(k, int) and 0 <= k < len(steps) for k in said), "spoken: 2-3 jab indexes to speak")
        need(text("stinger", 10, STING_MAX), f"stinger: the spoken last word ({STING_MAX} characters max)")
        need(it.get("stinger_say") is None or text("stinger_say", 10, 90), "stinger_say: how to read the stinger")
        if isinstance(jabs, list) and len(jabs) == len(steps):
            out += jab_problems(it)
    elif f == "boxscore":
        stats, ctx = it.get("stats"), it.get("context")
        need(isinstance(stats, list) and 1 <= len(stats) <= 4 and all(
            isinstance(x, dict) and isinstance(x.get("v"), str) and 1 <= len(x["v"]) <= 7
            and isinstance(x.get("k"), str) and 1 <= len(x["k"]) <= 16 for x in stats),
             "stats: 1-4 {v (7 max), k (16 max)}")
        need(isinstance(ctx, list) and 2 <= len(ctx) <= 3 and all(isinstance(c, str) and 5 <= len(c) <= 40 for c in ctx),
             "context: 2-3 lines, 40 characters max")
        need(text("twist", 10, 70), "twist: the line that makes it wilder (70 characters max)")
    elif f == "guess":
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


def meter(it: dict) -> list[float]:
    """The pain card's draining bar, one value per score shown (lead, steps, final). Drawn, not a model's number: a
    real lead drains with the lead (level = half), the final is empty; a tally (Bills 0-4) drains evenly."""
    seq = [it["lead"]["score"]] + [x["score"] for x in it["steps"]] + [it["final"]["score"]]
    d0 = seq[0][0] - seq[0][1]
    if d0 > 0:
        bars = [0.5 + 0.47 * (a - b) / d0 if a >= b else max(0.06, 0.5 + 0.5 * (a - b) / d0) for a, b in seq[:-1]]
    else:
        bars = [0.97 * (1 - k / (len(seq) - 1)) for k in range(len(seq) - 1)]
    return [round(b, 3) for b in bars] + [0.0]


def timeline(it: dict, track: str, voices: list[float] | None = None) -> dict:
    """When everything happens (seconds), snapped to the track's half-beats so hits land on the music.
    voices: how long each of the pain card's spoken lines is (pain_lines() order); the opening holds the first screen
    until it's said and a spoken jab holds its step. Without them (layout checks) they're estimated from the words."""
    half = 60 / music.bpm(track) / 2
    snap = lambda t: round(round(t / half) * half, 3)  # noqa: E731
    f = it["format"]
    data = {"format": f, "title": FORMATS[f], "sport": SPORTS.get(it["sport"], it["sport"].upper()), "hook": hook(it),
            "payoff": why(it)}
    if f in SHARE:
        data.update(opening=opening(it), send=it["send"], ask=it["ask"])
    if f == "cap":
        board = snap(1.8)
        timer = round(board + 1.0, 3)
        reveal = round(timer + 5.0, 3)
        t = {"board": board, "tiles": [round(board + 0.15 + 0.2 * k, 3) for k in range(4)], "timer": timer,
             "reveal": reveal, "real": round(reveal + 0.55, 3), "proof": round(reveal + 1.0, 3),
             "endcard": round(reveal + 3.2, 3), "end": round(reveal + 6.2, 3)}
        data.update(group=it["group"], names=it["names"], fake=it["fake"], proof=it["proof"], t=t)
        cover = reveal - 0.05
        checks = [1.2, cover, t["proof"] + 1.0]
    elif f == "pain":
        lines = pain_lines(it)
        secs = dict(zip([x["key"] for x in lines], voices or [len(x["say"].split()) / WORDS_PER_SECOND + 0.2
                                                               for x in lines]))
        board = snap(max(1.5, secs["open"] + VOICE_AT + 0.3))
        n = len(it["steps"])
        base = 4.8 / n
        steps, at = [], board + 1.8
        for k in range(n):  # a spoken jab holds its step until it's said
            steps.append(snap(at))
            at = steps[-1] + max(base, secs.get(f"jab{k}", 0) + 0.35)
        final = round(at + 0.2, 3)
        sting = round(final + 1.0, 3)
        endcard = round(max(final + 3.0, sting + secs["sting"] + 0.5), 3)
        t = {"board": board, "line": round(board + 0.5, 3), "steps": steps, "final": final,
             "record": round(final + 0.6, 3), "sting": sting, "endcard": endcard, "end": round(endcard + 2.4, 3),
             "reveal": final}
        data.update(kicker=it["kicker"], teams=it["teams"], lead=it["lead"], steps=it["steps"], final=it["final"],
                    record=it["record"], meter=it["meter"], bars=meter(it), jabs=it["jabs"], stinger=it["stinger"],
                    t=t)
        cover = 1.0  # the target line
        checks = [1.0, board + 1.5, *[x + 0.5 for x in steps], t["record"] + 0.5, sting + 1.0]
    elif f == "boxscore":
        n = len(it["context"])
        ctx = [snap(1.2 + 1.0 * k) for k in range(n)]
        twist = snap(ctx[-1] + 1.6)
        t = {"context": ctx, "twist": twist, "reveal": twist, "endcard": round(twist + 2.6, 3),
             "end": round(twist + 5.6, 3)}
        data.update(stats=it["stats"], context=it["context"], twist=it["twist"], t=t)
        cover = 0.9  # the stat line alone
        checks = [0.9, twist + 1.0]
    elif f == "guess":
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
    # when the fit and safe-area check looks: the share formats change layout several times
    data["checks"] = [round(x, 3) for x in (checks if f in SHARE else [cover])] + [round(t["end"] - 0.05, 3)]
    return data


# --- sound -------------------------------------------------------------------------------------------------------
def share_sounds(tl: dict, add, dip, rng) -> None:
    """Hits for the share formats (the shared title slam isn't used: each opens on its own sound)."""
    t, f = tl["t"], tl["format"]
    if f == "cap":
        add(sound.scratch(rng), 0.0, 0.45)
        add(sound.whoosh(rng, 0.3), t["board"] - 0.1, 0.3)
        for k, c in enumerate(t["tiles"]):
            add(sound.pop(1.0 + 0.12 * k), c, 0.45)
        for k in range(5):                          # the 5-second timer
            add(sound.tick(high=True), t["timer"] + k, 0.5)
            add(sound.tick(), t["timer"] + k + 0.5, 0.25)
        add(sound.riser(rng, 1.3), t["reveal"] - 1.3, 0.35)
        add(sound.impact(rng), t["reveal"], 1.0)
        add(sound.buzzer(), t["reveal"] + 0.02, 0.35)
        for k in range(3):
            add(sound.pop(1.3 + 0.1 * k), t["real"] + 0.12 * k, 0.35)
        add(sound.bell(), t["proof"], 0.4)
        dip(t["reveal"] - 0.1, 1.2)
    elif f == "pain":
        add(sound.scratch(rng), 0.0, 0.6)           # the record scratch under the target line
        add(sound.impact(rng), 0.02, 0.5)
        add(sound.whoosh(rng, 0.35), t["board"] - 0.15, 0.35)
        add(sound.impact(rng), t["board"], 0.7)
        add(sound.pop(1.0), t["line"], 0.35)
        for k, at in enumerate(t["steps"]):         # every point of the collapse lands like a hit
            add(sound.impact(rng), at, 0.75)
            add(sound.pop(0.9 - 0.06 * k), at, 0.45)
            dip(at - 0.05, 0.5, 0.7)
        add(sound.buzzer(), t["final"], 0.45)
        add(sound.bass_drop(), t["final"], 0.6)
        add(sound.impact(rng), t["final"], 0.9)
        dip(t["final"] - 0.1, 1.4)
        add(sound.tick(high=True), t["record"], 0.4)
        add(sound.whoosh(rng, 0.35), t["endcard"] - 0.1, 0.35)
        add(sound.pop(1.2), t["endcard"] + 0.1, 0.4)
    else:  # boxscore
        add(sound.impact(rng), 0.02, 0.9)
        add(sound.bass_drop(), 0.02, 0.6)
        for k, at in enumerate(t["context"]):
            add(sound.pop(1.0 + 0.1 * k), at, 0.4)
            for j in range(4):
                add(sound.tick(), at + 0.05 + j * 0.06, 0.1)
        add(sound.impact(rng), t["twist"], 0.9)
        add(sound.scratch(rng), t["twist"] - 0.05, 0.35)
        dip(t["twist"] - 0.1, 1.2)
        add(sound.whoosh(rng, 0.35), t["endcard"] - 0.1, 0.35)
        add(sound.pop(1.2), t["endcard"] + 0.1, 0.4)


def soundtrack(tl: dict, track: str, path: Path, voices: list[tuple[np.ndarray, float]] | None = None) -> None:
    """voices: (audio, start in seconds) for each spoken line."""
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
    if tl["format"] in SHARE:
        share_sounds(tl, add, dip, rng)
        for v, at in voices or []:                 # the pain card's spoken lines, the music ducked under each
            add(v / (np.abs(v).max() or 1) * 0.95, at, 1.0)
            dip(max(0, at - 0.05), len(v) / SR + 0.2, 0.3)
    else:
        add(sound.whoosh(rng, 0.35), 0, 0.35)      # the title slams in
        add(sound.impact(rng), 0.05, 0.6)
    if tl["format"] in SHARE:
        pass
    elif tl["format"] == "guess":
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
def render(it: dict, out: Path, track: str | None = None, voice_wavs: list[Path] | None = None) -> dict:
    """voice_wavs: the pain card's lines, spoken (build.py voices and checks them, pain_lines() order); without them
    the card is unvoiced."""
    track = track or track_for(it)
    clips = [make.load(w) for w in voice_wavs] if voice_wavs else None
    tl = timeline(it, track, None if clips is None else [len(v) / sound.SR for v in clips])
    voice = None
    if clips:
        t = tl["t"]
        starts = {"open": VOICE_AT, "sting": t["sting"], **{f"jab{k}": t["steps"][k] + 0.08 for k in range(len(t["steps"]))}}
        voice = [(v, starts[x["key"]]) for v, x in zip(clips, pain_lines(it))]
    tl["seed"] = int(it["id"].split("-")[-1]) + 100 * list(FORMATS).index(it["format"])
    work = Path(tempfile.mkdtemp(prefix=f"{it['format']}-"))
    (work / "frames").mkdir()
    make.frames(tl, work / "frames", cover_at=tl["cover"], template=PAGES / f"{it['format']}.html")
    soundtrack(tl, track, work / "sound.wav", voice)
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
            for at in tl["checks"]:  # the question, the answer, and every layout in between
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
