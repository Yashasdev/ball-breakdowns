"""Every story is postable: sources listed, each beat has a script and a caption that fit, scores add up."""

import json
import re
from pathlib import Path

import pytest

from breakdowns.build import heard_ok

ROOT = Path(__file__).resolve().parent.parent
ITEMS = json.loads((ROOT / "items.json").read_text())
SCENES = {"score", "stat", "need", "catch", "kick", "shots", "debate", "call"}


def test_ids_are_unique_and_numbered_in_order():
    ids = [i["id"] for i in ITEMS]
    assert ids == [f"crazy-{k:02d}" for k in range(1, len(ids) + 1)], "append only: never reorder or renumber"


@pytest.mark.parametrize("it", ITEMS, ids=[i["id"] for i in ITEMS])
def test_story(it):
    assert 10 <= len(it["ask"]) <= 80, "the caption's comment question"
    assert it["sources"] and all(s.startswith("https://") for s in it["sources"]), "every story lists its sources"
    assert 5 <= len(it["beats"]) <= 9
    assert it["beats"][0]["scene"]["type"] == "score", "open on the stakes: the scoreboard"
    assert it["beats"][-1]["scene"]["type"] == "debate", "end on a debate that starts comments"
    words = 0
    for b in it["beats"]:
        assert b["scene"]["type"] in SCENES
        if b["scene"]["type"] == "call":  # a famous broadcast call, quoted (never the audio), with who said it
            assert b["scene"]["quote"] and b["scene"]["who"] and b["scene"]["on"] and len(b["scene"]["quote"]) <= 70
        assert len(b["say"]) <= 130 and len(b["cap"]) <= 95, b["say"]
        assert max(len(w.strip("*.,!?")) for w in b["cap"].split()) <= 13, "every caption word fits one line"
        assert b["cap"].count("*") % 2 == 0, "accent *stars* come in pairs"
        assert not re.search(r"\d", b["say"]), "say numbers in words, so the narrator reads them right"
        words += len(b["say"].split())
    assert words <= 95, "about 30 seconds of narration at most"
    scores = [s for b in it["beats"] for s in (b.get("steps") or ([b["score"]] if b.get("score") else []))]
    assert scores, "the scoreboard needs at least one score"


def test_the_voice_check_catches_slurs_and_stray_sounds():
    # real transcripts from the first build: the transcriber's spellings pass...
    assert heard_ok("Twenty-eight to three. Third quarter. In the Super Bowl.", "28-3, third quarter in the Super Bowl.")[0]
    assert heard_ok("Down thirty-two. Their star quarterback is hurt. Backup Frank Reich is in.",
                    "Down 32.  Their star quarterback is hurt.  Back up.  Frank Reich is in.")[0]
    assert heard_ok("Then Kyrie Irving. A three, right over Curry.", "than Kyrie Irving, a three right over Curry.")[0]
    assert heard_ok("LeBron hits a three. Leonard makes one free throw. Still down three.",
                    "The Bron hits a three, Leonard makes one free throw, still down three.")[0]
    # ...while stray sounds, repeats and cut-off lines don't
    assert not heard_ok("Houston needs nine points just to win.", "One Ka, Houston needs nine points just to win.")[0]
    assert not heard_ok("Miami wins in overtime. Then wins game seven. Champions.",
                        "Tax Axe of, Miami wins in overtime, then wins game 7, champions!")[0]
    assert not heard_ok("Third and ten. The plan: a catch.", "3rd and 10. The plan. A catch. 3rd and 10. The plan. A catch.")[0]
    assert not heard_ok("It's short. Chris Davis is waiting, nine yards deep in the end zone.", "Chris Davis is")[0]


def chromium():
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            p.chromium.launch().close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not chromium(), reason="needs Chromium (python -m playwright install chromium)")
def test_nothing_that_matters_is_under_the_phone_overlays():
    """Header, score bug, scenes and captions stay inside make.SAFE at every beat and caption of every story."""
    from breakdowns import make
    assert make.unsafe(ITEMS) == {i["id"]: [] for i in ITEMS}


@pytest.mark.parametrize("it", ITEMS, ids=[i["id"] for i in ITEMS])
def test_every_story_has_a_send_line_for_the_losing_fans(it):
    """The caption's first line names who to send it to; it isn't in the video, so it doesn't rebuild it."""
    from breakdowns import build
    core = re.sub(r"^(a|an|the|your)\s+", "", it["who"], flags=re.I).lower()
    assert it["send"].startswith("Send this to ") and core in it["send"].lower() and len(it["send"]) <= 70
    changed = {**it, "send": "Send this to somebody else.", "who": "somebody else"}
    assert build.story_hash(changed) == build.story_hash(it), "a send line never rebuilds a video"
    info = build.post_info(it, {"seconds": 30.0, "music": "stadium-1"})
    assert info["send"] == it["send"] and info["hash"] == build.story_hash(it)


def test_a_new_send_line_updates_the_published_index_without_a_rebuild(monkeypatch, tmp_path):
    from breakdowns import build
    monkeypatch.setattr(build, "OUT", tmp_path)
    uploads = []
    monkeypatch.setattr(build.subprocess, "run", lambda args, **k: uploads.append(args))
    it = ITEMS[0]
    index = {it["id"]: {"id": it["id"], "format": "crazy", "hash": build.story_hash(it), "hook": "x"}}
    assert build.refresh_info(ITEMS, index) == [it["id"]]
    assert index[it["id"]]["send"] == it["send"] and index[it["id"]]["hook"] == "x"
    assert json.loads((tmp_path / f"{it['id']}.json").read_text())["send"] == it["send"] and uploads
    assert build.refresh_info(ITEMS, index) == [], "nothing to do the second time"


def test_the_strict_check_for_card_lines():
    from breakdowns.build import heard_ok, words
    assert heard_ok("Send this to a Warriors fan.", "Send this to a Warrior Spam.")[0], "the letter match is lenient"
    assert words("Send this to a Warriors fan.") != words("Send this to a Warrior Spam."), "so cards also compare words"
