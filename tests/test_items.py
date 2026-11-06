"""Every story is postable: sources listed, each beat has a script and a caption that fit, scores add up."""

import json
import re
from pathlib import Path

import pytest

from breakdowns.build import heard_ok

ROOT = Path(__file__).resolve().parent.parent
ITEMS = json.loads((ROOT / "items.json").read_text())
SCENES = {"score", "stat", "need", "catch", "kick", "shots", "debate"}


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
        assert len(b["say"]) <= 130 and len(b["cap"]) <= 95, b["say"]
        assert max(len(w.strip("*.,!?")) for w in b["cap"].split()) <= 13, "every caption word fits one line"
        assert b["cap"].count("*") % 2 == 0, "accent *stars* come in pairs"
        assert not re.search(r"\d", b["say"]), "say numbers in words, so the narrator reads them right"
        words += len(b["say"].split())
    assert words <= 95, "about 30 seconds of narration at most"
    scores = [s for b in it["beats"] for s in (b.get("steps") or ([b["score"]] if b.get("score") else []))]
    assert scores, "the scoreboard needs at least one score"


def test_the_voice_check_catches_slurs_and_stray_sounds():
    assert heard_ok("Atlanta never touched the ball.", "Atlanta never touched the ball.")[0]
    assert heard_ok("Fifty-seven seconds left. Tied.", "57 seconds left. Tied.")[0]
    assert not heard_ok("Houston needs nine points just to win.", "One Ka, Houston needs nine points just to win.")[0]
    assert not heard_ok("It's short. Chris Davis is waiting.", "Chris Davis is")[0]
