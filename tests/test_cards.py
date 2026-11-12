"""Every card in cards.json is postable: the schema, answers that match the values, text that fits on screen, a
length of 12-18 s, our own music, and placeholders that are never published."""

import copy
import json
import re
from pathlib import Path

import pytest

from bk import music
from breakdowns import build, cards

ROOT = Path(__file__).resolve().parent.parent
CARDS = cards.load()
REAL = cards.publishable(CARDS)
FIELDS = {"id", "format", "title", "hook", "ask", "answer", "why", "seconds", "music", "sport", "sources", "hash"}


def test_ids_are_unique_and_real_cards_are_numbered_in_order():
    ids = [c["id"] for c in CARDS]
    assert len(ids) == len(set(ids))
    for f in cards.FORMATS:
        real = [c["id"] for c in REAL if c["format"] == f]
        assert real == [f"{f}-{k:02d}" for k in range(1, len(real) + 1)], "append only: never reorder or renumber"


def test_placeholders_are_never_published():
    stories = json.loads((ROOT / "items.json").read_text())
    published = [i["id"] for i, _ in build.wanted(stories, CARDS)]
    assert not {c["id"] for c in CARDS if c.get("placeholder")} & set(published)
    assert published[: len(stories)] == [s["id"] for s in stories], "stories first, unchanged"
    assert all(c in published for c in (c["id"] for c in REAL)), "every real card is published"


@pytest.mark.parametrize("c", CARDS, ids=[c["id"] for c in CARDS])
def test_card(c):
    assert cards.problems(c) == []
    if not c.get("placeholder"):
        assert c["sources"] and all(s.startswith("https://") for s in c["sources"])
    tl = cards.timeline(c, cards.track_for(c))
    lo, hi = cards.LENGTHS.get(c["format"], (12, 18))
    assert lo <= tl["total"] + cards.COVER_TAIL <= hi, f"{lo}-{hi} seconds"
    assert tl["cover"] < tl["t"]["reveal"], "the cover shows the question, never the answer"
    assert cards.track_for(c) in music.TRACKS, "only our own composed tracks"
    info = cards.post_info(c, {"seconds": round(tl["total"] + cards.COVER_TAIL, 2), "music": cards.track_for(c)},
                           cards.card_hash(c))
    assert FIELDS <= set(info) and info["format"] == c["format"]
    assert len(info["ask"]) < 80 and info["why"]
    assert bool(info["answer"]) == (c["format"] not in ("pain", "boxscore")), "pain and boxscore have nothing to guess"
    if c["format"] in cards.SHARE:
        assert info["send"] == c["send"] and info["who"] == c["who"], "the poster opens the caption with the send line"


@pytest.mark.parametrize("c", [c for c in CARDS if c["format"] == "hilo"], ids=lambda c: c["id"])
def test_hilo_answer_matches_the_values(c):
    left, right = cards.number(c["left"]["value"]), cards.number(c["right"]["value"])
    assert c["answer"] == ("higher" if right > left else "lower")
    assert cards.answer(c).startswith(c["answer"].upper() + ": ")


@pytest.mark.parametrize("c", [c for c in CARDS if c["format"] == "rank"], ids=lambda c: c["id"])
def test_rank_order_matches_the_values(c):
    vals = [cards.number(c["items"][k]["value"]) for k in c["order"]]
    assert vals == sorted(vals, reverse=not c.get("lowest_first")) and len(set(vals)) == 4
    assert cards.answer(c).startswith(f"1. {c['items'][c['order'][0]]['name']} 2. ")


def test_the_checks_catch_mistakes():
    hilo = copy.deepcopy(next(c for c in CARDS if c["format"] == "hilo"))
    hilo["answer"] = "higher" if hilo["answer"] == "lower" else "lower"
    assert any("doesn't match" in p for p in cards.problems(hilo))
    rank = copy.deepcopy(next(c for c in CARDS if c["format"] == "rank"))
    rank["order"] = rank["order"][::-1]
    assert any("order" in p for p in cards.problems(rank))
    rank["order"] = [0, 1, 2, 3]
    assert cards.problems(rank)
    guess = copy.deepcopy(next(c for c in CARDS if c["format"] == "guess"))
    guess["clues"][0] = f"His last name is {guess['answer'].split()[-1]}"
    assert any("gives away" in p for p in cards.problems(guess))
    guess["clues"] = guess["clues"][:3]
    assert cards.problems(guess)
    long = copy.deepcopy(next(c for c in CARDS if c["format"] == "guess"))
    long["ask"] = "x" * 80
    assert cards.problems(long)


def test_values():
    assert cards.number("38,652") == 38652 and cards.number("41.2%") == 41.2 and cards.number(27.1) == 27.1
    assert cards.display(38652) == "38,652" and cards.display(649) == "649" and cards.display(27.1) == "27.1"
    assert cards.display("41.2%") == "41.2%"


def test_every_format_has_a_page():
    for f in cards.FORMATS:
        page = (cards.PAGES / f"{f}.html").read_text()
        assert "common.js" in page and "async function setup" in page and "function render" in page
        assert not re.search(r"https?://", page), "pages load nothing from the internet"


def chromium():
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            p.chromium.launch().close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not chromium(), reason="needs Chromium (python -m playwright install chromium)")
def test_all_text_fits_on_screen():
    """Every text fits its box, and nothing that matters is under the phone's overlays (make.SAFE)."""
    assert cards.overflow(CARDS) == {c["id"]: [] for c in CARDS}


@pytest.mark.skipif(not chromium(), reason="needs Chromium (python -m playwright install chromium)")
def test_the_safe_area_check_catches_a_covered_element():
    from playwright.sync_api import sync_playwright
    from breakdowns import make
    c = next(c for c in REAL if c["format"] == "hilo")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": make.W, "height": make.H})
        page.goto((cards.PAGES / "hilo.html").as_uri())
        page.evaluate("tl => setup(tl)", cards.timeline(c, cards.track_for(c)))
        page.evaluate("render(12)")
        assert page.evaluate("L => unsafe(L)", make.SAFE) == []
        page.evaluate("document.getElementById('btns').style.left = '200px'")   # under the like/comment column
        page.evaluate("document.getElementById('title').style.top = '150px'")   # under the clock
        bad = page.evaluate("L => unsafe(L)", make.SAFE)
        browser.close()
    assert any(b.startswith("btns: ") and "lower half" in b for b in bad) and any(b.startswith("title: top") for b in bad)


SHARE = [c for c in CARDS if c["format"] in cards.SHARE]


@pytest.mark.parametrize("c", SHARE, ids=[c["id"] for c in SHARE])
def test_share_card_names_who_to_send_it_to(c):
    """The send line names a person (SHARE_PLAN), opens the caption, and the facts are sourced."""
    core = re.sub(r"^(a|an|the|your)\s+", "", c["who"], flags=re.I).lower()
    assert c["send"].startswith("Send this to ") and core in c["send"].lower()
    assert cards.opening(c) == f"Send this to {c['who']}."
    assert c["reason"] in cards.REASONS and c["hook"]
    if not c.get("placeholder"):
        assert len(c["sources"]) >= 2, "every share card lists at least two sources"
    if c["format"] == "cap":
        assert sum(1 for x in c["names"] if x["name"] == c["names"][c["fake"]]["name"]) == 1
    if c["format"] == "pain":
        bars = cards.meter(c)
        # it can tick up when the leading team scores again, but it starts highest and ends empty
        assert len(bars) == len(c["steps"]) + 2 and bars[-1] == 0 and bars[0] == max(bars)


def test_fake_answers_are_spread_across_the_letters():
    fakes = [c["fake"] for c in REAL if c["format"] == "cap"]
    if len(fakes) >= 4:
        assert len(set(fakes)) == 4, "the invented player sits at every letter, A to D"


def test_the_share_checks_catch_mistakes():
    for f in cards.SHARE:
        c = next((c for c in CARDS if c["format"] == f), None)
        if c is None:
            continue
        bad = copy.deepcopy(c)
        bad["send"] = "Send this to someone."
        assert any("must name" in p for p in cards.problems(bad))
        bad = copy.deepcopy(c)
        bad["hook"] = bad["hook"] + " Place your bets."
        assert any("gambling" in p for p in cards.problems(bad))
    pain = next((c for c in CARDS if c["format"] == "pain"), None)
    if pain:
        bad = copy.deepcopy(pain)
        bad["final"]["score"] = bad["final"]["score"][::-1]
        assert cards.problems(bad)


PAIN = [c for c in REAL if c["format"] == "pain"]
NO_GO = re.compile(r"\b(damn|hell|crap|suck|sucks|stupid|idiot|choke artist|injur\w*|hurt|wife|girlfriend|mom|dad|"
                   r"bet|bets|betting|parlay|odds)\b", re.I)


@pytest.mark.parametrize("c", PAIN, ids=[c["id"] for c in PAIN])
def test_pain_cards_rub_it_in_within_the_rules(c):
    """Yashas (Nov 11, 2026): a jab on every step (2-3 spoken), a spoken stinger, a caption line that keeps the joke
    going; jabs that state the score match it; teams and results only, no profanity, injuries or family."""
    assert len(c["jabs"]) == len(c["steps"]) and 2 <= len(c["spoken"]) <= 3
    assert cards.jab_problems(c) == []
    lines = cards.pain_lines(c)
    assert [x["key"] for x in lines] == ["open", *[f"jab{k}" for k in sorted(c["spoken"])], "sting"]
    assert lines[0]["say"] == cards.opening(c) and lines[0].get("strict")
    assert not any(re.search(r"\d+-\d+", x["say"]) for x in lines), "scores are read out, not '28-3'"
    for text in [*c["jabs"], c["stinger"], c["send"], c.get("stinger_say") or ""]:
        assert not NO_GO.search(text), text
    assert len(c["send"]) > len(cards.opening(c)), "the caption's first line keeps the joke going"


def test_a_jab_with_the_wrong_score_is_caught():
    c = copy.deepcopy(PAIN[0])
    c["jabs"][0] = "Still up 40. Relax."
    assert any("the lead is" in p for p in cards.problems(c))
    c["jabs"][0] = "Tied already."
    assert any("isn't tied" in p for p in cards.problems(c))
    assert cards.spoken("0-7 since 1996") == "oh and 7 since 1996" and cards.spoken("Up 28-3.") == "Up 28 to 3."
