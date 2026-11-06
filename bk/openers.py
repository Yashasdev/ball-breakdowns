"""The parts of a reel's first second (the hook) that can vary, and the countdown hooks before test day.

Each reel gets one of each, picked by the tuner:
  words   the headline: one of the subject's "hooks" in config.json, or a countdown line near an exam
  sound   the opening sound (sound.py draws them)
  visual  the opening motion (render.py draws them)
The first option of each is the original. Standard library only, so the Tracker can use it.
"""

from __future__ import annotations

import math
from datetime import date
from fractions import Fraction

SOUNDS = ("whoosh", "buzzer", "scratch", "bass_drop")
FOLLOW = "Follow so you don't miss Day {next}"  # the original closing line
VISUALS = ("slam", "red_flash", "instant", "punch")


def countdown(cfg: dict, day: date) -> dict | None:
    """The next exam within the brand's countdown window (or today), as hook fields: {"exam": "SAT", "days": 9}."""
    cal = cfg.get("calendar")
    if not cal:
        return None
    for e in sorted(cal["exams"], key=lambda e: e["date"]):
        days = (date.fromisoformat(e["date"]) - day).days
        if 0 <= days <= cal["window_days"]:
            return {"exam": e["name"], "days": days}
    return None


def countdown_slot(cfg: dict, day: date, slot: str, days_left: int) -> bool:
    """Whether this slot gets a countdown line, spreading them evenly over the countdown window.

    calendar.countdown_share is the share of reels that count down (1/3 means one reel in three): each day
    gets its fair share (rounded so the total stays even), and the slot that gets it rotates from day to
    day, so countdowns don't always land at the same time. 1 means every reel counts down.
    """
    cal = cfg["calendar"]
    share = Fraction(str(cal.get("countdown_share", 1))).limit_denominator(24)
    # narrated explainers have no headline, so countdowns rotate over the other slots
    live = [s for s in cfg["slots"] if s.get("from", "") <= day.isoformat() <= s.get("until", "9999")]
    slots = sorted(s["time"] for s in live if cfg["subjects"][s["subject"]]["kind"] != "explainer")
    if slot not in slots:
        return False
    n, d = len(slots), cal["window_days"] - days_left  # d: days since the window opened
    before = math.floor(d * share * n)  # countdowns given on earlier days of the window
    today = math.floor((d + 1) * share * n) - before
    # rotate by how many have gone out so far, so with fewer than one a day every slot still takes turns
    return (slots.index(slot) - before) % n < today


def word_options(cfg: dict, subject: str, day: date, slot: str | None = None) -> tuple[list[str], dict]:
    """The headlines a reel posting at `slot` on `day` can open with, and the fields to fill them with.

    In the countdown window before an exam, a share of reels (calendar.countdown_share, spread evenly)
    open with a countdown line ("The SAT is in 9 days") and the rest with normal hooks. The day before,
    every reel opens with a "tomorrow" line, and on test day with a "today" line; every post goes out after
    the morning exam, so test-day lines ask how it went.
    """
    fields = countdown(cfg, day)
    if fields:
        cal = cfg["calendar"]
        if fields["days"] in (0, 1):
            return (cal["today"] if fields["days"] == 0 else cal["tomorrow"]), fields
        if slot is None or countdown_slot(cfg, day, slot, fields["days"]):
            return cal["hooks"], fields
    return cfg["subjects"][subject]["look"]["hooks"], {}


def is_countdown(cfg: dict, hook: str | None) -> bool:
    """Whether a hook is one of the brand's countdown, tomorrow or test-day lines."""
    cal = cfg.get("calendar") or {}
    return bool(hook) and hook in cal.get("hooks", []) + cal.get("tomorrow", []) + cal.get("today", [])


def outro_options(cfg: dict, subject: str) -> list[str]:
    """The closing call to action after the reveal (a subject's look.outros), e.g. a share prompt.
    {next} is the next day's number. Scored by shares per view, not watch time."""
    return cfg["subjects"][subject]["look"].get("outros") or [FOLLOW]


def sound_options(cfg: dict) -> list[str]:
    return cfg.get("openers", {}).get("sounds", list(SOUNDS))


def visual_options(cfg: dict) -> list[str]:
    return cfg.get("openers", {}).get("visuals", list(VISUALS))
