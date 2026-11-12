"""Stories and pain cards voiced on Yashas's Mac in a clone of his voice ("voice": "yashas"): GitHub's build never
builds them, so it can't overwrite them with the Chatterbox narrator; their published entries say which voice it is
(the poster adds a caption credit); and his recordings never enter this public repo."""

import json
import subprocess

import pytest

from breakdowns import build, cards
from breakdowns import voice as narrator

STORY = json.loads(build.ITEMS.read_text())[0]
PAIN = next(c for c in cards.load() if c["format"] == "pain")
AUDIO = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".aiff", ".pt", ".npy", ".npz", ".safetensors")


def ids(todo):
    return [i["id"] for i, _ in todo]


def test_github_never_builds_an_item_voiced_on_yashass_mac():
    mine, other = {**STORY, "id": "crazy-90", "voice": "yashas"}, {**STORY, "id": "crazy-91"}
    want = [(mine, build.story_hash(mine)), (other, build.story_hash(other))]
    assert ids(build.todo(want, set(), {}, 12)) == ["crazy-91"], "not published yet"
    done = {"crazy-90.mp4", "crazy-91.mp4"}
    stale = {"crazy-90": {"hash": "chatterbox"}, "crazy-91": {"hash": "chatterbox"}}
    assert ids(build.todo(want, done, stale, 12)) == ["crazy-91"], "published, but its hash changed (or BUILD was raised)"
    current = {i["id"]: {"hash": h} for i, h in want}
    assert build.todo(want, done, current, 12) == []


def test_the_same_goes_for_pain_cards():
    mine = {**PAIN, "voice": "yashas"}
    assert build.todo([(mine, cards.card_hash(mine))], set(), {}, 12) == []
    assert ids(build.todo([(PAIN, cards.card_hash(PAIN))], set(), {}, 12)) == [PAIN["id"]]


def test_published_entries_say_which_voice_narrates_them():
    result = {"seconds": 30.0, "music": "stadium-1"}
    assert build.post_info({**STORY, "voice": "yashas"}, result)["voice"] == "yashas-omnivoice"
    assert "voice" not in build.post_info({k: v for k, v in STORY.items() if k != "voice"}, result)
    assert cards.post_info({**PAIN, "voice": "yashas"}, result, "h")["voice"] == "yashas-omnivoice"
    assert "voice" not in cards.post_info({k: v for k, v in PAIN.items() if k != "voice"}, result, "h")


def test_marked_items_use_a_known_voice_and_have_a_voice_to_clone():
    for it in json.loads(build.ITEMS.read_text()) + cards.load():
        if "voice" in it:
            assert narrator.local_only(it), f"{it['id']}: unknown voice {it['voice']!r}"
            assert "beats" in it or cards.voiced(it), f"{it['id']}: only stories and pain cards are spoken"


def test_no_recordings_or_voice_prompts_in_the_repo():
    """The only audio in the repo is Kokoro's am_onyx reference (no real person). Yashas's recordings and saved voice
    prompts stay on his Mac, and out-yashas/ (local renders) is ignored."""
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=build.ROOT,
                           capture_output=True, text=True, check=True).stdout.split()
    assert [f for f in files if f.lower().endswith(AUDIO)] == ["assets/voice/ref_onyx.wav"]
    assert "out-yashas/" in (build.ROOT / ".gitignore").read_text().split()


def test_the_reference_clip_must_live_outside_the_repo(monkeypatch):
    from breakdowns import yashas
    monkeypatch.setenv("BK_VOICE_REF", str(narrator.REF))
    monkeypatch.setenv("BK_VOICE_REF_TEXT", "Some words.")
    with pytest.raises(SystemExit):
        yashas.ref_from_env()
    monkeypatch.delenv("BK_VOICE_REF")
    with pytest.raises(SystemExit):
        yashas.ref_from_env()
