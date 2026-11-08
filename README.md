# How Crazy Was It? (Ball Knowledge breakdowns)

Short narrated breakdowns of famous games for Ball Knowledge Daily (Instagram, TikTok, YouTube). This repo only
**makes** the videos; posting happens in the private `Yashasdev/act-reels` repo, which downloads finished videos
from this repo's [videos release](../../releases/tag/videos).

- `items.json`: the stories, in posting order. Append only; never reorder or renumber. Every fact is checked
  against a box score or play-by-play, and each story lists its sources.
- `breakdowns/build.py`: voices each line (Chatterbox, checked word by word with faster-whisper and redone if a
  word is wrong), renders the page `breakdowns/template.html` frame by frame in headless Chrome, and publishes
  `<id>.mp4`, `<id>.jpg` (cover), `<id>.json` and `index.json`.
- `cards.json` and `breakdowns/cards.py`: three short formats with no narration (on-screen text, our own music and
  sounds, 12-18 s, built to loop): `guess` (Guess the Player), `hilo` (Higher or Lower) and `rank` (Rank These), each
  with its own page in `breakdowns/cards/`. Append only, ids numbered per format (`guess-01`, ...). Items marked
  `"placeholder": true` are only for testing the pages and are never published. Check a file with
  `python -m breakdowns.cards --check --items FILE`; render one with `python -m breakdowns.cards guess-01`.
  Their `<id>.json` adds `format`, `answer` and `why` for the caption (stories have `"format": "crazy"`).
- `.github/workflows/build.yml`: runs the build when `items.json` changes and once a day. Public repos get free,
  unlimited Actions minutes, which is why this lives outside the private repo.

Local run (Python 3.11): `pip install torch==2.6.0 torchaudio==2.6.0 && pip install -r requirements.txt &&
python -m playwright install chromium`, then `python -m breakdowns.build --dry --max 1` (renders into `out/`,
publishes nothing).

## Credits

- Narration: [Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI (MIT). Its reference voice,
  `assets/voice/ref_onyx.wav`, was spoken by the `am_onyx` voice of [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)
  (Apache 2.0). No real person's voice is imitated.
- Music and sound effects: composed by code in `bk/` (nothing sampled or downloaded).
- Fonts: Inter (SIL Open Font License, `assets/fonts/LICENSE-Inter.txt`) and the fonts in `assets/fonts/breakdown/`
  (see the license files there).
- Team names and scores are facts. No league logos, footage or photos are used.
