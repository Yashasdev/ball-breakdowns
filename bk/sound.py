"""Procedural sound design for a reel: an opening sound, beat bed, pops, clock ticks,
a riser into the reveal, and an impact + bell on the answer.

The opening sound is the reel's first second, the "audio hook". OPENERS lists the choices; the
tuner picks one per reel. "whoosh" is the original.

Everything is synthesized here, so there are no licensing questions.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .openers import SOUNDS as OPENERS

SR = 44100


def _t(dur: float) -> np.ndarray:
    return np.arange(int(dur * SR)) / SR


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-np.maximum(t - attack, 0) / decay)


def _lowpass(x: np.ndarray, cutoff) -> np.ndarray:
    """One-pole lowpass; cutoff may be a scalar or a per-sample array (Hz)."""
    cutoff = np.broadcast_to(np.asarray(cutoff, dtype=float), x.shape)
    alpha = 1 - np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):  # short sounds only, so a Python loop is fine
        acc += alpha[i] * (x[i] - acc)
        y[i] = acc
    return y


def kick() -> np.ndarray:
    t = _t(0.35)
    freq = 45 + 110 * np.exp(-t * 28)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    return np.sin(phase) * _env(len(t), 0.002, 0.12)


def hat(rng) -> np.ndarray:
    n = int(0.05 * SR)
    noise = rng.uniform(-1, 1, n)
    hp = noise - _lowpass(noise, 6000)
    return hp * _env(n, 0.001, 0.015)


def bass_note(freq: float, dur: float) -> np.ndarray:
    t = _t(dur)
    saw = 2 * ((t * freq) % 1) - 1
    tone = 0.6 * saw + 0.4 * np.sin(2 * np.pi * freq * t)
    env = np.minimum(1, t / 0.01) * np.minimum(1, (dur - t) / 0.05)
    return _lowpass(tone, 500) * env


def tick(high: bool = False) -> np.ndarray:
    t = _t(0.06)
    f1, f2 = (2400, 3600) if high else (1700, 2600)
    tone = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    return tone * _env(len(t), 0.0005, 0.012)


def pop(pitch: float = 1.0) -> np.ndarray:
    t = _t(0.09)
    freq = (500 + 700 * np.exp(-t * 40)) * pitch
    return np.sin(2 * np.pi * np.cumsum(freq) / SR) * _env(len(t), 0.002, 0.03)


def whoosh(rng, dur: float = 0.55) -> np.ndarray:
    n = int(dur * SR)
    noise = rng.uniform(-1, 1, n)
    x = np.linspace(0, 1, n)
    cutoff = 300 + 5000 * np.sin(np.pi * x) ** 2
    shaped = _lowpass(noise, cutoff)
    return shaped * np.sin(np.pi * x) ** 1.5 * 2.2


def crowd(rng, dur: float = 3.0) -> np.ndarray:
    """A stadium crowd erupting: band-limited noise (many voices) with a fast swell, a wobble and a long tail.
    Synthesized, so there's nothing to license."""
    n = int(dur * SR)
    x = np.linspace(0, 1, n)
    roar = _lowpass(rng.uniform(-1, 1, n), 2600) - _lowpass(rng.uniform(-1, 1, n), 260)
    wobble = 1 + 0.25 * np.sin(2 * np.pi * 3.1 * x * dur) * np.sin(2 * np.pi * 0.7 * x * dur)
    swell = np.minimum(1, x * dur / 0.35) * np.exp(-np.maximum(0, x * dur - 0.9) * 1.1)
    return roar * wobble * swell * 2.0


def impact(rng) -> np.ndarray:
    t = _t(0.9)
    freq = 38 + 70 * np.exp(-t * 9)
    boom = np.sin(2 * np.pi * np.cumsum(freq) / SR) * _env(len(t), 0.002, 0.28)
    n = int(0.25 * SR)
    burst = np.zeros_like(boom)
    burst[:n] = _lowpass(rng.uniform(-1, 1, n), 2500) * _env(n, 0.001, 0.05) * 1.5
    return boom + burst


def riser(rng, dur: float) -> np.ndarray:
    t = _t(dur)
    x = t / dur
    freq = 180 * (8 ** x)
    tone = np.sin(2 * np.pi * np.cumsum(freq) / SR) * 0.35
    noise = _lowpass(rng.uniform(-1, 1, len(t)), 800 + 7000 * x) * 0.9
    return (tone + noise) * x ** 2


def bell() -> np.ndarray:
    t = _t(1.6)
    out = np.zeros_like(t)
    for ratio, amp, dec in [(1, 1.0, 0.6), (2.0, 0.5, 0.35), (3.01, 0.3, 0.2), (4.2, 0.15, 0.12)]:
        out += amp * np.sin(2 * np.pi * 1046.5 * ratio * t) * np.exp(-t / dec)
    return out * np.minimum(1, t / 0.003) * 0.5


def buzzer() -> np.ndarray:
    """Game-show wrong-answer buzzer: two detuned square waves, low and harsh."""
    t = _t(0.55)
    sq = np.sign(np.sin(2 * np.pi * 146 * t)) + np.sign(np.sin(2 * np.pi * 151 * t))
    return _lowpass(sq * 0.5, 2200) * np.minimum(1, t / 0.005) * np.minimum(1, (0.55 - t) / 0.06)


def scratch(rng) -> np.ndarray:
    """Record scratch: noise through a filter that sweeps back and forth fast, with a pitch wobble."""
    n = int(0.42 * SR)
    t = np.arange(n) / SR
    sweep = 900 + 3500 * (0.5 + 0.5 * np.sin(2 * np.pi * 7.5 * t ** 0.8))
    noise = _lowpass(rng.uniform(-1, 1, n), sweep) - _lowpass(rng.uniform(-1, 1, n), 250) * 0.3
    wobble = np.sin(2 * np.pi * np.cumsum(220 + 180 * np.sin(2 * np.pi * 7.5 * t)) / SR) * 0.35
    return (noise * 1.6 + wobble) * _env(n, 0.003, 0.2)


def bass_drop() -> np.ndarray:
    """A deep 808-style drop: a sub tone sliding down, slightly overdriven so phones can hear it."""
    t = _t(1.1)
    freq = 32 + 140 * np.exp(-t * 6)
    tone = np.sin(2 * np.pi * np.cumsum(freq) / SR) * _env(len(t), 0.004, 0.45)
    return np.tanh(tone * 2.2) * 0.8




def _add(buf: np.ndarray, sound: np.ndarray, at: float, gain: float = 1.0, pan: float = 0.0) -> None:
    i = int(at * SR)
    if i >= buf.shape[0]:
        return
    s = sound[: buf.shape[0] - i] * gain
    buf[i:i + len(s), 0] += s * (1 - max(pan, 0))
    buf[i:i + len(s), 1] += s * (1 + min(pan, 0))


def build_soundtrack(path: Path, length: float, timer_start: float, reveal_at: float,
                     pop_times: list[float], seed: int, with_bed: bool = True, opener: str = "whoosh") -> Path:
    if opener not in OPENERS:
        raise ValueError(f"unknown opening sound {opener!r}")
    rng = np.random.default_rng(seed)
    buf = np.zeros((int(length * SR), 2))

    # Beat bed: 120 BPM, kick on beats, hats on off-beats, a minor-key bass line.
    if with_bed:
        beat = 0.5
        roots = [110.0, 87.31, 130.81, 98.0]  # A2 F2 C3 G2
        k, h = kick(), hat(rng)
        n_beats = int(length / beat)
        for b in range(n_beats):
            at = b * beat
            duck = 0.35 if reveal_at - 0.1 <= at < reveal_at + 1.0 else 1.0
            _add(buf, k, at, 0.55 * duck)
            _add(buf, h, at + beat / 2, 0.18 * duck, pan=0.3 if b % 2 else -0.3)
            if b % 4 == 0:
                _add(buf, bass_note(roots[(b // 4) % 4], beat * 4 - 0.02), at, 0.22 * duck)

    # Opening sound as the headline slams in. The original whoosh and impact are always drawn from rng,
    # so the rest of the soundtrack stays the same whichever opener plays; the others use their own.
    swoosh, hit = whoosh(rng), impact(rng)
    other = np.random.default_rng(seed + 1)
    if opener == "whoosh":
        _add(buf, swoosh, 0.0, 0.5)
        _add(buf, hit, 0.12, 0.7)
    elif opener == "buzzer":
        _add(buf, buzzer(), 0.0, 0.45)
        _add(buf, hit, 0.05, 0.5)
    elif opener == "scratch":
        _add(buf, scratch(other), 0.0, 0.6)
        _add(buf, hit, 0.4, 0.6)
    elif opener == "bass_drop":
        _add(buf, whoosh(other, 0.25), 0.0, 0.35)
        _add(buf, bass_drop(), 0.1, 0.9)
    for i, pt in enumerate(pop_times):
        _add(buf, pop(1 + 0.12 * i), pt, 0.45)

    # Countdown clock: a tick every second, doubling up in the last 3 seconds.
    sec = timer_start
    while sec < reveal_at - 0.05:
        left = reveal_at - sec
        urgent = left <= 3.01
        _add(buf, tick(high=urgent), sec, 0.55 if urgent else 0.35)
        if urgent:
            _add(buf, tick(high=True), sec + 0.5, 0.45)
        sec += 1.0

    _add(buf, riser(rng, 1.4), reveal_at - 1.4, 0.45)
    _add(buf, impact(rng), reveal_at, 1.0)
    _add(buf, bell(), reveal_at + 0.02, 0.7)

    # Fade the last half second so the loop back to the start is smooth.
    fade = int(0.5 * SR)
    buf[-fade:] *= np.linspace(1, 0, fade)[:, None]

    peak = np.max(np.abs(buf)) or 1.0
    pcm = (buf / peak * 0.89 * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path
