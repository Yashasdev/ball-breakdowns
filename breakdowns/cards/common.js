// Shared by the card pages: easing, text fitting, number formatting, and the moving film grain.
const $ = (id) => document.getElementById(id);
const spring = (x, k = 7, w = 11) => x <= 0 ? 0 : 1 - Math.exp(-k * x) * Math.cos(w * x);
const ease = (x) => x <= 0 ? 0 : x >= 1 ? 1 : 1 - Math.pow(1 - x, 3);
const easeIO = (x) => x <= 0 ? 0 : x >= 1 ? 1 : x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

// Put text in a fixed-size .fit box at the largest font size (max..min px) that fits; mark it if even min doesn't.
// Breaks only between words, never inside one ("Antetokounmpo" shrinks rather than splitting).
function fit(el, text, max, min, html = false) {
  el.innerHTML = `<span>${html ? text : esc(text)}</span>`;
  const span = el.firstChild;
  const cs = getComputedStyle(el);
  const w = el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  const h = el.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
  // offset sizes ignore transforms, so a box mid-animation measures the same as at rest
  const over = () => span.offsetHeight > h + 1 || span.scrollWidth > w + 1 || span.offsetWidth > w + 1;
  let s = max;
  el.style.fontSize = s + "px";
  while (s > min && over()) { s -= 2; el.style.fontSize = s + "px"; }
  el.dataset.fit = over() ? "overflow" : "ok";
  return s;
}
// every box whose text doesn't fit (the tests and `cards --check` call this)
function overflowing() {
  return [...document.querySelectorAll('[data-fit="overflow"]')].map((e) => (e.id || e.className) + ": " + e.textContent);
}

// Key elements outside the phone's safe area, at this moment (the tests and `cards --check` call this; L is
// make.SAFE). Elements marked data-safe: "head" (header text), "text" (measured by its text) or "box" (its box);
// hidden ones are skipped.
function unsafe(L) {
  const out = [];
  for (const el of document.querySelectorAll("[data-safe]")) {
    let o = 1;
    for (let e = el; e && e !== document.body; e = e.parentElement) {
      const cs = getComputedStyle(e);
      if (cs.display === "none" || cs.visibility === "hidden") o = 0;
      o *= parseFloat(cs.opacity);
    }
    if (o < 0.05) continue;
    let r;
    if (el.dataset.safe === "box") r = el.getBoundingClientRect();
    else { const g = document.createRange(); g.selectNodeContents(el); r = g.getBoundingClientRect(); }
    if (!r.width || !r.height) continue;
    const name = el.id || el.className, why = [];
    if (r.top < (el.dataset.safe === "head" ? L.head : L.top)) why.push(`top ${Math.round(r.top)}`);
    if (r.bottom > L.bottom) why.push(`bottom ${Math.round(r.bottom)}`);
    if (r.left < L.left) why.push(`left ${Math.round(r.left)}`);
    if (r.right > L.right) why.push(`right ${Math.round(r.right)}`);
    if (r.bottom > L.low_from && r.right > L.low_right) why.push(`right ${Math.round(r.right)} in the lower half`);
    if (why.length) out.push(`${name}: ${why.join(", ")}`);
  }
  return out;
}

// a shown value ("38,652", "27.1", "41.2%") at share k of the way from 0, in the same style
function partial(shown, k) {
  const m = String(shown).match(/^(\D*?)(\d[\d,]*(?:\.\d+)?)(.*)$/);
  if (!m) return shown;
  const dec = (m[2].split(".")[1] || "").length, commas = m[2].includes(",");
  const v = parseFloat(m[2].replace(/,/g, "")) * k;
  let s = v.toFixed(dec);
  if (commas) { const [a, b] = s.split("."); s = a.replace(/\B(?=(\d{3})+(?!\d))/g, ",") + (b ? "." + b : ""); }
  return m[1] + s + m[3];
}

function grain(t) { $("grain").style.transform = `translate(${(t * 41) % 40 - 20}px, ${(t * 29) % 40 - 20}px)`; }

// Fonts load only once something uses them, so load all three before setup() measures any text.
const FONTS = Promise.all(["50px Anton", "800 50px Inter", "30px Mono"].map((f) => document.fonts.load(f)));

// Share formats: the opening target line ("Send this to a Falcons fan.") and the end card (send line + question).
// Pages include SHARE_HTML's elements (#open, #endc) and call these from setup() and render().
const SHARE_HTML = `<div id="open"><div id="openTo" class="mono" data-safe="text">SEND THIS TO</div>
  <div id="openWho" class="fit anton" data-safe="text"></div><div id="openSub" class="fit inter" data-safe="text"></div></div>
  <div id="endc"><div id="endTop" class="fit anton" data-safe="text"></div><div id="endSend" class="fit anton" data-safe="text"></div>
  <div id="endAsk" class="fit inter" data-safe="box"></div></div>`;
function shareSetup(tl, sub, top) {
  document.getElementById("stage").insertAdjacentHTML("beforeend", SHARE_HTML);
  const who = tl.opening.replace(/^Send this to\s+/, "").toUpperCase();
  fit($("openWho"), who, 150, 70);
  if (sub) fit($("openSub"), sub, 54, 30); else $("openSub").style.display = "none";
  if (top) fit($("endTop"), top, 64, 36); else $("endTop").style.display = "none";
  fit($("endSend"), tl.send.toUpperCase(), 110, 54);
  fit($("endAsk"), tl.ask, 56, 30);
}
// openUntil: when the target line gives way; endAt: when the end card comes in
function shareRender(t, openUntil, endAt) {
  const o = $("open"), p = spring(t, 8, 12), out = clamp((t - openUntil) / 0.18);
  o.style.opacity = 1 - out;
  o.style.display = out >= 1 ? "none" : "block";
  $("openWho").style.transform = `scale(${1.5 - 0.5 * p}) rotate(${-2 * (1 - p)}deg)`;
  $("openTo").style.opacity = clamp(t / 0.1);
  $("openSub").style.opacity = clamp((t - 0.35) / 0.2);
  const e = $("endc"), y = t - endAt;
  e.style.opacity = clamp(y / 0.2);
  e.style.display = y < 0 ? "none" : "block";
  $("endSend").style.transform = `scale(${y > 0 ? 1.3 - 0.3 * spring(y, 8, 12) : 1.3})`;
  $("endAsk").style.transform = `translateY(${(1 - ease((y - 0.4) / 0.35)) * 60}px)`;
  $("endAsk").style.opacity = clamp((y - 0.4) / 0.2);
  $("endTop").style.opacity = clamp((y - 0.1) / 0.2);
}
