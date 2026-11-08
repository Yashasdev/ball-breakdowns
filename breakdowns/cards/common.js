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
