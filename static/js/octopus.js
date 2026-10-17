'use strict';
// Adapted from o5s's `o5s ui` (../o5s/crates/server/src/index.html).
// The octopus is a wall of code; the creature is the code's light. Its shape
// is drawn into an offscreen canvas one pixel per character cell, and each
// cell's character takes that pixel's brightness.
const $ = id => document.getElementById(id);
const still = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const SOURCE = `template IsZero() { signal input in; signal output out; signal inv;
    inv <-- in != 0 ? 1 / in : 0; out <== -in * inv + 1; in * out === 0; }
fn mutate(trace: &mut Trace, rng: &mut Rng) -> Input { let k = rng.pick(trace.signals()); trace.flip(k) }
theorem sound : ∀ w, Constraints C w → Spec C w
if let Some(w) = solver.find(|w| c.holds(w) && !spec.holds(w)) { report(Bug::UnderConstrained(w)) }
fn bound(node: &Node) -> u64 { node.cost + node.remaining.iter().map(|s| s.lower()).sum::<u64>() }
while let Some(node) = frontier.pop() { if bound(&node) >= best { continue; } frontier.extend(node.branch()); }
theorem zkvm_equiv : ∀ p s, run_spec p s = run_circuit p s
assert!(verify(&vk, &proof, &public_inputs)); let witness = prover.witness(&circuit, &inputs)?;
loss = criterion(model(x), y); grad = torch.autograd.grad(loss, params); x_rec = invert(logits)
`.replace(/\s+/g, ' ');
const CW = 7.2, CH = 13, FONT = '12px "IBM Plex Mono", Menlo, monospace';
// Cell colour from the shape: red is flesh, green the eye, blue the pupil.
function flesh(r, alpha) {
  const k = r / 255;
  return `rgba(${Math.round(170 + 85 * k)},${Math.round(28 + 70 * k * k)},${Math.round(24 + 64 * k * k)},${alpha + k * (1 - alpha)})`;
}

const octopus = (() => {
  const wall = $('wall');
  if (!wall || !window.requestAnimationFrame) return {running() {}};
  const ctx = wall.getContext('2d');
  // Finer cells than the rest of the page, so a short hero still has room
  // for an octopus with eyes.
  const CW = 4.8, CH = 8.7, FONT = '8px "IBM Plex Mono", Menlo, monospace';
  const shape = document.createElement('canvas'), sctx = shape.getContext('2d', {willReadFrequently: true});
  let cols = 0, rows = 0, cells = [], width = 0, height = 0, on = false, frame = 0;
  let mouse = {x: 0.7, y: 0.4}, blinkAt = 0, arms = [];
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = wall.clientWidth; height = wall.clientHeight;
    wall.width = Math.round(width * dpr); wall.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cols = Math.ceil(width / CW); rows = Math.ceil(height / CH);
    shape.width = cols; shape.height = rows;
    cells = [];
    for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) cells.push(SOURCE[(y * 131 + x) % SOURCE.length]);
  }
  // Octopus geometry in cell units. It keeps o5s's proportions (rx ≈ 1.17 ry,
  // from mantle top at cy - 1.5 ry to arm tips at cy + 2.4 ry), scaled to fit
  // below the header and pushed to the right edge.
  function body(t) {
    const top = 84 / CH, ry = Math.max(3, (rows - top - 1) / 3.9), rx = ry * 1.17;
    return {cx: cols - rx * 3.2 - 3, cy: top + ry * 1.5 + Math.sin(t * 0.0009) * ry * 0.1, rx, ry};
  }
  function init() {
    arms = Array.from({length: 8}, (_, i) => ({i, phase: i * 0.8}));
  }
  function armPoints(arm, b, t) {
    // Waves, curls and widths are o5s's, which drew a body with ry ≈ 14.5.
    const n = 22, base = (arm.i - 3.5) / 3.5, points = [], k = b.ry / 14.5;
    const bx = b.cx + base * b.rx * 0.8, by = b.cy + b.ry * 0.78;
    const goal = {x: b.cx + base * b.rx * 2.6, y: b.cy + b.ry * 2.4 + Math.abs(base) * -4 * k}, side = base < 0 ? -1 : 1;
    for (let s = 0; s <= n; s++) {
      const u = s / n;
      // A slow travelling wave, and a curl that tightens toward the tip.
      const wave = Math.sin(t * 0.0016 + arm.phase + u * 4.4) * 2.4 * k * u * u;
      const curl = u * u * u * 5 * k * side * (0.6 + 0.4 * Math.sin(t * 0.0009 + arm.phase));
      points.push({x: bx + (goal.x - bx) * u + wave * 1.4 + curl, y: by + (goal.y - by) * u + wave - curl * 0.6 * u,
        w: (Math.pow(1 - u, 1.4) * 4.2 + 0.35) * k});
    }
    return points;
  }
  function drawShape(t) {
    sctx.fillStyle = '#000'; sctx.fillRect(0, 0, cols, rows);
    const b = body(t);
    sctx.lineCap = 'round';
    for (const arm of arms) {
      const pts = armPoints(arm, b, t);
      for (let s = 1; s < pts.length; s++) {
        sctx.strokeStyle = `rgb(${Math.round(150 + 90 * (1 - s / pts.length))},0,0)`;
        sctx.lineWidth = pts[s].w;
        sctx.beginPath(); sctx.moveTo(pts[s - 1].x, pts[s - 1].y); sctx.lineTo(pts[s].x, pts[s].y); sctx.stroke();
      }
    }
    // The mantle swept back over the head, shaded from below; the head band carries the eyes.
    const lean = -0.42 + Math.sin(t * 0.0007) * 0.05;
    const mx = b.cx + b.rx * 0.35, my = b.cy - b.ry * 0.55;
    const grad = sctx.createLinearGradient(mx - b.rx, my - b.ry * 1.2, b.cx, b.cy + b.ry);
    grad.addColorStop(0, 'rgb(120,0,0)'); grad.addColorStop(0.6, 'rgb(205,0,0)'); grad.addColorStop(1, 'rgb(255,0,0)');
    sctx.fillStyle = grad;
    sctx.beginPath(); sctx.ellipse(mx, my, b.rx * 1.05, b.ry * 0.95, lean, 0, Math.PI * 2); sctx.fill();
    sctx.beginPath(); sctx.ellipse(b.cx, b.cy + b.ry * 0.35, b.rx * 0.92, b.ry * 0.48, 0, 0, Math.PI * 2); sctx.fill();
    // Eyes: a dim iris with a horizontal slit pupil that follows the pointer.
    const lid = t < blinkAt + 380 ? Math.sin((t - blinkAt) / 380 * Math.PI) : 0;
    const ey = b.cy + b.ry * 0.28, gx = (mouse.x * cols - b.cx) / cols, gy = (mouse.y * rows - ey) / rows;
    for (const side of [-1, 1]) {
      const ex = b.cx + side * b.rx * 0.6, r = Math.max(2.6, b.rx * 0.17);
      sctx.fillStyle = 'rgb(0,200,0)';
      sctx.beginPath(); sctx.ellipse(ex, ey, r * 1.2, r * 0.8 * (1 - lid * 0.9), side * 0.15, 0, Math.PI * 2); sctx.fill();
      if (lid < 0.6) {
        sctx.fillStyle = 'rgb(0,0,255)';
        const px = ex + Math.max(-1, Math.min(1, gx * 5)) * r * 0.35, py = ey + Math.max(-1, Math.min(1, gy * 5)) * r * 0.2;
        sctx.fillRect(px - r * 0.8, py - 0.55, r * 1.6, 1.1);
      }
    }
    if (t > blinkAt + 5000 + Math.random() * 6000) blinkAt = t;
    return sctx.getImageData(0, 0, cols, rows).data;
  }
  function draw(t) {
    const px = drawShape(t);
    ctx.fillStyle = '#050506'; ctx.fillRect(0, 0, width, height);
    ctx.font = FONT; ctx.textBaseline = 'top';
    const drift = Math.floor(t / 900);
    for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
      const i = y * cols + x, r = px[i * 4], g = px[i * 4 + 1], bl = px[i * 4 + 2];
      const ch = cells[(i + (y % 4 === 0 ? drift : 0)) % cells.length];
      if (ch === ' ' || bl > 120) continue;
      ctx.fillStyle = g > 100 ? 'rgb(245,214,160)' : r > 30 ? flesh(r, 0.42) : 'rgba(235,230,219,0.07)';
      ctx.fillText(ch, x * CW, y * CH);
    }
  }
  function loop(t) {
    if (!on) return;
    if (frame++ % 2 === 0) draw(t);
    requestAnimationFrame(loop);
  }
  window.addEventListener('resize', () => { resize(); if (still) draw(0); });
  window.addEventListener('pointermove', e => { mouse = {x: e.clientX / Math.max(1, width), y: e.clientY / Math.max(1, height)}; });
  resize(); init();
  return {running(want) {
    if (still) { if (want) { resize(); draw(0); } return; }
    if (want && !on) { on = true; requestAnimationFrame(loop); }
    if (!want) on = false;
  }};
})();

// A small octopus in the corner, once the hero is gone. When a section is
// chosen in the navigation, it reaches an arm out to that section's heading.
const guide = (() => {
  const canvas = $('guide-octopus'), arm = $('tentacle');
  if (!canvas || !arm || !window.requestAnimationFrame) return {visible() {}};
  const c = canvas.getContext('2d');
  const shape = document.createElement('canvas'), sc = shape.getContext('2d', {willReadFrequently: true});
  const ac = arm.getContext('2d'), ashape = document.createElement('canvas'), as = ashape.getContext('2d', {willReadFrequently: true});
  const COLS = 34, ROWS = 20, GW = 5.6, GH = 9;
  let acols = 0, arows = 0, aw = 0, ah = 0, on = false, blink = 0, pointer = null, target = null, grown = 0, release = 0;
  function fitArm() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    aw = window.innerWidth; ah = window.innerHeight;
    arm.width = Math.round(aw * dpr); arm.height = Math.round(ah * dpr); ac.setTransform(dpr, 0, 0, dpr, 0, 0);
    acols = Math.ceil(aw / CW); arows = Math.ceil(ah / CH); ashape.width = acols; ashape.height = arows;
  }
  fitArm(); window.addEventListener('resize', fitArm);
  window.addEventListener('pointermove', e => { pointer = {x: e.clientX, y: e.clientY}; });
  canvas.width = COLS * GW * 2; canvas.height = ROWS * GH * 2; c.scale(2, 2);
  shape.width = COLS; shape.height = ROWS;
  function origin() {
    const r = canvas.getBoundingClientRect();
    return {x: r.left + r.width * 0.3, y: r.top + r.height * 0.72};
  }
  // The arm lands on the heading's right end, where the text stops.
  function aim() {
    const r = target.getBoundingClientRect();
    return {x: Math.min(aw - 220, r.right + 14), y: r.top + r.height / 2};
  }
  function drawOctopus(t) {
    sc.fillStyle = '#000'; sc.fillRect(0, 0, COLS, ROWS);
    const cx = COLS * 0.55, cy = ROWS * 0.36;
    sc.lineCap = 'round';
    for (let i = 0; i < 7; i++) {
      const base = (i - 3) / 3, sx = cx + base * 6, sy = cy + 4;
      sc.strokeStyle = 'rgb(180,0,0)';
      sc.beginPath(); sc.moveTo(sx, sy);
      for (let u = 0.1; u <= 1; u += 0.1) {
        const wave = Math.sin(t * 0.002 + i + u * 5) * 1.4 * u;
        sc.lineWidth = (1 - u) * 2.2 + 0.4;
        sc.lineTo(sx + base * 7 * u + wave, sy + u * 9 - Math.abs(base) * u * 2);
      }
      sc.stroke();
    }
    const grad = sc.createLinearGradient(cx, cy - 8, cx, cy + 5);
    grad.addColorStop(0, 'rgb(130,0,0)'); grad.addColorStop(1, 'rgb(255,0,0)');
    sc.fillStyle = grad;
    sc.beginPath(); sc.ellipse(cx + 2, cy - 3, 8.5, 6.5, -0.4, 0, Math.PI * 2); sc.fill();
    sc.beginPath(); sc.ellipse(cx, cy + 1.5, 7, 3.6, 0, 0, Math.PI * 2); sc.fill();
    // Eyes turn toward what it reaches for, or else the pointer.
    let gx = -1, gy = 0;
    const o = origin(), look = target ? aim() : pointer;
    if (look) { const d = Math.hypot(look.x - o.x, look.y - o.y) || 1; gx = (look.x - o.x) / d; gy = (look.y - o.y) / d; }
    const lid = t < blink + 320 ? Math.sin((t - blink) / 320 * Math.PI) : 0;
    if (t > blink + 4000 + Math.random() * 4000) blink = t;
    for (const side of [-1, 1]) {
      const ex = cx + side * 4.2, ey = cy + 1.2;
      sc.fillStyle = 'rgb(0,200,0)';
      sc.beginPath(); sc.ellipse(ex, ey, 1.8, 1.3 * (1 - lid * 0.9), 0, 0, Math.PI * 2); sc.fill();
      if (lid < 0.6) { sc.fillStyle = 'rgb(0,0,255)'; sc.fillRect(ex - 1 + gx * 0.7, ey - 0.4 + gy * 0.5, 2, 0.8); }
    }
    const px = sc.getImageData(0, 0, COLS, ROWS).data;
    c.clearRect(0, 0, COLS * GW, ROWS * GH);
    c.font = '9px "IBM Plex Mono", Menlo, monospace'; c.textBaseline = 'top';
    for (let y = 0; y < ROWS; y++) for (let x = 0; x < COLS; x++) {
      const i = y * COLS + x, r = px[i * 4], g = px[i * 4 + 1], b = px[i * 4 + 2];
      if (b > 120 || (r < 30 && g < 100)) continue;
      const ch = SOURCE[(y * 53 + x + Math.floor(t / 1200)) % SOURCE.length];
      c.fillStyle = g > 100 ? 'rgb(245,214,160)' : flesh(r, 0.45);
      c.fillText(ch === ' ' ? '·' : ch, x * GW, y * GH);
    }
  }
  function drawArm(t) {
    ac.clearRect(0, 0, aw, ah);
    if (!target) return;
    grown = release && t > release ? Math.max(0, grown - 0.04) : Math.min(1, grown + 0.03);
    if (grown <= 0) { target = null; return; }
    const o = origin(), a = aim();
    const dx = a.x - o.x, dy = a.y - o.y, len = Math.hypot(dx, dy) || 1, nx = -dy / len, ny = dx / len;
    const wob = Math.sin(t * 0.003) * Math.min(40, len * 0.08), wob2 = Math.cos(t * 0.0023) * Math.min(30, len * 0.06);
    const c1 = {x: o.x + dx * 0.3 + nx * (len * 0.18 + wob), y: o.y + dy * 0.3 + ny * (len * 0.18 + wob)};
    const c2 = {x: o.x + dx * 0.72 - nx * (len * 0.1 + wob2), y: o.y + dy * 0.72 - ny * (len * 0.1 + wob2)};
    const reach = 1 - Math.pow(1 - grown, 3), n = 48;
    const at = u => {
      const v = 1 - u;
      return {x: v * v * v * o.x + 3 * v * v * u * c1.x + 3 * v * u * u * c2.x + u * u * u * a.x,
              y: v * v * v * o.y + 3 * v * v * u * c1.y + 3 * v * u * u * c2.y + u * u * u * a.y};
    };
    // Rasterize the grown part into cells, tapering toward the tip.
    as.fillStyle = '#000'; as.fillRect(0, 0, acols, arows);
    as.lineCap = 'round';
    let prev = at(0);
    for (let k = 1; k <= n; k++) {
      const p = at(k / n * reach);
      as.strokeStyle = `rgb(${Math.round(255 - 110 * (k / n))},0,0)`;
      as.lineWidth = (1 - k / n) * 2.2 + 0.8;
      as.beginPath(); as.moveTo(prev.x / CW, prev.y / CH); as.lineTo(p.x / CW, p.y / CH); as.stroke();
      prev = p;
    }
    const px = as.getImageData(0, 0, acols, arows).data, flow = Math.floor(t / 120);
    ac.font = FONT; ac.textBaseline = 'top';
    for (let y = 0; y < arows; y++) for (let x = 0; x < acols; x++) {
      const r = px[(y * acols + x) * 4];
      if (r < 40) continue;
      const ch = SOURCE[(y * 131 + x + flow) % SOURCE.length];
      // A dark cell under each character, so the arm reads over the text.
      ac.fillStyle = 'rgba(5,5,6,0.88)'; ac.fillRect(x * CW, y * CH, CW, CH);
      ac.fillStyle = flesh(r, 0.5);
      ac.fillText(ch === ' ' ? '·' : ch, x * CW, y * CH);
    }
  }
  function loop(t) {
    if (!on) return;
    drawOctopus(t); drawArm(t);
    requestAnimationFrame(loop);
  }
  canvas.addEventListener('click', () => window.scrollTo({top: 0, behavior: still ? 'auto' : 'smooth'}));
  document.querySelectorAll('.nav a[href^="#"], .bio a[href^="#"]').forEach(link => link.addEventListener('click', () => {
    const section = document.querySelector(link.getAttribute('href'));
    const heading = section && section.querySelector('h2');
    if (!heading || still) return;
    if (target !== heading) grown = 0;
    target = heading; release = performance.now() + 2600;
  }));
  return {visible(want) {
    canvas.classList.toggle('show', want);
    if (still) { if (want) drawOctopus(0); return; }
    if (want && !on) { on = true; requestAnimationFrame(loop); }
    if (!want) { on = false; target = null; ac.clearRect(0, 0, aw, ah); }
  }};
})();

// Run the big octopus only while the hero is on screen, the small one after.
const stage = document.querySelector('.stage');
if (stage && 'IntersectionObserver' in window) {
  new IntersectionObserver(([entry]) => {
    octopus.running(entry.isIntersecting);
    guide.visible(!entry.isIntersecting && window.innerWidth > 1000);
  }, {threshold: 0.05}).observe(stage);
} else octopus.running(true);

// The frame gets a ground once the page scrolls, and the nav marks the section in view.
const frameTop = $('frame'), links = [...document.querySelectorAll('.nav a[href^="#"]')];
const marked = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
function onScroll() {
  frameTop.classList.toggle('scrolled', window.scrollY > 40);
  let current = null;
  for (const s of marked) if (s.getBoundingClientRect().top < window.innerHeight * 0.35) current = s.id;
  links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + current));
}
window.addEventListener('scroll', onScroll, {passive: true});
onScroll();
