// The site's motion vocabulary: one spring, one lerp, one FLIP.
export const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

// stiffness 420, damping 26, mass 0.85: settles in about half a second with a
// slight overshoot. Simulated once and handed to WAAPI as a linear() easing.
function sampleSpring({ stiffness = 420, damping = 26, mass = 0.85 } = {}) {
  const dt = 1 / 240;
  const curve = [0];
  let x = 0;
  let v = 0;
  for (let t = 0; t < 2; t += dt) {
    v += ((-stiffness * (x - 1) - damping * v) / mass) * dt;
    x += v * dt;
    curve.push(x);
    if (Math.abs(1 - x) < 5e-4 && Math.abs(v) < 5e-3) break;
  }
  const points = 48;
  const stops = [];
  for (let i = 0; i <= points; i++) {
    const value = curve[Math.round((i / points) * (curve.length - 1))];
    stops.push(i === points ? 1 : +value.toFixed(4));
  }
  return { easing: `linear(${stops.join(", ")})`, duration: Math.round(curve.length * dt * 1000) };
}

const spring = sampleSpring();
const linearSupported = CSS.supports("animation-timing-function", "linear(0, 1)");
export const SPRING = linearSupported
  ? spring
  : { easing: "cubic-bezier(.22, 1, .36, 1)", duration: 650 };

/** Animate with the spring. Skipped entirely under reduced motion. */
export function springTo(el, keyframes, options = {}) {
  if (reducedMotion.matches) return null;
  return el.animate(keyframes, { ...SPRING, ...options });
}

/**
 * FLIP with real springs: each element keeps its own offset and velocity, so
 * when the layout changes again mid-flight it carries its momentum instead of
 * restarting from rest. Used where the layout changes every frame.
 *
 * Elements that wrapped onto another line fade in place instead: a token
 * flying 800px across a paragraph reads as noise, not as reflow.
 */
export class SpringLayout {
  constructor({ stiffness = 420, damping = 26, mass = 0.85 } = {}) {
    this.k = stiffness;
    this.c = damping;
    this.m = mass;
    this.moving = new Map();
    this.frame = 0;
    this.last = 0;
  }

  update(elements, mutate) {
    const before = new Map();
    for (const el of elements) before.set(el, el.getBoundingClientRect());
    mutate();
    if (reducedMotion.matches) return;
    for (const [el, old] of before) {
      const state = this.moving.get(el);
      if (!el.isConnected) {
        this.moving.delete(el);
        continue;
      }
      const now = el.getBoundingClientRect();
      if (Math.abs(old.top - now.top) > now.height / 2) {
        if (state) this.settle(el);
        el.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 350, easing: "ease-out" });
        continue;
      }
      const x = old.left - now.left + (state?.x ?? 0);
      if (Math.abs(x) < 0.5 && !state) continue;
      this.moving.set(el, { x, v: state?.v ?? 0 });
      // the style must always match the state, or the next measurement lies
      el.style.transform = `translateX(${x.toFixed(2)}px)`;
    }
    if (this.moving.size && !this.frame) {
      this.last = performance.now();
      this.frame = requestAnimationFrame((t) => this.step(t));
    }
  }

  step(now) {
    // dt is clamped so a background tab does not fling everything at once
    const dt = Math.min((now - this.last) / 1000, 1 / 30);
    this.last = now;
    for (const [el, s] of this.moving) {
      s.v += ((-this.k * s.x - this.c * s.v) / this.m) * dt;
      s.x += s.v * dt;
      if (Math.abs(s.x) < 0.3 && Math.abs(s.v) < 3) this.settle(el);
      else el.style.transform = `translateX(${s.x.toFixed(2)}px)`;
    }
    this.frame = this.moving.size ? requestAnimationFrame((t) => this.step(t)) : 0;
  }

  settle(el) {
    this.moving.delete(el);
    el.style.transform = "";
  }
}

// Numbers that count toward their target with cur += (target - cur) * 0.085
// per frame. One shared loop for every counter on the page.
const counters = new Set();
let frame = 0;

function tick() {
  frame = 0;
  for (const counter of counters) {
    counter.current += (counter.target - counter.current) * 0.085;
    if (Math.abs(counter.target - counter.current) < counter.epsilon) {
      counter.current = counter.target;
      counters.delete(counter);
    }
    counter.render();
  }
  if (counters.size) frame = requestAnimationFrame(tick);
}

export class Counter {
  constructor(el, decimals = 0) {
    this.el = el;
    this.decimals = decimals;
    this.current = 0;
    this.target = 0;
    this.epsilon = decimals ? 0.5 * 10 ** -decimals : 0.5;
    this.format = new Intl.NumberFormat("en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  }

  set(value) {
    this.target = value;
    if (reducedMotion.matches || document.hidden) {
      this.current = value;
      this.render();
      return;
    }
    counters.add(this);
    if (!frame) frame = requestAnimationFrame(tick);
  }

  render() {
    this.el.textContent = this.format.format(this.current);
  }
}
