import { loadCl100k } from "./cl100k.js";
import { mountExplainer } from "./explainer.js";
import { reducedMotion } from "./motion.js";
import { mountPlayground } from "./playground.js";
import { mountTrainer } from "./train.js";

const encoderReady = loadCl100k(new URL("../data/cl100k_base.bin.gz", import.meta.url));

const sections = {};
for (const section of document.querySelectorAll(".view")) sections[section.dataset.view] = section;

const views = {
  gpt4: mountPlayground(sections.gpt4, encoderReady),
  train: mountTrainer(sections.train),
  how: mountExplainer(sections.how, encoderReady),
};

const pill = document.querySelector(".pill");
const links = [...pill.querySelectorAll(".pill-btn")];
let current = null;
let swap = 0;

function show(name) {
  if (!(name in views)) name = "gpt4";
  if (name === current) return;
  const previous = current && sections[current];
  current = name;

  links.forEach((link, i) => {
    if (link.dataset.view === name) {
      link.setAttribute("aria-current", "page");
      pill.style.setProperty("--at", i);
    } else {
      link.removeAttribute("aria-current");
    }
  });

  const enter = () => {
    for (const section of Object.values(sections)) {
      section.hidden = section !== sections[name];
      section.classList.remove("leave");
    }
    const next = sections[name];
    next.classList.add("enter");
    scrollTo({ top: 0, behavior: "instant" });
    views[name].shown?.();
    // two frames: the first one lays out the start state, the second animates
    requestAnimationFrame(() => requestAnimationFrame(() => next.classList.remove("enter")));
  };

  clearTimeout(swap);
  if (!previous || reducedMotion.matches) {
    enter();
  } else {
    previous.classList.add("leave");
    swap = setTimeout(enter, 300);
  }
}

addEventListener("hashchange", () => show(location.hash.slice(1)));
show(location.hash.slice(1));
