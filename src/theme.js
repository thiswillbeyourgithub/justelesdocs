/* Light/dark with three states, matching ../justelesRCP/src/theme.js in behaviour.

   The three states are deliberate: "auto" sets no attribute and leaves the OS
   preference in charge through the CSS media query, while "light" and "dark" stamp
   data-theme on <html> and win over it. A two-state toggle cannot express "follow
   the system", which is what most readers actually want. */

import { load, save } from "./store.js";

const ORDER = ["auto", "light", "dark"];

function stored() {
  const v = load("theme");
  return ORDER.includes(v) ? v : "auto";
}

function paint(mode) {
  if (mode === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", mode);
}

let mode = stored();
paint(mode);

/** Advance to the next mode and persist it. Returns the new mode. */
export function cycleTheme() {
  mode = ORDER[(ORDER.indexOf(mode) + 1) % ORDER.length];
  save("theme", mode);
  paint(mode);
  return mode;
}

/** A short label for the toggle, per mode. Not translated: these are symbols. */
export function themeLabel() {
  return { auto: "◐", light: "☀", dark: "☾" }[mode];
}
