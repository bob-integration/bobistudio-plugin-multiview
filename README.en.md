# Multiview

*[Version française](README.md)*

Composites several video sources into a single output flow. It is part of
[Bobi.Studio](https://github.com/bob-integration/bobistudio), a broadcast orchestrator built on
the ST 2110 / MXL bus.

---

## What it does

Each window reads a flow from the MXL bus, scales it down and places it. Around the picture come
the parts of a production wall: name strip, tally lamps, audio meters, viewfinder frame, clocks,
countdowns, text overlays.

The layout is described by a **template** — a list of components positioned in relative
coordinates — which windows either inherit from the wall or override one by one. Nothing is
hard-wired: a regular grid, an asymmetric mosaic and a full-screen PiP are the same code with
different templates.

**How many windows?** The question is put badly, and it is useful to know why. Forty windows have
already run on a single wall — but bare ones: no name strip, no meter, no history strip. The cost
lies in what SURROUNDS the picture, not in the number of pictures: every decoration has its own
price, and it is their sum that decides, together with the node carrying the wall. A number on its
own would mislead in both directions. So the wall publishes its frame budget and the breakdown of
its compositing time, to be judged on the measurements of YOUR installation.

---

## History strips: "what happened on this source?"

This is the least common tool on this wall, and the one that changes most how it is used. A
multiviewer shows the present instant; a history strip shows the **minute just gone**.

**Video strip** — one thumbnail per second, and under the band a ribbon of events: freeze, black,
signal loss. Sampling runs in its own thread, never in the mixing loop.

> ★ **The thumbnail is captured AT THE INSTANT of the event**, then pinned into its time slot:
> regular sampling no longer overwrites it. The band therefore shows the picture **it froze on**
> — and for a signal loss, the last valid picture — instead of any picture taken within that
> second. That detail is the difference between "there was an incident" and "here is what we were
> broadcasting when it happened."

**Audio strip** — peak envelope, persistent clipping in red, silence greyed out. A channel that
has been mute for forty seconds is seen at a glance, without having had to watch it.

Depth of your choosing: 10, 30, 60 or 120 seconds. Each strip exists in two forms, sharing the
same rendering code — as a component of a window template (the source being the window's), or as
a free block placed on the wall with its own source.

**What it costs the frame: almost nothing**, and that is a design choice. Recomposing a strip
costs tens of milliseconds; that work is done by a dedicated thread which publishes ready-made
tiles, and the mixing loop only pays for blending them in. A stage recomposing inside the frame
would drop the picture.

---

## The editor

The wall is drawn in the browser, over a preview of what the screen will show — not over an
abstract grid. You move and resize windows, place the components of a template, and lock whatever
must no longer move.

Snapping aligns on the edges and centres of neighbours, with visible guides. Window templates are
**aspect-free**: a template drawn for 16:9 applies to a 4:3 cell without its elements distorting
— and snapping itself never alters a window's proportions, which it did for a long time without
it showing.

Everything adjustable here is also drivable by **macro or trigger**: a parameter tree for what is
dialled, discrete actions for what is fired. A capability that can only be reached with a mouse
is a dead capability on show day.

---

## Worth knowing

**The cost of a wall is a memory matter, not a compute one.** It is bound not by processing power
but by moving bytes. The decisions that paid off concerned tile size and decoration reuse — not
the number of threads.

**Decoration is cached by signature.** Text, frames and lamps are only re-rendered when their
rendering ACTUALLY changes. Without that, a tally pushed ten times a second re-renders the
full-frame decoration ten times a second, and the wall drops a frame each time.

**Slice mode** (`slice_mode`, off by default) reads inputs in slices and publishes the output by
progressive commit instead of waiting for the whole picture. A stage working whole-frame adds one
frame of latency to every chain running through it — and that debt shows on no counter, since the
stage reports a perfect frame rate.

**The GPU path** (CUDA, with optional `gpu_slice`) changes only WHERE the bytes are, never the
protocol: same slice waits, same per-tile budgets, same progressive commit. `force_cpu` disables
it without the container holding the card — which was not a given: the auto-detection probe used
to open the device before the setting was even read.

---

## What it publishes

The container exposes its metrics on `:8080` — frame rate, per-stage breakdown of compositing
time, truncated frames, own latency, GPU state, decoration re-renders per second. They are not
ornamental: a placement choice shows its cost instead of hiding it.

Live control goes through the endpoints declared in `plugin.json`: windows, styles, overlays,
clocks, tally, texts.

---

## Installing it

**From Bobi.Studio** — the **Catalogue** page, which lists published components and installs them.
Or Settings → Plugins → *Import*, with a `.mxlplugin` package.

**By hand** — clone this repository into `plugins/multiview/` of an instance, then reload the
plugin registry.

> Changing `hooks.py` requires a registry reload: the orchestrator imports it once, at scan time.
> A hook that never fires is a perfectly silent failure.

---

## Reading it

- `script.py` — the whole plugin, a `str.format` template rendered by the orchestrator and run
  inside the container. **Every literal brace is doubled `{{ }}`**, comments included.
- `hooks.py` — the lifecycle hooks, which run in the orchestrator.
- `multiview.js` — the wall editor. `monitoring.html` — the supervision page.
- `plugin.json` — wiring, config schema, macro surface, control endpoints.
- `meta.json` — the version log, and where the *why* lives: each entry says what was broken, what
  was measured, and what the fix cost.
- `TISSU_SLICE_GPU.md` — the GPU slice work, with its bench verdicts (in French).

---

## Licence

GPL-3.0-or-later — see [LICENSE](LICENSE). Copyright © 2026 BOBI SAS, France.
