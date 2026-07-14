# PaperBanana figure spec — `tapt_overview`

Prompt / layout brief for the figure generator. Written as a layout brief, not prose.

---

## Overall

A single-row, three-panel architecture diagram for a CVPR/NeurIPS-style paper.
Full text width, roughly 3:1 aspect. Clean vector look, thin strokes, generous
whitespace. Left-to-right dataflow with arrows crossing panel boundaries.

**Color legend (consistent, shown once, bottom-left):**
- **Grey** = frozen (SigLIP patch embed, SigLIP transformer blocks, entropy tokenizer)
- **Orange** = trainable (Conv^i merge, ZeroMLP, run-length embedding ψ)
- **Blue** = FRESH token (re-embedded this frame)
- **Green** = REDUNDANT (carried forward, no token emitted)
- **Magenta** = MISSED REUSE (unchanged, but unreusable — the money shot, see Panel B)

---

## Panel A (left) — "① Spatial: entropy quadtree (APT)"

- Three video frames stacked with slight perspective offset, labelled `t−1`, `t`, `t+1`.
- Overlay on frame `t` a **quadtree partition grid**: a mostly-flat region (sky/wall)
  merged into a few big `4p` cells, a mid-detail region into `2p` cells, and a cluttered
  region left as many small `p` cells. Make the size disparity obvious.
- Small inset heatmap labelled *per-patch entropy* feeding into the grid, with a threshold
  symbol `τ`.
- Caption strip: **"Each frame partitioned independently. Strict partition: every base cell
  covered exactly once."**

## Panel B (centre — the core contribution, give it the most space)

Two inputs converge, drawn as arrows entering from the left:

1. **From Panel A**: the partitions of frame `t−1` and frame `t`, drawn **side by side and
   visibly disagreeing** — a region `t−1` split into four `p` cells is a single `2p` cell at
   `t`. This disagreement is the whole reason the panel exists; do not draw them alike.
2. **From the raw pixels**: a **dirty mask** — a binary `G×G` grid, filled squares where
   |x_t − x_ref| > τ_rlt. Label the arrow **"RLT dirty test — anchored to the frame the token
   is actually reused from, not t−1"**.

Centre of the panel: a **2×2 decision table**. Rows = *shape matches t−1?*, columns =
*did anything inside change?*

|                    | nothing changed        | something changed |
|--------------------|------------------------|-------------------|
| **shape matches**  | 🟢 **REDUNDANT** — reuse | 🔵 FRESH          |
| **shape differs**  | 🟣 **MISSED REUSE** → FRESH | 🔵 FRESH      |

Make the diagonal read clearly: **reuse needs BOTH conditions**; anything else is FRESH.

Beside the table, three small cell diagrams:

- **REDUNDANT** (green): an arrow curving back to frame `t−1`'s token, run-length counter `ℓ`
  incrementing. Annotate: *no token emitted — this is the entire saving.*
- **FRESH** (blue): the cell's pixels → frozen `E(·)` → the orange merge (`Conv^i` → `ZeroMLP`),
  **plus** a parallel frozen `E(Resize_p)` anchor branch, summed. Annotate: *APT's Eq. 2,
  unchanged. Zero-init merge ⇒ at step 0 this is exactly E(Resize_p) + π.*
- **MISSED REUSE** (magenta) — **draw this one biggest, it is the finding**: show frame `t`'s
  `2p` cell sitting over frame `t−1`'s four `p` cells, with the dirty mask **entirely empty**
  (nothing changed). Then a **red ✗ on the carry-forward arrow**, annotated: *nothing changed —
  but t−1 has no token of this shape to carry. Pays full price anyway.* Sub-caption:
  **"Ceiling on savings. A partition-stability problem, not a classification one."**

## Panel C (right) — "③ Encode surviving events"

- The token events (blue + magenta, **visibly fewer** than dense) packed into a 1-D sequence,
  with a ghosted "dense would be T×P" bar behind for contrast.
- Each token gets `+ ψ(ℓ)` — small orange box.
- The packed sequence enters the **frozen SigLIP blocks** (grey). Draw attention as
  block-diagonal: **queries = this frame's events only**, but **keys/values = this frame's full
  partition** — carried green tokens joining the KV set with a **dashed border** (not
  recomputed). Annotate: *events attend over the full partition — no frame is starved of
  context. Sound because every op but attention is position-wise. Reduces exactly to per-frame
  APT when nothing is reused.*
- Output: **scatter-back** to a dense `(T, P, C)` grid via the carry index — one coarse token
  fanning out to fill the base cells it covers; one redundant cell pulling from an earlier
  frame's token.
- Then a small, de-emphasized grey chain: `→ DTS/SAE (4× temporal) → projector → LLM`.
  Annotate: **"Video-XL-Pro pipeline unchanged."**

---

## Text on the figure (keep minimal)

- Panel titles: **① Spatial: entropy quadtree (APT)** / **② Temporal: reuse iff shape matches
  AND nothing changed (ours)** / **③ Encode surviving events**
- One equation, small, under Panel B's FRESH diagram:
  `h = ZeroMLP(Conv^i(children)) + E(Resize_p) + π_s + ψ(ℓ)`
- Bottom-right badge: **"zero-init + frozen anchor ⇒ TAPT is exactly APT at step 0"**

## Things to avoid

- **Don't draw frame `t−1` and frame `t` with the same partition.** They must disagree, or
  Panels B and C have no reason to exist.
- **Don't drop the magenta case.** It is the honest limitation and the strongest part of the
  story; a reviewer will ask about it and the figure should answer first.
- Don't draw the merge as an average/pool — it is a learned conv, and that distinction is a
  stated contribution.
- Don't show cross-frame attention. Attention is strictly *within* a frame; the only things
  crossing frames are the **carried k/v**, which must be drawn dashed/ghosted, clearly
  distinct from the solid attention arrows.
- Don't imply FRESH is a special or expensive path — it is the *fallback*, i.e. plain APT.
  The savings come from REDUNDANT alone.
