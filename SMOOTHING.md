# Pitch Edit Boundary Smoothing

When a note is pitch-shifted, the modified segment must be spliced back into the
original audio without audible clicks or abrupt pitch jumps at the cut points.
The approach has two stages: **f0 domain smoothing** (before WORLD synthesis)
and **audio domain cross-fade** (after synthesis).

---

## Why naive splicing fails

A hard cut at the note boundary leaves WORLD with a discontinuous f0 trajectory.
Even a short linear ramp is not enough because:

- A linear ramp has non-zero slope right up to its endpoints, creating a
  perceivable "kink" where flat regions meet the ramp.
- Pitch perception is logarithmic. Linear interpolation in Hz sounds uneven;
  the same number of Hz sounds like a large change at low pitches and a small
  one at high pitches.
- At 16–22 kHz sample rates, even a 20 ms ramp spans only a handful of WORLD
  frames, leaving little room to absorb the transition.

---

## F0 domain smoothing (3 layers)

### Layer 1 — Raised cosine (Hann) blend weight

A weight function `w[i] ∈ [0, 1]` is built for every WORLD frame in the
padded segment:

```
w[i] = 0                        outside the transition zones
w[i] = 0.5 × (1 − cos(π t))    rising edge   (note start ± ramp)
w[i] = 1                        inside the note
w[i] = 0.5 × (1 + cos(π t))    falling edge  (note end ± ramp)
```

The raised cosine has **zero derivative at both endpoints**, so the weight
joins the flat regions (0 and 1) tangentially. This eliminates the kink that
a linear ramp produces.

Ramp duration: **80 ms** (was 20 ms in earlier versions).

### Layer 2 — Gaussian smoothing of the weight function

The weight array is convolved with a Gaussian kernel (σ = 40 % of the ramp
duration in frames):

```python
weight = gaussian_filter1d(weight, sigma=ramp_frames * 0.5)
```

This smears the transition zone outward, so the slope of the f0 curve changes
gradually over a wide region rather than having any single frame where the
curvature is large. WORLD's spectral estimation window (≈ 3 × pitch period)
then sees a smoothly evolving f0 throughout.

### Layer 3 — Blending in log-f0 space

The modified f0 is computed as:

```
f0_modified[i] = f0_original[i] × ratio^w[i]
```

which in log space is:

```
log f0_modified[i] = log f0_original[i] + w[i] × log(ratio)
```

This is a **linear interpolation in semitones**, not in Hz.  Because semitones
are the unit of pitch perception, the glide sounds uniform to the ear — the
same number of semitones per unit time throughout the transition, regardless of
absolute frequency.

A second, narrower Gaussian pass (σ = 2 frames) is applied to log-f0 in the
transition region (where `0.02 < w < 0.98`), blended with the unsmoothed
values by a factor that peaks at `w = 0.5`:

```
blend[i] = clip(4 × w[i] × (1 − w[i]), 0, 1)
log_f0[i] = (1 − blend[i]) × log_f0[i] + blend[i] × gaussian_smoothed[i]
```

This removes any residual micro-discontinuities caused by voiced/unvoiced
frame transitions near the boundary.

---

## Audio domain cross-fade (after synthesis)

After WORLD synthesises the modified segment, the splice is made with a
**raised-cosine cross-fade** (not linear):

```
t  = 0.5 × (1 − cos(π × linspace(0, 1, cf_samples)))   # fade-in
t′ = 0.5 × (1 + cos(π × linspace(0, 1, cf_samples)))   # fade-out

result[fade_zone] = original × (1 − t) + synthesised × t
```

Cross-fade length: `ramp_sec / 2` = **40 ms**, capped by the available padding.

This eliminates phase discontinuities at the exact cut points that even perfect
f0 smoothing cannot address, because the original and resynthesised audio have
independent harmonic phases.

---

## Parameters

| Parameter | Value | Purpose |
|---|---|---|
| `pad_sec` | 120 ms | Analysis context given to WORLD on each side |
| `ramp_sec` | 80 ms | Width of the f0 transition zone |
| Weight Gaussian σ | 40 ms (≈ `ramp * 0.5`) | Spread of the slope change |
| Log-f0 Gaussian σ | 2 WORLD frames (10 ms) | Micro-discontinuity removal |
| Cross-fade length | 40 ms | Audio-domain phase blending |
| WORLD frame period | 5 ms | ~200 frames/s |

---

## Data flow summary

```
original audio
      │
      ▼
[extract padded segment]  ← 120 ms padding on each side
      │
      ▼
[WORLD analysis]  →  f0, spectral envelope (sp), aperiodicity (ap)
      │
      ▼
[build raised-cosine weight]
      │
[Gaussian-smooth weight]          Layer 1 + 2
      │
[blend f0 in log space]           Layer 3
      │
[micro-smooth log-f0 at boundary] Layer 3b
      │
      ▼
[WORLD synthesis with modified f0]
      │
      ▼
[raised-cosine cross-fade splice into original audio]
      │
      ▼
modified audio
```
