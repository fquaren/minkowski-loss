# Why a Minkowski loss on a deterministic model produces a fixed texture

Write-up for the paper appendix / thesis (researcher's request, 2026-10-26). Evidence and scripts: DECISIONS §37
(hourly bracket) and §38 (striping measure, texture orientation). Status labels: *theorem* (derivable),
*measured* (run in this project, with the file), *inference* (our reading, not tested directly).

## 1. The observation

On the hourly corrected target (v4), backbones trained with the Minkowski structural loss show a texture that the
targets do not have:

- strokes and stripes inside large rain areas, more pronounced as the weight grows (6e-5 < 1.8e-4 < 5.4e-4 <
  1.8e-3), visible in `notes/data_analysis_v4/bracket_gallery.pdf`;
- the strokes have one dominant direction across patches, whatever the orientation of the rain in the target
  (researcher's reading of the gallery: S-shaped strokes running the same way in the centre of rainy patches);
- at the highest weight the texture becomes grid-aligned striping and rectangular blocks.

## 2. What the loss can and cannot see

The loss compares, at each threshold u of a fixed grid, three numbers of the excursion set {f >= u}: the area A,
the perimeter P and the Euler characteristic χ (number of pieces minus number of holes), on a log scale,
integrated over the thresholds (`src/losses/minkowski.py`).

- *Theorem.* In the continuum these are invariant under rigid motions: V_k(gK) = V_k(K) for every translation
  and rotation g. By Hadwiger's characterization theorem, in 2-D every additive, continuous, motion-invariant
  functional of sets is a linear combination of exactly these three [CITE?: Hadwiger's characterization theorem,
  e.g. Hadwiger 1957 or Klain & Rota 1997]. The loss is therefore blind, by construction, to where a structure
  lies and to which way it points; it constrains the *statistics* of the excursion sets, not their *arrangement*.
- Consequence: in the continuum the loss has a flat direction along every rotation and translation of the
  texture. A field and its 90°-rotated copy score the same.
- This is a stronger form of the established finding that "the functionals carry no positional information, so
  no variant improves FSS" (CLAUDE.md, established findings).

## 3. The causal chain

1. **MSE sets where texture may go.** The sub-grid structure of the target (which way a band runs inside a
   25 km input pixel, where a cell sits) is not determined by the 12.5x coarser input. Under MSE every specific
   placement is penalised twice (where it was drawn but is not, and where it is but was not drawn)
   [CITE?: double-penalty problem in precipitation verification], so MSE alone prefers no texture: the
   conditional mean. *Inference, consistent with the images:* the cheapest compromise with the Minkowski term,
   which asks for the right perimeter and number of pieces at each threshold, is a low-amplitude modulation
   inside large rain areas. Where the mean is high, a small wiggle crosses several thresholds at once and adds
   perimeter and pieces at all of them for a small squared-error cost. Hence strokes in the centre of rainy patches.
2. **The grid estimators set which way it points.** On the pixel grid the estimators break the rotation symmetry
   (*measured*, `scripts/evaluate/` and DECISIONS §38):
   - perimeter, 4-direction Cauchy-Crofton estimator (weights 0.473 axial, 0.218 diagonal) [CITE?: Cauchy-Crofton
     formula / digital perimeter estimation]: a 150 x 20 px ellipse is under-counted by 4.8% when its long axis
     lies on a grid axis and over-counted by about 1.5% at 25-35° and 55-65°, so oblique edges earn about 6.6%
     more perimeter per unit length than axis-aligned ones;
   - Euler characteristic, 4-connected cubical estimator (χ = V - E_x - E_y + F): a 1-px-wide line has χ = 1 along
     an axis but up to 212 at 45°, because it falls apart into corner-touching pixels; from 1.5 px width the
     effect vanishes.
   The network adds preferred directions of its own (convolutions, the coarse input cells and their upsampling;
   the vanilla backbone already shows faint edges along the coarse cells).
3. **A flat direction plus a weak symmetry-breaking field selects one direction.** Because the texture carries no
   information about the input, the cheapest solution is one learned stroke ("stamp") applied wherever the mean is
   high. *Measured* (`scripts/evaluate/texture_orientation.py`, 64 heavy validation patches):
   - predicting on the input rotated by 90° and rotating back gives a texture that correlates with the original
     at only 0.04 (5.4e-4) and 0.11 (1.8e-3): the texture is not tied to the input;
   - the targets' texture directions are nearly uniform (peak share 0.062 per 10°, uniform 0.056); at 1.8e-3 the
     texture edges concentrate at one direction with 3x the uniform share, and after the rotation test the peak
     moves by exactly 90°, i.e. it stays the same in the network's own frame: the texture is locked to the grid;
   - at lower weights the peaks are weaker (0.073-0.083); single test, not repeated.
4. **The weight scales the effect.** The larger the weight, the more perimeter and pieces the loss pays for, the
   stronger the stamp. The high-pass striping measure rises monotonically with the weight (median axis excess
   0.017 vanilla, 0.038, 0.048, 0.049, 0.098 at 1.8e-3), and so does the MAE cost (+4.7% to +19%).

A physics analogy: spontaneous symmetry breaking with a crystal field. The energy of a ferromagnet does not
depend on its direction, but the lattice makes some directions slightly easier and the system settles into one.
The analogy breaks in that nothing orders collectively here: one set of network weights chooses one texture, and
the easy directions come from the discretisation of the loss, not from physics.

## 4. Why the model cannot draw the target's shapes, and what follows

The target's specific shapes are aleatoric given the coarse input: many fine fields are compatible with one input.
A deterministic model must commit to one field, receives no credit from the Minkowski loss for correct placement
and is penalised by MSE for incorrect placement. Matching the excursion-set statistics with a fixed stamp is
therefore the optimum of this objective, not a training failure. It is a concrete instance of the
perception-distortion trade-off [CITE?: Blau & Michaeli, The Perception-Distortion Tradeoff, CVPR 2018].

A generative model removes the dilemma: each flow-matching sample can place a different, plausible structure while
the ensemble mean stays near the conditional mean. This is the argument for coupling the structural loss to the
generative stage (Study 2), and the reason the FM experiments come next (DECISIONS §38).

A common misreading to name in the text: "the Minkowski loss teaches the model the shape of rain". It constrains
three numbers per threshold. A field of horizontal bands and the same field rotated by 90° score identically; so
do a cell on the left and the same cell on the right.

## 5. Where this account stops

- The curved (S-shaped) form of the stroke is not measured; only its direction is. Training applies random
  horizontal and vertical flips, which map one diagonal onto the other, so a consistent diagonal preference
  cannot come from the data.
- The orientation evidence at low weights is weak (64 patches, one test).
- The chain's first link (texture placed where the mean is high because it is cheapest under MSE) is an inference.

## 6. Checks and possible remedies (EXPERIMENTS §5, To explore)

- *Show the stamp:* average the high-pass texture over many patches; a fixed stamp survives averaging, real texture
  cancels.
- *Less anisotropic estimators:* more Crofton directions for the perimeter (adding the 2:1 "knight's move" lines
  reduces the orientation error to about ±1%), and an Euler characteristic mixing 4- and 8-connectivity to remove the
  bonus of thin diagonal lines. Prediction: the orientation peak weakens. One ~20 h run at a high weight tests it.
- *For the shapes themselves:* stochasticity (flow matching), or a structural term that sees position (windowed or
  FSS-like losses).
