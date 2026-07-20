# Pareto-convexity test suite (mathematically guaranteed)

Drop-in companions to `convex_pf.csv / concave_pf.csv / linear_pf.csv`.
Same format: header `f1,f2,f3,w1,w2,w3`, weights on the regular simplex grid
(step 1/30, 496 points). Judge results with `classify_pareto_convexity_lp`
(exact). Load 3D files with `obj_cols=(0,1,2)`; the 2D file with `obj_cols=(0,1)`.

## Construction and the logical chain  ("because A, B")

All 3D fronts are p-power fronts realised by `f_i = w_i^(1/p)`, so that
`sum_i f_i^p = sum_i w_i = 1` exactly.

  Because the front satisfies `sum f_i^p = 1`  (A)
  -> the attainable set `{ y>=0 : sum y_i^p >= 1 }` is the superlevel set of
     `g(y)=sum y_i^p`, which is **concave for p<1** and **convex for p>1**;
  -> for p<1 the superlevel set is convex  => the front is CONVEX (toward the
     ideal): no point is dominated by a convex combination of the others,
     so the LP margin t* > 0 at every interior point;
  -> for p>1 the superlevel set is non-convex => the front is CONCAVE: every
     interior point is dominated by a convex combination, so t* < 0.   (B)

| file                       |  p  | guaranteed class | expected LP output            |
|----------------------------|-----|------------------|-------------------------------|
| convex_strong_p0.30.csv    | 0.3 | convex           | 100% convex                   |
| convex_mild_p0.80.csv      | 0.8 | convex           | 100% convex                   |
| concave_mild_p1.30.csv     | 1.3 | concave          | 99.4% concave (3 vertices conv)|
| concave_strong_p3.00.csv   | 3.0 | concave          | 99.4% concave (3 vertices conv)|

Together with the originals (p=0.5 convex, p=1 linear, p=2 sphere/concave) the
signed curvature score is **monotone and changes sign exactly at p=1** — see
`validation_figure.png` panel (a). NB: under the LP test a *linear* front
collapses to `concave` (t*≈0 < eps); this is expected, not an error.

## Edge cases

- **disconnected_convex_3d.csv** — the p=1/2 convex surface (`f_i=w_i^2`,
  `sum sqrt(f_i)=1`) with a band `0.35<w1<0.58` removed → **two** components,
  gap of width ≈0.25 in f1. Class is convex everywhere; LP returns 100% convex,
  testing robustness to disconnection (cf. *Disconnected_Pareto_Front*).

- **mixed_2d.csv** — convex arc (f1≤1, bulging toward the ideal) stitched to a
  concave arc (f1≥1, bulging away); per-point ground truth is exact. This is the
  discriminating case and it exposes two real, distinct notions:
    * *local curvature* (cross-product / SVD): recovers each arc's sign at ~99%
      **but** the 2D method's global orientation flip mislabels the whole front
      when the concave part dominates the centroid — i.e. the heuristic must be
      made local for heterogeneous fronts;
    * *global supportedness* (LP, ~80%): only convex-hull vertices read `convex`;
      locally-convex points that are dominated across the concave gap read
      `concave` — exactly the points weighted-sum scalarization cannot reach.
