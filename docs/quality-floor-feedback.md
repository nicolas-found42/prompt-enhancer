# Quality floor feedback calibration

Each completed run can record an accept or reject label. When the run has a
selected candidate, history links that label to the candidate ID and its six
dimension score vector. A rejected candidate also lists the dimensions scoring
below the vector's median. Runs without a selected candidate keep their
feedback visible with an `unavailable` linkage state; they are excluded from
calibration.

Floor recalibration is explicit. Call `POST /api/quality/floors/recalibrate`
to calculate and persist a new floor set. The endpoint returns the sample size,
keep/reject counts, per-dimension movement, the conservative bound, and active
floors. It also stores the last calibration record alongside the settings JSON.
No run or background task recalibrates floors automatically.

The calculation requires at least 12 linked run labels, including at least
three accepts and three rejections. For each score dimension, it sets the
proposed floor to the midpoint between the mean accepted score and mean
rejected score, then clamps the result to 75% of that dimension's shipped floor
and 1.0. Below the minimum sample, it records an unchanged result and movement
of zero. The shipped floors remain the starting reference on every explicit
recalibration, so evidence can move a floor in either direction while the
75% lower bound limits reductions. Twelve is an operational minimum, not a
claim of statistical certainty across future prompts; the application has no
domain-labeled history with which to establish a stronger sample threshold.

Persisted floors load into `Settings` at optimizer startup and are used when
scoring later candidates. Existing keep/reject records without a linked vector
remain in history and do not count toward the minimum sample.
