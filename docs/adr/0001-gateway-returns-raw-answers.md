# The Gateway returns raw answers

The Gateway gives callers raw Jev and writer answers. It does not parse them, and usage accounting stays in the HTTP adapter. We considered a typed Gateway (issue #31) that would return text and parsed decisions and own the decision log and usage. We rejected it for three reasons:

- **Unusable Jev answers are handled differently on purpose.** Diagnosis fails open, fidelity fails closed, grading scores only the affected item zero, and clarification falls back to asking the user. One parsing step at the Gateway would force one rule on all of them. Grading evidence and failure-prediction training also need the raw answers.
- **Weak-panel output is read leniently on purpose.** Writer replies are read strictly, and recorded replays depend on the lenient weak-panel reader, so a single text reader at the Gateway would change results.
- **Only the HTTP adapter can compute cost.** It needs the provider response, the route and catalog prices, and recordings do not store per-call usage. Moving usage accounting up would change the recording format.

The decision log is the only piece duplicated across adapters, and it is too small to justify a new module.
