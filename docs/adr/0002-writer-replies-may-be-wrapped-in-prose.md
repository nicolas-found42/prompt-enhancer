---
status: accepted
supersedes: the sentence "Writer replies are read strictly" in ADR-0001
---

# Writer replies may be wrapped in prose

Some writer models, Llama 3.3 70B among them, wrap the JSON the engine asked for in a code fence, add a sentence of prose around it, or end it with a stray character such as an extra `}`, and a strict read turned each of those into a failed run. Callers now read a writer reply with `parse_reply_json`, which returns the first complete top-level object or array in it. The Gateway still returns raw text and parses nothing, so ADR-0001's decision stands; only its statement that writer replies are read strictly is replaced.

## Consequences

- A reply that is entirely JSON parses exactly as before, so recordings of such replies replay unchanged.
- A reply that stops partway through a value is still an error. The reader never returns a complete value nested inside an unfinished one.
- Each caller still checks the shape it needs and keeps its own failure handling (fail closed, fall back to asking the user, or report the failure).
- The weak-panel reader stays lenient in its own way; this does not change it.
