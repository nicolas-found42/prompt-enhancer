# Gateway capture integrity

`RecordingGateway` retains a deep snapshot for each request, a unique correlation
identifier, its exact replay key, and its answer. Decisions must match the new
Gateway log entries in both count and order before they can be recorded. This
prevents a shared batch snapshot from being mistaken for per-request evidence.
Decision batches are validated before their first record is written.

Every save validates capture counts, unique identifiers, payload-derived hashes,
request order, and response associations. An invalid capture leaves the last
valid file intact. Historical replay bundles remain readable; they lack these
new integrity records and cannot serve as a verified capture audit.

For an independent audit, retain the ordered expected request keys separately
from the bundle and run:

```sh
uv run --locked python scripts/audit_gateway_capture.py recording.json --expected-keys expected-request-keys.json
```

The expected-keys file is a JSON string array from the actual request sequence,
including repeated requests. Use that independent record as the audit contract;
copying keys out of the capture being audited only checks internal consistency.
The command emits a complete receipt only after validation. Missing records,
reused identifiers, incorrect hashes, reordered requests, and wrong answer
associations exit nonzero and emit a failed receipt. This verifies capture
integrity, not the semantic correctness of the model's answers.
