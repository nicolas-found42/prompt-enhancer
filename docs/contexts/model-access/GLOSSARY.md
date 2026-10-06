# Model Access

This context gives the engine one way to reach the models it uses while improving a prompt.

## Language

**Gateway**:
The single way the engine reaches any model: the writer, the weak panel, the strong check, and Jev. Live, scripted, recorded, and replayed runs differ only in which gateway they use.
_Avoid_: client, provider, model API

## Runtime bounds

Each Gateway operation has a wall-clock deadline. It includes model-catalog
routing, provider requests, retries, and retry delays. The default is 180
seconds and `PROMPT_ENHANCER_OPERATION_TIMEOUT` sets another value in seconds.
`PROMPT_ENHANCER_TIMEOUT` sets the per-request socket timeout. A
`time_limit_s` run control remains a separate pause at a completed Round
boundary.

The Gateway caller returns a timeout or cancellation result by the operation
deadline even if a transport does not stop promptly. It limits unfinished
transport workers to eight per Gateway and asks adapters with request-scoped
abort support to close their request. With the built-in HTTP transport, this
can close a response body after its response handle exists; the standard
library does not expose a handle while waiting for response headers, so that
worker can remain until the connection ends or the server process stops.
