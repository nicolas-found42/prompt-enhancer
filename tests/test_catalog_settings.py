from __future__ import annotations

from prompt_enhancer.catalog import JEV_MODEL, LiveModelCatalog, parse_catalog
from prompt_enhancer.gateway import GatewayConfig, HttpGateway


class CatalogTransport:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.requests = []

    def request(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return next(self.payloads)


def test_live_catalog_decodes_both_providers_and_never_exposes_keys():
    transport = CatalogTransport([
        {"data": [{"id": "go-model", "name": "Go", "pricing": {"prompt": "0.1", "completion": "0.2"}}]},
        {"models": [{"id": "or-model", "context_length": 4096}]},
    ])
    catalog = LiveModelCatalog(
        transport,
        go_url="https://go.test/models",
        openrouter_url="https://or.test/models",
        go_api_key="go-secret",
        openrouter_api_key="or-secret",
    )
    snapshot = catalog.fetch()
    assert [model.id for model in snapshot.go] == ["go-model"]
    assert [model.id for model in snapshot.openrouter] == ["or-model"]
    public = snapshot.to_public_dict()
    assert public["judge"]["id"] == JEV_MODEL
    assert "go-secret" not in repr(public)
    assert "or-secret" not in repr(public)


def test_unknown_model_routes_openrouter_even_without_catalog():
    gateway = HttpGateway(config=GatewayConfig(max_retries=0))
    assert gateway.route_model("unknown-model").provider == "openrouter"
    assert gateway.route_model(JEV_MODEL).provider == "openrouter"


def test_parser_accepts_bare_model_mapping():
    models = parse_catalog({"a": {"name": "A"}, "b": {"name": "B"}}, "go")
    assert {model.id for model in models} == {"a", "b"}
