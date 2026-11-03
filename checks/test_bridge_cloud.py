"""Training on a rented pod (fizgig.bridge.cloud): which offers it tries, and in what order."""
from fizgig.bridge import cloud


def test_offers_are_filtered_by_card_and_price_and_ranked_by_value(monkeypatch):
    monkeypatch.setattr(cloud.pods, "offers", lambda key: {"gpus": [
        {"gpu": "RTX A6000", "id": "a6000", "vram": 48, "speed": 0.45, "secure": {"price": 0.59, "stock": "Low"}},
        {"gpu": "RTX 5090", "id": "5090", "vram": 32, "speed": 1.0, "secure": {"price": 1.19, "stock": "Low"},
         "community": {"price": 0.99, "stock": "High"}},
        {"gpu": "RTX 4090", "id": "4090", "vram": 24, "speed": 0.75, "secure": {"price": 0.59, "stock": "High"}},
        {"gpu": "H100 SXM", "id": "h100", "vram": 80, "speed": 1.5, "secure": {"price": 2.99, "stock": "High"}}]})
    got = [(o["id"], o["cloud"]) for o in cloud.candidates("k", 2.0, 32)]
    # the 4090 is too small, the H100 over the price; the rest go cheapest per run first (price over speed)
    assert got == [("5090", "COMMUNITY"), ("5090", "SECURE"), ("a6000", "SECURE")]
