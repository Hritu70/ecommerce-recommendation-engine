import pytest
from config import MODEL_DIR

pytest.importorskip("httpx")
if not (MODEL_DIR / "recommender.joblib").exists():
    pytest.skip("run src/train.py first to create the model artifact", allow_module_level=True)

from fastapi.testclient import TestClient
import api

client = TestClient(api.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_recommend_returns_k_ranked_items_with_explanations():
    body = client.get("/recommend/5", params={"k": 5}).json()
    assert body["strategy"] == "hybrid" and len(body["recommendations"]) == 5
    assert [x["rank"] for x in body["recommendations"]] == [1, 2, 3, 4, 5]
    assert all("explanation" in x for x in body["recommendations"])
    assert "limitations" in body


def test_recommendations_are_real_model_output_not_constant():
    a = [x["item_id"] for x in client.get("/recommend/5", params={"k": 10}).json()["recommendations"]]
    b = [x["item_id"] for x in client.get("/recommend/1200", params={"k": 10}).json()["recommendations"]]
    assert a != b, "different users should not all get the identical list"


def test_unknown_user_gets_popularity_fallback():
    body = client.get("/recommend/999999", params={"k": 3}).json()
    assert body["strategy"] == "popularity_fallback" and len(body["recommendations"]) == 3


@pytest.mark.parametrize("k", [0, 51, -1])
def test_invalid_k_is_rejected(k):
    assert client.get("/recommend/5", params={"k": k}).status_code == 422
