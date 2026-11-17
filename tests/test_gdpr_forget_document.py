from fastapi.testclient import TestClient

from neuroflow.api import app


def test_gdpr_tenant_erasure():
    client = TestClient(app)

    # Ingest docs for tenant_eu
    client.post("/v1/ingest", json=[
        {"doc_id": "gdpr-doc", "tenant_id": "tenant_eu", "text": "Confidential customer financial records and account details."}
    ])

    # Query to populate cache and index
    res = client.post("/v1/query", json={"query": "financial records", "tenant_id": "tenant_eu"})
    assert res.status_code == 200
    assert len(res.json()["results"]) > 0

    # Execute GDPR Right to Erasure
    del_res = client.delete("/v1/tenants/tenant_eu/forget")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "erased"

    # Query again -> Must be 0 results
    after_res = client.post("/v1/query", json={"query": "financial records", "tenant_id": "tenant_eu"})
    assert len(after_res.json()["results"]) == 0
