import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base, get_db
from main import app

TEST_DATABASE_URL = "sqlite:///./test_consulting.db"

test_engine = create_engine(
    TEST_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=test_engine
)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_and_teardown_db():
    """Reset database tables before each test so tests are completely isolated."""
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


def test_create_practice_area():
    response = client.post(
        "/practice-areas",
        json={"name": "Data Engineering", "floor_number": 3},
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["name"] == "Data Engineering"
    assert data["floor_number"] == 3


def test_create_consultant_under_existing_parent():
    parent = client.post(
        "/practice-areas",
        json={"name": "Data Engineering", "floor_number": 3},
    ).json()

    response = client.post(
        f"/practice-areas/{parent['id']}/consultants",
        json={
            "name": "Alex Morgan",
            "designation": "Senior",
            "consultation_fee": 800,
            "experience_years": 8,
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["name"] == "Alex Morgan"
    assert data["designation"] == "Senior"
    assert data["consultation_fee"] == 800
    assert data["experience_years"] == 8


def test_create_consultant_under_missing_parent():
    response = client.post(
        "/practice-areas/999/consultants",
        json={
            "name": "Alex Morgan",
            "designation": "Senior",
            "consultation_fee": 800,
            "experience_years": 8,
        },
    )
    assert response.status_code == 404


def test_get_parent_with_nested_consultants_filter_and_sort():
    parent = client.post(
        "/practice-areas",
        json={"name": "Cloud Practice", "floor_number": 2},
    ).json()
    pid = parent["id"]

    client.post(
        f"/practice-areas/{pid}/consultants",
        json={
            "name": "Zara Khan",
            "designation": "Senior",
            "consultation_fee": 900,
            "experience_years": 6,
        },
    )
    client.post(
        f"/practice-areas/{pid}/consultants",
        json={
            "name": "Ben Carter",
            "designation": "Senior",
            "consultation_fee": 600,
            "experience_years": 4,
        },
    )
    client.post(
        f"/practice-areas/{pid}/consultants",
        json={
            "name": "Aarav Mehta",
            "designation": "Junior",
            "consultation_fee": 400,
            "experience_years": 1,
        },
    )

    # 1. Verify nested children returned
    res_all = client.get(f"/practice-areas/{pid}")
    assert res_all.status_code == 200
    assert len(res_all.json()["consultants"]) == 3

    # 2. Filter by exact designation ("Senior")
    res_filtered = client.get(
        f"/practice-areas/{pid}", params={"designation": "Senior"}
    )
    assert res_filtered.status_code == 200
    filtered_list = res_filtered.json()["consultants"]
    assert len(filtered_list) == 2
    assert all(c["designation"] == "Senior" for c in filtered_list)

    # 3. Blank designation filter treated as no filter
    res_blank = client.get(
        f"/practice-areas/{pid}", params={"designation": "   "}
    )
    assert res_blank.status_code == 200
    assert len(res_blank.json()["consultants"]) == 3

    # 4. Sort by name ascending
    res_sort_name = client.get(
        f"/practice-areas/{pid}", params={"sort": "name"}
    )
    assert res_sort_name.status_code == 200
    names = [c["name"] for c in res_sort_name.json()["consultants"]]
    assert names == ["Aarav Mehta", "Ben Carter", "Zara Khan"]

    # 5. Sort by consultation_fee ascending
    res_sort_fee = client.get(
        f"/practice-areas/{pid}", params={"sort": "consultation_fee"}
    )
    assert res_sort_fee.status_code == 200
    fees = [c["consultation_fee"] for c in res_sort_fee.json()["consultants"]]
    assert fees == [400, 600, 900]

    # 6. Invalid sort value returns HTTP 404
    res_bad_sort = client.get(
        f"/practice-areas/{pid}", params={"sort": "experience_years"}
    )
    assert res_bad_sort.status_code == 404


def test_update_consultant_success_and_ownership_boundaries():
    p1 = client.post(
        "/practice-areas", json={"name": "Practice 1", "floor_number": 1}
    ).json()
    p2 = client.post(
        "/practice-areas", json={"name": "Practice 2", "floor_number": 2}
    ).json()

    c1 = client.post(
        f"/practice-areas/{p1['id']}/consultants",
        json={
            "name": "Alex Morgan",
            "designation": "Junior",
            "consultation_fee": 500,
            "experience_years": 2,
        },
    ).json()

    updated_payload = {
        "name": "Alex Morgan Updated",
        "designation": "Principal",
        "consultation_fee": 1200,
        "experience_years": 9,
    }

    # Valid update under owning parent
    res_ok = client.put(
        f"/practice-areas/{p1['id']}/consultants/{c1['id']}",
        json=updated_payload,
    )
    assert res_ok.status_code == 200
    assert res_ok.json()["name"] == "Alex Morgan Updated"
    assert res_ok.json()["designation"] == "Principal"
    assert res_ok.json()["consultation_fee"] == 1200
    assert res_ok.json()["experience_years"] == 9

    # Update with missing parent -> 404
    res_missing_parent = client.put(
        f"/practice-areas/999/consultants/{c1['id']}",
        json=updated_payload,
    )
    assert res_missing_parent.status_code == 404

    # Update with missing child -> 404
    res_missing_child = client.put(
        f"/practice-areas/{p1['id']}/consultants/999",
        json=updated_payload,
    )
    assert res_missing_child.status_code == 404

    # Update child belonging to another parent (c1 belongs to p1, requested via p2) -> 404
    res_wrong_parent = client.put(
        f"/practice-areas/{p2['id']}/consultants/{c1['id']}",
        json=updated_payload,
    )
    assert res_wrong_parent.status_code == 404


def test_validation_errors_return_422():
    # Invalid floor_number (< 1)
    assert (
        client.post(
            "/practice-areas", json={"name": "Valid", "floor_number": 0}
        ).status_code
        == 422
    )
    # Empty practice area name
    assert (
        client.post(
            "/practice-areas", json={"name": "", "floor_number": 1}
        ).status_code
        == 422
    )

    parent = client.post(
        "/practice-areas", json={"name": "Cybersecurity", "floor_number": 2}
    ).json()
    pid = parent["id"]

    # Empty consultant name
    assert (
        client.post(
            f"/practice-areas/{pid}/consultants",
            json={
                "name": "",
                "designation": "Senior",
                "consultation_fee": 500,
                "experience_years": 3,
            },
        ).status_code
        == 422
    )
    # Fee below 300
    assert (
        client.post(
            f"/practice-areas/{pid}/consultants",
            json={
                "name": "Sam",
                "designation": "Senior",
                "consultation_fee": 250,
                "experience_years": 3,
            },
        ).status_code
        == 422
    )
    # Negative experience
    assert (
        client.post(
            f"/practice-areas/{pid}/consultants",
            json={
                "name": "Sam",
                "designation": "Senior",
                "consultation_fee": 500,
                "experience_years": -1,
            },
        ).status_code
        == 422
    )