"""The domain logic behind the tools, tested without MCP."""

import pytest

from realestate_mcp import agreements, feasibility, helpdesk, rera


def statuses(agreement_id: str) -> dict[str, str]:
    return {r["clause"]: r["status"] for r in agreements.scan(agreement_id)["clauses"]}


def test_scan_finds_the_planted_deviations():
    a = statuses("A-101")
    assert a["possession"] == a["maintenance"] == "values differ"
    b = statuses("B-204")
    assert b["cancellation_by_allottee"] == "values differ"
    assert b["car_parking"] == "missing"
    assert b["dispute_resolution"] == "wording differs"  # Kolkata -> Mumbai: no number changes
    c = statuses("C-307")
    assert c["payment_schedule"] == "values differ"
    assert c["delay_compensation"] == "wording differs"


def test_scan_leaves_unchanged_clauses_alone():
    b = statuses("B-204")
    assert b["governing_law"] == "matches template"
    assert b["parties"] == b["property"] == "filled-in fields only"
    assert agreements.scan("B-204")["flagged"] == ["cancellation_by_allottee", "car_parking", "dispute_resolution"]


def test_scan_reports_values_on_each_side():
    row = next(r for r in agreements.scan("B-204")["clauses"] if r["clause"] == "cancellation_by_allottee")
    assert row["only_in_template"] == ["10%", "45 days"]
    assert row["only_in_agreement"] == ["20%", "90 days"]


def test_same_wording_allows_filled_placeholders_only():
    template = "Apartment No. {{unit_no}} on floor {{floor}}."
    assert agreements.same_wording(template, "Apartment No.  B-204 on floor 2.")
    assert not agreements.same_wording(template, "Apartment No. B-204 on floor 2, with a terrace.")


def test_review_clause_and_bad_input():
    assert agreements.review_clause("B-204", "car_parking")["missing"] is True
    with pytest.raises(ValueError):
        agreements.review_clause("B-204", "no_such_clause")
    with pytest.raises(ValueError):
        agreements.scan("template")
    with pytest.raises(ValueError):
        agreements.get("Z-999")


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("refund if I cancel", "cancellation_refund-1"),
        ("who do I complain to", "escalation_contacts-4"),
        ("can I transfer my booking", "cancellation_refund-5"),
        ("can I keep a dog", "maintenance_society-6"),
    ],
)
def test_project_search_ranks_the_right_passage_first(query, expected):
    assert helpdesk.search(query)[0]["id"] == expected


def test_project_search_maps_everyday_words():
    ids = [r["id"] for r in helpdesk.search("possession is late")]
    assert "possession_handover-2" in ids  # "late" is expanded to the documents' word "delay"


def test_rera_lookup():
    assert rera.read_rera_section("Section 18")["title"] == "Return of amount and compensation"
    assert "error" in rera.read_rera_section("999")
    assert rera.search_rera("compensation for delay in possession")[0]["section"] == "18"


def test_feasibility_numbers_match_week_3():
    parcel = feasibility.load_parcels()["parcel-b"]
    f = feasibility.run_financials(parcel, 5500, None, 10_000_000)
    assert (f["margin_pct"], f["breakeven_price_per_sqft_carpet"]) == (13.7, 4745)
