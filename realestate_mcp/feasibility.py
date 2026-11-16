"""Land feasibility arithmetic, reused unchanged from the Week 3 Land Feasibility Panel.

Finance model, comparable statistics, site and title rubrics, RERA registration check, weighted
score and legal veto. Plain Python: a model that calls these tools quotes the numbers, it never
computes them.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
SQFT_PER_ACRE = 43_560
SQM_PER_SQFT = 0.09290304
CRORE = 10_000_000

APPROVALS_PCT_OF_CONSTRUCTION = 0.08
MARKETING_PCT_OF_REVENUE = 0.03
FINANCE_PCT_OF_LAND_AND_CONSTRUCTION = 0.10

ROLES = ("legal", "finance", "market", "risk")
WEIGHTS = {"legal": 0.30, "finance": 0.30, "market": 0.20, "risk": 0.20}
MAX_ADJUST = 2
GO_AT = 7.0
CONDITIONAL_AT = 5.0

RERA_THRESHOLD_SQM = 500
RERA_THRESHOLD_APARTMENTS = 8


# ------------------------------------------------------------------ data
def load_parcels(folder: Path = DATA_DIR / "parcels") -> dict[str, dict]:
    parcels = {}
    for path in sorted(folder.glob("*.json")):
        p = json.loads(path.read_text(encoding="utf-8"))
        p["land"]["area_sqft"] = round(p["land"]["area_acres"] * SQFT_PER_ACRE)
        parcels[p["id"]] = p
    return parcels


def crore(value: float) -> str:
    return f"₹{value / CRORE:,.2f} cr"


# ------------------------------------------------------------------ finance
def run_financials(
    parcel: dict,
    sale_price: float | None = None,
    construction_cost: float | None = None,
    extra_costs: float = 0,
) -> dict:
    """Project economics. Optional overrides let agents test "what if" prices and costs."""
    plan, land = parcel["plan"], parcel["land"]
    price = float(sale_price or plan["sale_price_per_sqft_carpet"])
    cost = float(construction_cost or plan["construction_cost_per_sqft_builtup"])
    built_up = land["area_sqft"] * plan["far"]
    carpet = built_up * plan["carpet_efficiency"]
    revenue = carpet * price
    land_cost = land["asking_price_inr"]
    construction = built_up * cost
    approvals = APPROVALS_PCT_OF_CONSTRUCTION * construction
    marketing = MARKETING_PCT_OF_REVENUE * revenue
    finance = FINANCE_PCT_OF_LAND_AND_CONSTRUCTION * (land_cost + construction)
    extra = float(extra_costs or 0)
    total_cost = land_cost + construction + approvals + marketing + finance + extra
    profit = revenue - total_cost
    years = plan["timeline_years"]
    roi = profit / total_cost
    result = {
        "sale_price_per_sqft_carpet": round(price),
        "construction_cost_per_sqft_builtup": round(cost),
        "built_up_sqft": round(built_up),
        "carpet_sqft": round(carpet),
        "apartments_estimate": round(carpet / plan["avg_unit_carpet_sqft"]),
        "revenue_inr": round(revenue),
        "cost_breakdown_inr": {
            "land": round(land_cost),
            "construction": round(construction),
            "approvals_and_fees": round(approvals),
            "marketing": round(marketing),
            "finance": round(finance),
            "extra_what_if": round(extra),
        },
        "total_cost_inr": round(total_cost),
        "profit_inr": round(profit),
        "margin_pct": round(100 * profit / revenue, 1),
        "roi_on_cost_pct": round(100 * roi, 1),
        "annualised_roi_pct": round(100 * ((1 + roi) ** (1 / years) - 1), 1) if roi > -1 else -100.0,
        "breakeven_price_per_sqft_carpet": round(total_cost / carpet),
        "timeline_years": years,
        "assumptions": {
            "approvals_pct_of_construction": APPROVALS_PCT_OF_CONSTRUCTION,
            "marketing_pct_of_revenue": MARKETING_PCT_OF_REVENUE,
            "finance_pct_of_land_and_construction": FINANCE_PCT_OF_LAND_AND_CONSTRUCTION,
        },
    }
    return result


def sensitivity(parcel: dict, sale_price: float | None = None, construction_cost: float | None = None) -> dict:
    base = run_financials(parcel, sale_price, construction_cost)
    price, cost = base["sale_price_per_sqft_carpet"], base["construction_cost_per_sqft_builtup"]
    return {
        "base_margin_pct": base["margin_pct"],
        "price_minus_10_pct": run_financials(parcel, price * 0.9, cost)["margin_pct"],
        "cost_plus_10_pct": run_financials(parcel, price, cost * 1.1)["margin_pct"],
        "both": run_financials(parcel, price * 0.9, cost * 1.1)["margin_pct"],
    }


def finance_score(margin_pct: float, margin_at_minus_10: float) -> int:
    bands = [(25, 9), (20, 8), (15, 7), (10, 5), (5, 4), (0, 3)]
    score = next((s for floor, s in bands if margin_pct >= floor), 1)
    if margin_at_minus_10 < 5:
        score -= 1
    return max(1, min(10, score))


# ------------------------------------------------------------------ market
def comps_stats(parcel: dict, planned_price: float | None = None) -> dict:
    prices = [c["price_per_sqft_carpet"] for c in parcel["market"]["comparables"]]
    planned = float(planned_price or parcel["plan"]["sale_price_per_sqft_carpet"])
    median = statistics.median(prices)
    return {
        "comparables": len(prices),
        "median_price_per_sqft_carpet": round(median),
        "min": min(prices),
        "max": max(prices),
        "planned_price_per_sqft_carpet": round(planned),
        "planned_vs_median_pct": round(100 * (planned - median) / median, 1),
    }


def market_score(gap_pct: float) -> int:
    if gap_pct <= 3:
        return 8
    if gap_pct <= 8:
        return 6
    if gap_pct <= 15:
        return 4
    return 3


# ------------------------------------------------------------------ site
def site_risk(parcel: dict) -> dict:
    s = parcel["site"]
    deductions = []

    def deduct(points: float, why: str) -> None:
        deductions.append({"points": points, "reason": why})

    depth = s["waterlogging_max_cm_last_5_monsoons"]
    if depth > 60:
        deduct(4, f"waterlogging up to {depth} cm")
    elif depth > 30:
        deduct(2.5, f"waterlogging up to {depth} cm")
    elif depth > 15:
        deduct(1, f"waterlogging up to {depth} cm")
    if s["years_waterlogged_of_5"] >= 3:
        deduct(1.5, f"waterlogged in {s['years_waterlogged_of_5']} of the last 5 monsoons")
    if s["road_width_m"] < 6:
        deduct(2, f"approach road only {s['road_width_m']} m wide")
    elif s["road_width_m"] < 9:
        deduct(1, f"approach road {s['road_width_m']} m wide")
    if s["distance_to_highway_km"] > 5:
        deduct(1, f"{s['distance_to_highway_km']} km from the highway")
    if s["soil"] == "weak":
        deduct(1.5, "weak soil")
    if s["pending_approvals"]:
        deduct(min(2, 0.5 * len(s["pending_approvals"])), f"{len(s['pending_approvals'])} approvals pending")
    if s["ht_line_over_plot"]:
        deduct(1, "high-tension line over the plot")
    score = 10 - sum(d["points"] for d in deductions)
    return {"score": max(1, min(10, round(score))), "deductions": deductions, "facts": s}


# ------------------------------------------------------------------ legal
def title_checklist(parcel: dict) -> dict:
    t = parcel["title"]
    red_flags, curable = [], []
    red_flags += [f"litigation: {x}" for x in t["litigation"]]
    red_flags += [f"gap in title chain: {x}" for x in t["chain_gaps"]]
    if t["owners_consenting"] < t["owners"]:
        red_flags.append(f"only {t['owners_consenting']} of {t['owners']} owners have consented")
    if t["chain_years_verified"] < 30:
        curable.append(f"title chain verified for {t['chain_years_verified']} years (30 expected)")
    if not t["mutation_done"]:
        curable.append("mutation (record-of-rights update in the seller's name) not done")
    if t["conversion_status"] != "converted":
        curable.append(f"land-use conversion {t['conversion_status']}")
    curable += [f"encumbrance: {x}" for x in t["encumbrances"]]
    score = 10 - 4 * len(red_flags) - 1.5 * len(curable)
    verdict = "red" if red_flags else ("amber" if curable else "green")
    return {
        "verdict": verdict,
        "score": max(1, min(10, round(score))),
        "red_flags": red_flags,
        "curable_issues": curable,
    }


def rera_check(parcel: dict) -> dict:
    """Section 3(2)(a): registration is not required only if land <= 500 sq m and apartments <= 8."""
    area_sqm = parcel["land"]["area_sqft"] * SQM_PER_SQFT
    apartments = run_financials(parcel)["apartments_estimate"]
    required = area_sqm > RERA_THRESHOLD_SQM or apartments > RERA_THRESHOLD_APARTMENTS
    return {
        "registration_required": required,
        "land_area_sqm": round(area_sqm),
        "apartments_estimate": apartments,
        "rule": "RERA Act 2016, Section 3(2)(a): exempt only if land <= 500 sq m and apartments <= 8, all phases",
    }


# ------------------------------------------------------------------ scoring
def rubric_scores(parcel: dict, sale_price: float | None = None, construction_cost: float | None = None) -> dict:
    fin = run_financials(parcel, sale_price, construction_cost)
    sens = sensitivity(parcel, sale_price, construction_cost)
    comps = comps_stats(parcel, fin["sale_price_per_sqft_carpet"])
    return {
        "legal": title_checklist(parcel)["score"],
        "finance": finance_score(fin["margin_pct"], sens["price_minus_10_pct"]),
        "market": market_score(comps["planned_vs_median_pct"]),
        "risk": site_risk(parcel)["score"],
    }


def clamp_score(proposed, rubric: int) -> int:
    """Specialists may move the rubric score by at most MAX_ADJUST points."""
    try:
        value = round(float(proposed))
    except (TypeError, ValueError):
        return rubric
    return max(1, min(10, max(rubric - MAX_ADJUST, min(rubric + MAX_ADJUST, value))))


def weighted_score(scores: dict[str, int]) -> float:
    return round(sum(WEIGHTS[r] * scores[r] for r in ROLES), 2)


def recommendation(scores: dict[str, int], legal_verdict: str) -> dict:
    total = weighted_score(scores)
    label = "Go" if total >= GO_AT else ("Conditional go" if total >= CONDITIONAL_AT else "No-go")
    rule = f"weighted score {total} (Go >= {GO_AT}, Conditional go >= {CONDITIONAL_AT})"
    if legal_verdict == "red" and scores["legal"] <= 3:
        label, rule = "No-go", rule + "; legal red flag with legal score <= 3 forces No-go"
    elif legal_verdict == "red" and label == "Go":
        label, rule = "Conditional go", rule + "; legal red flag caps the result at Conditional go"
    return {"label": label, "weighted_score": total, "rule": rule}


def tensions(parcel: dict, sale_price: float | None = None) -> list[dict]:
    """Conflicts between specialists that code can detect, handed to the chair as starting points."""
    fin = run_financials(parcel, sale_price)
    comps = comps_stats(parcel, fin["sale_price_per_sqft_carpet"])
    site = parcel["site"]
    found = []
    if comps["planned_vs_median_pct"] > 5:
        at_median = run_financials(parcel, comps["median_price_per_sqft_carpet"])
        found.append(
            {
                "between": ["finance", "market"],
                "issue": f"Finance assumes ₹{fin['sale_price_per_sqft_carpet']:,}/sq ft but the comparable median is "
                f"₹{comps['median_price_per_sqft_carpet']:,} ({comps['planned_vs_median_pct']:+}%). "
                f"At the median the margin falls from {fin['margin_pct']}% to {at_median['margin_pct']}%.",
            }
        )
    if site["waterlogging_max_cm_last_5_monsoons"] > 30:
        found.append(
            {
                "between": ["risk", "finance"],
                "issue": f"Waterlogging up to {site['waterlogging_max_cm_last_5_monsoons']} cm, but the cost model has "
                "no line for raising the plinth or drainage works.",
            }
        )
    if parcel["title"]["conversion_status"] != "converted":
        found.append(
            {
                "between": ["legal", "finance"],
                "issue": f"Land-use conversion is {parcel['title']['conversion_status']}, but the timeline of "
                f"{parcel['plan']['timeline_years']} years assumes approvals start immediately.",
            }
        )
    return found
