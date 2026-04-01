from db.database import get_leads_by_filter
from typing import List, Dict


def interpret_simple_filter(request_text: str) -> Dict:
    request_text = request_text.lower()
    filter_criteria = {}

    # --- 1. NEW LEADS (Pending) ---
    if "new leads" in request_text or "not called" in request_text:
        filter_criteria["interaction.call_status"] = "Pending"

    # --- 2. INTERACTION COMPLETED (Baat ho chuki hai) ---
    elif "interaction" in request_text or "baat hui" in request_text or "called leads" in request_text:
        # Jinka status Pending nahi hai aur jinpe transcript maujood hai
        filter_criteria["interaction.call_status"] = {"$ne": "Pending"}
        filter_criteria["analysis.transcript"] = {"$ne": None}

    # --- 3. MOCK/ANALYSIS ONLY ---
    elif "fetch leads completed mock" in request_text:
        return {"analysis.transcript": {"$ne": None}}

    # --- 4. LOCATION FILTER ---
    if "mumbai" in request_text:
        filter_criteria["personal.location"] = "Mumbai"
    elif "gurugram" in request_text:
        filter_criteria["personal.location"] = "Gurugram"
    elif "delhi" in request_text:
        filter_criteria["personal.location"] = "New Delhi"

    return filter_criteria


def fetch_leads(manager_request: str, max_leads: int = 10) -> List[Dict]:
    request_text = manager_request.lower()
    query_filter = interpret_simple_filter(manager_request)

    # --- 🟢 HIGH PROFILE LOGIC ---
    if "high-scoring" in request_text or "hot" in request_text or "high profile" in request_text:
        query_filter["score.current_score"] = {"$gte": 50}

        # Conflict fix: Agar high profile hai toh 'Pending' status ignore karein
        if "interaction.call_status" in query_filter and query_filter["interaction.call_status"] == "Pending":
            del query_filter["interaction.call_status"]

    print(f"🔍 Agent 1 executing filter: {query_filter}")
    leads_profiles = get_leads_by_filter(query_filter, limit=max_leads)
    return leads_profiles