from db.database import get_leads_by_filter
from typing import List, Dict


# --- Helper Function for Simple/Fallback Filter (Location-Only) ---
def interpret_simple_filter(request_text: str) -> Dict:
    """

    """
    request_text = request_text.lower()

    # --- 1. OVERRIDE: FETCH ALL LEADS (For debugging/admin) ---
    if "fetch all leads" in request_text:
        print("Agent 1: OVERRIDE ACTIVATED. Fetching ALL documents from DB.")
        return {}  # Empty filter fetches ALL documents

    # --- 2. CHECK FOR ANALYSIS PROMPT (For displaying mock chats) ---
    # This filter is kept for displaying processed leads in Streamlit.
    if "fetch leads completed mock" in request_text:
        filter_criteria = {"analysis.transcript": {"$ne": None}}
        return filter_criteria

    # --- 3. STANDARD LOCATION FILTER (Default) ---
    # The filter starts empty and is populated only by location.
    filter_criteria = {}

    # Location Logic
    if "mumbai" in request_text:
        filter_criteria["personal.location"] = "Mumbai"
    elif "gurugram" in request_text:
        filter_criteria["personal.location"] = "Gurugram"
    elif "new delhi" in request_text or "delhi" in request_text:
        filter_criteria["personal.location"] = "New Delhi"
    elif "up" in request_text or "uttar pradesh" in request_text:
        filter_criteria["personal.location"] = "UP"

    # If no location is found, filter_criteria remains {}, fetching ALL leads (useful for a master list).
    return filter_criteria


# --- MAIN AGENT 1 FUNCTION (Primary Query Logic) ---
def fetch_leads(manager_request: str, max_leads: int = 10) -> List[Dict]:
    """
    Agent 1's main function: Interprets request and fetches data from DB.
    """
    print(f"Agent 1 running query for: '{manager_request}'")

    # Start with the location-only filter
    query_filter = interpret_simple_filter(manager_request)

    # Apply the STRICTION (Score Filter) if the prompt includes the keyword
    if ("high-scoring" in manager_request.lower() or "hot" in manager_request.lower()):
        # This acts as the Primary (strict) query check.
        if query_filter == {}:
            # If no location was specified, and they ask for high score, enforce a dummy filter
            query_filter["score.current_score"] = {"$gte": 50}
        else:
            # If location WAS specified, combine it with the score filter
            query_filter["score.current_score"] = {"$gte": 50}

    # Retrieve the leads
    leads_profiles = get_leads_by_filter(query_filter, limit=max_leads)

    return leads_profiles
