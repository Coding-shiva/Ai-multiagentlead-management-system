from dotenv import load_dotenv
load_dotenv()
from pymongo import MongoClient
import json
import os
from typing import List, Dict, Optional # ⬅ ADDED Optional
from bson.objectid import ObjectId

from datetime import datetime
import uuid
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Naya Path Logic: Ye 'db' folder ka path nikalega
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CREDS_PATH = os.path.join(BASE_DIR, "credentials.json")

# Load environment variables

# Ensure your .env has MONGO_URI set correctly
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "sales_leads"
COLLECTION_NAME = "leads"

# Initialize MongoDB Connection
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    leads_collection = db[COLLECTION_NAME]

    # The ismaster command is cheap and does not require auth.
    client.admin.command('ismaster')
    print(f"Successfully connected to MongoDB database: {DB_NAME}")

except Exception as e:
    print(f"ERROR: Could not connect to MongoDB. Please check MONGO_URI in .env. Details: {e}")
    client = None
    db = None
    leads_collection = None


def get_leads_by_filter(filter_criteria: dict, limit: int = 10) -> List[Dict]:
    """
    Retrieves leads from MongoDB based on dynamic filter criteria.
    Called by Agent 1.
    """
    if leads_collection is None:
        return []

    # Sort by current_score descending, then limit
    leads = list(
        leads_collection.find(filter_criteria)
        .sort("score.current_score", -1)
        .limit(limit)
    )

    # Convert MongoDB's ObjectID to string for JSON/API transport
    for lead in leads:
        lead['_id'] = str(lead['_id'])

    return leads


# 🟢 NEW FUNCTION: Retrieve a single lead by its unique lead_id
def get_lead_by_id(lead_id: str) -> Optional[Dict]:
    """
    Retrieves a single lead document by its lead_id string.
    This is used by the FastAPI real-time monitor endpoint.
    """
    if leads_collection is None:
        return None

    lead = leads_collection.find_one({"lead_id": lead_id})

    if lead:
        # Convert MongoDB's ObjectID to string for JSON/API transport
        lead['_id'] = str(lead['_id'])
        return lead
    return None


def update_lead_status(lead_id: str, updates: dict):
    """
    Updates a lead's status and interaction fields, supporting both $set and $push operations.
    """
    if leads_collection is None:
        return {"acknowledged": False}

    try:
        # Check if the update uses a MongoDB operator like $set or $push
        if any(key.startswith('$') for key in updates):
            # If operators are present, use the updates dictionary directly
            result = leads_collection.update_one(
                {"lead_id": lead_id},
                updates
            )
        else:
            # Otherwise, wrap the updates in a $set operator for simple key updates
            result = leads_collection.update_one(
                {"lead_id": lead_id},
                {"$set": updates}
            )
        return {"acknowledged": result.acknowledged, "modified_count": result.modified_count}
    except Exception as e:
        print(f"Update failed for {lead_id}: {e}")
        return {"acknowledged":False}


# --- 1. Helper: Check if lead exists (by Email) ---
def check_lead_exists_in_db(email: str) -> bool:
    if leads_collection is None:
        return False
    return leads_collection.find_one({"personal.email": email}) is not None


# --- 2. Helper: Save formatted lead to DB ---
def save_lead_to_db(lead_data: dict):
    if leads_collection is None:
        return None
    return leads_collection.insert_one(lead_data)


# --- 3. Mapper: Sheet Row to your Schema ---
def map_sheet_row_to_lead(row):
    unique_id = f"LMS-{str(uuid.uuid4())[:4].upper()}"
    return {
        "lead_id": unique_id,
        "personal": {
            "name": row.get('Name', 'Unknown'),
            "phone_number": str(row.get('Phone', '')),
            "email": row.get('Email', ''),
            "location": row.get('Location', 'Unknown'),
            "source": "Google Sheets Sync"
        },
        "enrollment": {
            "course_interest": row.get('Course', 'N/A'),
            "grade_level": row.get('Grade', 'N/A'),
            "specific_query": row.get('Query', None)
        },
        "score": {
            "initial_score": 0,
            "current_score": 0,
            "priority_tag": "Cold"
        },
        "interaction": {
            "last_activity": datetime.now(),
            "call_status": "Pending",
            "call_count": 0
        },
        "analysis": {
            "summary": "",
            "sentiment": "neutral",
            "next_steps": "",
            "transcript": ""
        },
        "live_transcript": [],
        "report_body_html": "",
        "report_subject": ""
    }


def sync_google_sheets_to_db(sheet_name: str):
    """
    Syncs leads from Google Sheets to MongoDB.
    Supports credentials from 'GOOGLE_CREDS_JSON' env var or local file.
    """
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    try:
        # Step A: Load Credentials
        creds_json_str = os.getenv("GOOGLE_CREDS_JSON")

        if creds_json_str:
            # For Render/Production
            creds_info = json.loads(creds_json_str)
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
            print("💡 Using Google Credentials from Environment Variable")
        elif os.path.exists(LOCAL_CREDS_PATH):
            # For Local Development
            creds = ServiceAccountCredentials.from_json_keyfile_name(LOCAL_CREDS_PATH, scope)
            print(f"💡 Using Google Credentials from Local File: {LOCAL_CREDS_PATH}")
        else:
            print("❌ Error: No Google Credentials found (Env or File)!")
            return -1

        # Step B: Authorize and Open Sheet
        client_g = gspread.authorize(creds)
        try:
            sheet = client_g.open(sheet_name).sheet1
        except gspread.exceptions.SpreadsheetNotFound:
            print(f"❌ Error: Sheet '{sheet_name}' not found!")
            return -1

        # Step C: Process Data
        data = sheet.get_all_records()
        new_leads_added = 0

        for row in data:
            email = row.get('Email') or row.get('email')
            if email and not check_lead_exists_in_db(email):
                formatted_lead = map_sheet_row_to_lead(row)
                save_lead_to_db(formatted_lead)
                new_leads_added += 1

        return new_leads_added

    except Exception as e:
        print(f"❌ Google Sheets Sync Error: {e}")
        return -1