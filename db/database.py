import os
import json
import uuid
import gc  # Memory management ke liye
from datetime import datetime
from typing import List, Dict, Optional
from bson.objectid import ObjectId
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables (Localhost ke liye)
load_dotenv()

# Naya Path Logic: Local development ke liye credentials.json ka path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CREDS_PATH = os.path.join(BASE_DIR, "credentials.json")

# Ensure your .env has MONGO_URI set correctly
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "sales_leads"
COLLECTION_NAME = "leads"

# Initialize MongoDB Connection
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    leads_collection = db[COLLECTION_NAME]

    # Connection check
    client.admin.command('ismaster')
    print(f"✅ Successfully connected to MongoDB: {DB_NAME}")

except Exception as e:
    print(f"❌ ERROR: MongoDB connection failed: {e}")
    leads_collection = None


def get_leads_by_filter(filter_criteria: dict, limit: int = 10) -> List[Dict]:
    if leads_collection is None: return []
    leads = list(
        leads_collection.find(filter_criteria)
        .sort("score.current_score", -1)
        .limit(limit)
    )
    for lead in leads:
        lead['_id'] = str(lead['_id'])
    return leads


def get_lead_by_id(lead_id: str) -> Optional[Dict]:
    if leads_collection is None: return None
    lead = leads_collection.find_one({"lead_id": lead_id})
    if lead:
        lead['_id'] = str(lead['_id'])
        return lead
    return None


def update_lead_status(lead_id: str, updates: dict):
    if leads_collection is None: return {"acknowledged": False}
    try:
        if any(key.startswith('$') for key in updates):
            result = leads_collection.update_one({"lead_id": lead_id}, updates)
        else:
            result = leads_collection.update_one({"lead_id": lead_id}, {"$set": updates})
        return {"acknowledged": result.acknowledged, "modified_count": result.modified_count}
    except Exception as e:
        print(f"Update failed for {lead_id}: {e}")
        return {"acknowledged": False}


def check_lead_exists_in_db(email: str) -> bool:
    if leads_collection is None: return False
    return leads_collection.find_one({"personal.email": email}) is not None


def save_lead_to_db(lead_data: dict):
    if leads_collection is None: return None
    return leads_collection.insert_one(lead_data)


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
        "score": {"initial_score": 0, "current_score": 0, "priority_tag": "Cold"},
        "interaction": {"last_activity": datetime.now(), "call_status": "Pending", "call_count": 0},
        "analysis": {"summary": "", "sentiment": "neutral", "next_steps": "", "transcript": ""},
        "live_transcript": [],
        "report_body_html": "",
        "report_subject": ""
    }


def sync_google_sheets_to_db(sheet_name: str):
    """
    Syncs leads from Google Sheets to MongoDB.
    Supports Streamlit Cloud Secrets and Local .env/files.
    """
    # 1. HEAVY IMPORTS ANDAR (Memory Optimization)
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    import streamlit as st  # Zaroori for Cloud Secrets

    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    try:
        # 2. HYBRID CREDENTIALS LOGIC
        creds_json_str = None

        # Check Streamlit Cloud Secrets
        try:
            if "GOOGLE_CREDS_JSON" in st.secrets:
                creds_json_str = st.secrets["GOOGLE_CREDS_JSON"]
        except:
            pass  # Local par st.secrets error de sakta hai

        # Check Local Environment if Cloud Secrets not found
        if not creds_json_str:
            creds_json_str = os.getenv("GOOGLE_CREDS_JSON")

        # 3. AUTHORIZATION PROCESS
        if creds_json_str:
            # Parse String to Dict
            creds_info = json.loads(creds_json_str)

            # 🔥 CRITICAL FIX: Replace double backslash with actual newline
            if "private_key" in creds_info:
                creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")

            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
            print("💡 Using JSON credentials (Cloud or Env)")

        elif os.path.exists(LOCAL_CREDS_PATH):
            creds = ServiceAccountCredentials.from_json_keyfile_name(LOCAL_CREDS_PATH, scope)
            print(f"💡 Using local credentials file")
        else:
            print("❌ No credentials found!")
            return -1

        # 4. SHEET SYNC PROCESS
        client_g = gspread.authorize(creds)
        try:
            sheet = client_g.open(sheet_name).sheet1
        except Exception as sheet_err:
            print(f"❌ Sheet Error: {sheet_err}")
            return -1

        data = sheet.get_all_records()
        new_leads_added = 0

        for row in data:
            email = row.get('Email') or row.get('email')
            if email and not check_lead_exists_in_db(email):
                formatted_lead = map_sheet_row_to_lead(row)
                save_lead_to_db(formatted_lead)
                new_leads_added += 1

        # 5. GARBAGE COLLECTION (Release RAM)
        del data
        gc.collect()

        return new_leads_added

    except Exception as e:
        print(f"❌ Sync Error: {e}")
        gc.collect()
        return -1