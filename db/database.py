from dotenv import load_dotenv
load_dotenv()
from pymongo import MongoClient

import os
from typing import List, Dict, Optional # ⬅ ADDED Optional
from bson.objectid import ObjectId

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
