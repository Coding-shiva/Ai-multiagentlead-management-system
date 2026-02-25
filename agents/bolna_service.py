# agents/bolna_service.py
from dotenv import load_dotenv
load_dotenv()

import os
import requests

# Retrieve keys from environment variables set in .env
BOLNA_API_KEY = os.getenv("BOLNA_API_KEY")
BOLNA_AGENT_ID = os.getenv("BOLNA_AGENT_ID")

# NOTE: Replace with the actual production URL when deploying
BASE_URL = "https://api.bolna.ai"
OUTBOUND_URL = f"{BASE_URL}/call"

# --- CRITICAL FOR LIVE TRANSCRIPT CAPTURE ---
WEBHOOK_RECEIVER_URL=os.getenv("WEBHOOK_RECEIVER_URL")


def start_outbound_call(phone_number: str, conversation_context: str, lead_id: str):
    """
    Starts an outbound call to the given phone number, passing Webhook details.
    """
    if not BOLNA_API_KEY or not BOLNA_AGENT_ID:
        return {"success": False, "error": "Configuration Error: Bolna keys are not loaded."}

    headers = {
        "Authorization": f"Bearer {BOLNA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "agent_id": BOLNA_AGENT_ID,
        "recipient_phone_number": phone_number,
        # The 'user_data' field passes the lead_id for the webhook to use later.
        "user_data": {"lead_id": lead_id},
        "initial_context": conversation_context,
        "webhook_url": WEBHOOK_RECEIVER_URL, # The final destination for the transcript
        # 🟢 CRITICAL ADDITIONS FOR REAL-TIME STREAMING
        "transcript_mode": "real_time",
        "stream_events": ["transcription_update", "call_complete"]
    }

    try:
        response = requests.post(OUTBOUND_URL, headers=headers, json=payload, timeout=20)

        if response.status_code in (200, 201):
            return {"success": True, "response": response.json()}
        else:
            return {"success": False, "error": response.text}

    except Exception as e:
        return {"success": False, "error": str(e)}