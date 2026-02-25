# agents/agent2_initiator.py

import os
import time
from typing import List, Dict, Tuple
from datetime import datetime
from db.database import update_lead_status
from agents.gemini_service import generate_dummy_transcript
from agents.bolna_service import start_outbound_call

# --- Configuration (Kept for context, but is also in .env) ---
LLM_SYSTEM_PROMPT = """
You are an AI enrollment counselor for the IERT.AI education system.
Your goal is to qualify the lead and book a demo.
Maintain a friendly, professional, and confident tone.
Start the conversation by referencing the lead's specific course interest.
Personalization Data: {personal_data}
"""

def generate_conversation_script(lead: Dict) -> str:
    """Generates the personalized LLM system prompt for the bolna.ai agent."""
    personal_data = {
        "name": lead.get('personal', {}).get('name', 'Customer'),
        "location": lead.get('personal', {}).get('location', 'Unknown City'),
        "course": lead.get('enrollment', {}).get('course_interest', 'General Inquiry')
    }
    return LLM_SYSTEM_PROMPT.format(personal_data=str(personal_data))


def initiate_call_campaign(leads: List[Dict]) -> Tuple[int, int]:
    """
    Agent 2's main function: Loops through leads and initiates calls.
    If the real-time call attempt fails (API error/failure response),
    it falls back to generating a mock transcript.
    """
    successful_calls = 0 # Successful initiation of call via Bolna API
    failed_calls = 0     # Failed initiation (triggers mock fallback)

    for lead in leads:
        lead_id = lead.get('lead_id')
        if not lead_id: continue

        phone_number = lead.get('personal', {}).get('phone_number')
        conversation_context = generate_conversation_script(lead)
        current_call_count = lead['interaction'].get('call_count', 0)

        # 1. Update status in DB to 'Calling' (Pre-call update)
        update_lead_status(lead_id, {
            "interaction.call_status": "Calling",
            "interaction.call_count": current_call_count + 1,
        })

        # --- REAL CALL PATH ATTEMPT ---
        try:
            bolna_result = start_outbound_call(phone_number, conversation_context, lead_id)

            if bolna_result.get('success'):
                successful_calls += 1
                # Status remains 'Calling' until the Webhook delivers the LIVE transcript.
                print(f"Agent 2: Real call initiated successfully for {lead_id}. Awaiting webhook.")
            else:
                # --- FALLBACK 1: Bolna API returned an explicit failure message ---
                failed_calls += 1
                print(f"Agent 2: Bolna API failed for {lead_id} ({bolna_result.get('error')}). Running Fallback...")

                # Generate and save dummy chat
                gemini_result = generate_dummy_transcript(
                    lead.get('personal', {}).get('name'),
                    lead.get('enrollment', {}).get('course_interest'),
                    lead.get('personal', {}).get('location')
                )

                update_lead_status(lead_id, {
                    "interaction.call_status": "Completed - Mock",
                    "analysis.transcript": gemini_result['transcript'],
                })
                print(f"Agent 2: Mock transcript generated for API failure: {lead_id}.")

        except Exception as e:
            # --- FALLBACK 2: API Connection Failure (Network, timeout, etc.) ---
            failed_calls += 1
            print(f"Agent 2: Critical API connection error for {lead_id} ({e}). Running Fallback...")

            # Generate and save dummy chat
            gemini_result = generate_dummy_transcript(
                lead.get('personal', {}).get('name'),
                lead.get('enrollment', {}).get('course_interest'),
                lead.get('personal', {}).get('location')
            )

            update_lead_status(lead_id, {
                "interaction.call_status": "Completed - Mock",
                "analysis.transcript": gemini_result['transcript'],
            })
            print(f"Agent 2: Mock transcript generated for connection error: {lead_id}.")

    return successful_calls,failed_calls
