from fastapi import FastAPI, HTTPException, APIRouter, Request
from pydantic import BaseModel
from typing import List, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime
import json
import logging  # Import logging for clear output

# Set up logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Agent Logic
from agents.agent1_data_fetcher import fetch_leads, interpret_simple_filter
from agents.agent2_initiator import initiate_call_campaign
from agents.agent3_analyzer import run_conversation_analysis
from agents.agent4_reporter import generate_followup
from agents.agent5_scorer import run_lead_scoring ,explain_lead_score # 🟢 AGENT 5 IMPORTED

# Import DB functions
from db.database import get_leads_by_filter, update_lead_status, get_lead_by_id ,register_manager,verify_password,managers_collection


# CRITICAL: Import the new service files created in the 'agents' directory
from agents import gemini_service, bolna_service


# Define the application lifespan context to handle graceful shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application lifespan STARTUP...")
    yield
    logger.info("Application lifespan SHUTDOWN... attempting client close (ignored if error occurs).")


# Initialize FastAPI app with the lifespan
app = FastAPI(
    title="AI Lead Manager Backend API",
    description="API for multi-agent system orchestration (Agent 1 & 2)",
    lifespan=lifespan
)

# --- CORS MIDDLEWARE CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Data Models ---
class LeadProfile(BaseModel):
    _id: Optional[str] = None
    lead_id: str
    personal: Dict
    enrollment: Dict
    score: Dict
    interaction: Dict
    analysis: Dict


class AgentRequest(BaseModel):
    manager_prompt: str
    max_leads: int = 10


class Agent2CallRequest(BaseModel):
    leads_to_call: List[LeadProfile]


# --- Webhook processing logic ---
@app.post("/webhooks/bolna_live_updates", status_code=200)
async def bolna_live_webhook_receiver(request: Request):
    """
    Receives real-time transcript chunks and the final call summary from the Bolna platform.
    """
    try:
        data = await request.json()
    except json.JSONDecodeError:
        logger.error("WEBHOOK ERROR: Invalid JSON received.")
        return {"status": "error", "message": "Invalid JSON"}

    # 1. Lead ID Extraction (Robust)
    context_details = data.get("context_details", {})
    recipient_data = context_details.get("recipient_data", {})
    lead_id = recipient_data.get("lead_id")

    if not lead_id:
        user_data = data.get("user_data", {})
        lead_id = user_data.get("lead_id")

    if not lead_id:
        logger.warning(f"WEBHOOK WARNING: Missing critical 'lead_id'. Ignoring payload.")
        return {"status": "ignored", "reason": "missing lead_id"}

    # 2. Determine Event/Status
    event = data.get("event")
    status_field = data.get("status", "")

    final_event = event if event else status_field

    if not final_event:
        logger.info(f"WEBHOOK INFO: Received payload with no event/status for {lead_id}. Ignoring.")
        return {"status": "ignored", "reason": "empty_event"}

    logger.info(f"🟢 Processing Event/Status for {lead_id}: {final_event}")

    # --- A. Handle Real-Time Chunks ---
    if final_event == "transcription_update":
        speaker = data.get("speaker", "unknown")
        text = data.get("transcript", "")
        timestamp = datetime.now().isoformat()

        if text.strip():
            # 🎯 PUSH LIVE CHUNKS
            result = update_lead_status(lead_id, {
                "$push": {"analysis.live_transcript": {
                    "speaker": speaker,
                    "text": text,
                    "timestamp": timestamp
                }},
                # 🎯 CRITICAL: Update call status and last_activity on successful chunk push
                "$set": {
                    "interaction.last_activity": datetime.now(),
                    "interaction.call_status": "In Progress - Live"
                }
            })
            if result.get("acknowledged"):
                logger.info(f"🎙️ Live Chunk PUSHED to DB for {lead_id}. [{speaker}] {text}")
            else:
                logger.error(f"🔴 DB WRITE FAILED for live chunk on {lead_id}.")
        else:
            logger.info(f"⚠️ Received empty text chunk for {lead_id}. Ignoring DB write.")

    # --- B. Handle Call Completion (Final Transcript & Status Update) ---
    elif final_event in ["call_complete", "call-disconnected", "completed", "failed"]:
        final_transcript = data.get("transcript", "")
        execution_id = data.get("id")

        # Determine final status for DB
        final_status_db = "Transcript Received" if final_event in ["call_complete", "completed"] else final_event

        # 🎯 Consolidated Update Operation
        updates = {
            "$set": {
                # Final Interaction Status
                "interaction.call_status": final_status_db,
                "interaction.last_activity": datetime.now(),

                # Final Analysis Data
                "analysis.transcript": final_transcript,
                "analysis.source": f"Bolna Call: {execution_id}",
                "analysis.summary": data.get("summary"),
                "analysis.sentiment": data.get("sentiment"),
                "analysis.next_steps": data.get("next_steps"),

                # Clear the live transcription array immediately after saving the final transcript
                "analysis.live_transcript": []
            }
        }

        # Execute the single consolidated update
        result = update_lead_status(lead_id, updates)

        if result.get("modified_count", 0) > 0:
            logger.info(f"✅ Final Transcript, Status, and Analysis saved for {lead_id}.")
        else:
            logger.error(f"🔴 FINAL DB WRITE FAILED or Lead {lead_id} not found. Check DB connection.")

    # --- C. Handle Intermediate Status Updates (Ringing, Initiated, Busy) ---
    elif final_event in ["initiated", "ringing", "in-progress", "busy"]:
        # 🎯 This ensures the last_activity and call_status update regardless of transcript data
        update_lead_status(lead_id, {
            "$set": {
                "interaction.last_activity": datetime.now(),
                "interaction.call_status": final_event
            }
        })
        logger.info(f"🔄 Updated call status to: {final_event}")

    return {"status": "ok"}


# --- Agent 1 Endpoint: Fetch Leads with Fallback (omitted for brevity) ---
@app.post("/api/v1/agent1/fetch_leads", response_model=List[LeadProfile])
async def trigger_agent1_fetch(request: AgentRequest):
    manager_prompt = request.manager_prompt
    max_leads = request.max_leads
    strict_leads = fetch_leads(manager_prompt, max_leads)
    if strict_leads:
        logger.info(f"Agent 1: Primary Query Success: Found {len(strict_leads)} high-priority leads.")
        return strict_leads
    fallback_filter = interpret_simple_filter(manager_prompt)
    logger.info(f"Agent 1: Primary Query Failed (0 results). Running Fallback Query: {fallback_filter}")
    fallback_leads = get_leads_by_filter(fallback_filter, limit=max_leads)
    if fallback_leads:
        logger.info(f"Agent 1: Fallback Query Success: Found {len(fallback_leads)} leads (score 0).")
        return fallback_leads
    else:
        raise HTTPException(status_code=404, detail="No leads found matching any criteria in database.")


# --- Agent 2 Endpoint: Initiate Voice Call Campaign (omitted for brevity) ---
@app.post("/api/v1/agent2/initiate_call", response_model=Dict)
async def trigger_agent2_call(request: Agent2CallRequest):
    if not request.leads_to_call:
        raise HTTPException(status_code=400, detail="No leads provided for the call campaign.")
    try:
        leads_list = [
            lead.model_dump(exclude={'_id'})
            for lead in request.leads_to_call
        ]
        successful, failed = initiate_call_campaign(leads_list)
        return {
            "status": "success",
            "message": "Call campaign initiated (Mock).",
            "total_leads": len(request.leads_to_call),
            "successful_calls": successful,
            "failed_calls": failed,
        }
    except Exception as e:
        logger.error(f"Error in Agent 2 API: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# 🟢 NEW AGENT 3 ENDPOINT: Trigger Conversation Analysis
@app.post("/api/v1/agent3/analyze_lead/{lead_id}", response_model=Dict)
async def trigger_agent3_analysis(lead_id: str):
    """
    Endpoint to trigger Agent 3: Analyzes a single lead's final transcript.
    """
    result = run_conversation_analysis(lead_id)

    if result.get("success"):
        return {"status": "success", "analysis": result["analysis"]}
    else:
        raise HTTPException(status_code=500, detail=result.get("error"))


# 🟢 NEW AGENT 4 ENDPOINT: Generate Follow-up and Finalize
@app.post("/api/v1/agent4/generate_followup/{lead_id}", response_model=Dict)
async def trigger_agent4_followup(lead_id: str):
    """
    Endpoint to trigger Agent 4: Generates personalized follow-up email and finalizes status.
    """
    result = generate_followup(lead_id)

    if result.get("success"):
        return {"status": "success", "report": result["report"]}
    else:
        raise HTTPException(status_code=500, detail=result.get("error"))


# 🟢 NEW AGENT 5 ENDPOINT: Run Lead Scoring
@app.post("/api/v1/agent5/run_scoring/{lead_id}", response_model=Dict)
async def trigger_agent5_scoring(lead_id: str):
    """
    Endpoint to trigger Agent 5: Calculates and updates the lead score and priority tag.
    """
    result = run_lead_scoring(lead_id)

    if result.get("success"):
        return {"status": "success", "score": result["score"], "tag": result["tag"]}
    else:
        raise HTTPException(status_code=500, detail=result.get("error"))


# 🟢 NEW AGENT 6 ENDPOINT: Explainable AI
@app.get("/api/v1/agent5/explain/{lead_id}")
async def explain_lead(lead_id: str):
    """
    Endpoint to trigger Agent 6 (SHAP XAI) for a single lead score.
    """
    result = explain_lead_score(lead_id)

    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])

# 🟢 NEW ENDPOINT: Fetch a single lead for real-time monitoring
@app.get("/api/v1/lead/{lead_id}", response_model=LeadProfile)
async def get_single_lead(lead_id: str):
    """
    Retrieves a single lead document by lead_id for real-time monitoring.
    """
    lead_data = get_lead_by_id(lead_id)
    if lead_data:
        return lead_data
    else:
        raise HTTPException(status_code=404, detail=f"Lead with ID {lead_id} not found.")

@app.post("/api/v1/register")
def signup(data: dict):
    success, msg = register_manager(data['username'], data['password'])
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": msg}

@app.post("/api/v1/login")
def login(data: dict):
    manager = managers_collection.find_one({"username": data['username']})
    if manager and verify_password(data['password'], manager['password']):
        return {"status": "success", "username": manager['username']}
    raise HTTPException(status_code=401, detail="Invalid credentials")


# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "AI Lead Manager API"}


# --- Uvicorn Execution Block (For direct running) ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)