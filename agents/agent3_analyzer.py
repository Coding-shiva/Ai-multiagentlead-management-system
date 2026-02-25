# agents/agent3_analyzer.py
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from db.database import get_lead_by_id, update_lead_status
from datetime import datetime
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# --- Configuration ---
try:
    from agents.gemini_service import client, MODEL_NAME
except ImportError:
    client = None
    MODEL_NAME = 'mock-model'
    logger.error("Gemini client not available for Agent 3. Using mock analysis.")


# --- 1. Define Structured Output Schema using Pydantic ---
class AnalysisOutput(BaseModel):
    """Defines the required output structure for Gemini."""
    summary: str = Field(
        description="A concise, 3-4 sentence summary of the entire call, focusing on lead motivation and key topics discussed."
    )
    sentiment: str = Field(
        description="Overall sentiment of the lead: positive, neutral, or negative."
    )
    next_steps: str = Field(
        description="The immediate, actionable next step for the sales team (e.g., Schedule Demo, Send Pricing, Re-qualify)."
    )


SYSTEM_INSTRUCTION = """
You are a highly efficient conversation analysis expert for an education consulting firm. Your task is to analyze the provided sales call transcript and extract key insights in a structured JSON format.
Strictly adhere to the provided JSON schema for output. Do not add any conversational text, headers, or markdown outside of the JSON structure.
"""


# --- 2. Main Analysis Function (Agent 3) ---
def run_conversation_analysis(lead_id: str) -> Dict:
    """Fetches the final transcript, runs Gemini analysis, and updates the database."""
    logger.info(f"Agent 3: Starting analysis for Lead {lead_id}...")
    lead = get_lead_by_id(lead_id)

    if not lead or lead.get('analysis', {}).get('transcript') is None:
        return {"success": False, "error": "Transcript not found or call is incomplete."}

    transcript = lead['analysis']['transcript']

    if client is None or MODEL_NAME == 'mock-model':
        # --- Mock Fallback Logic ---
        logger.warning("Agent 3: Using mock analysis due to missing Gemini client.")
        analysis_result = {
            "summary": "Mock: The lead was interested but needed a specific installment plan.",
            "sentiment": "neutral",
            "next_steps": "Send detailed installment plan via email."
        }
    else:
        # 3. Gemini API Call for structured output
        USER_PROMPT = f"Analyze the following call transcript to generate the required summary, sentiment, and next steps:\n\nTRANSCRIPT:\n---\n{transcript}\n---"

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[USER_PROMPT],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=AnalysisOutput,
                    temperature=0.0
                )
            )
            analysis_result = response.parsed.model_dump()

        except Exception as e:
            logger.error(f"Gemini API Analysis Failed for {lead_id}: {e}")
            return {"success": False, "error": str(e)}

    # 4. Update MongoDB with Analysis Results
    analysis_updates = {
        "$set": {
            "analysis.summary": analysis_result.get('summary'),
            "analysis.sentiment": analysis_result.get('sentiment'),
            "analysis.next_steps": analysis_result.get('next_steps'),
            "interaction.last_activity": datetime.now(),
            "interaction.call_status": "Analyzed - Ready for Follow-up"
        }
    }

    db_result = update_lead_status(lead_id, analysis_updates)

    if db_result.get('acknowledged'):
        logger.info(f"✅ Agent 3: Analysis complete and saved for {lead_id}.")
        return {"success": True, "analysis": analysis_result}
    else:
        logger.error(f"🔴 Agent 3: Failed to save analysis to DB for {lead_id}.")
        return {"success": False, "error": "Failed to save analysis to database."}