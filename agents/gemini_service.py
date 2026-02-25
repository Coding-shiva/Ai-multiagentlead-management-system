# agents/gemini_service.py
from dotenv import load_dotenv

load_dotenv()

import os
import json
from google import genai
from google.genai.errors import APIError

# --- Configuration ---
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

# --- Global Client Initialization ---
global client

try:
    # Correct initialization using the Client constructor and the loaded API key
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Import types for use in function config
    import google.genai.types

    MODEL_NAME = 'gemini-2.5-flash'
    print("DEBUG: Gemini Client initialized successfully.")

except Exception as e:
    # This executes if the API key fails validation, forcing the hardcoded fallback
    print(f"WARNING: Gemini client failed to initialize. Using mock transcript fallback. Error: {e}")
    client = None
    import google.genai.types

    MODEL_NAME = "mock-model"

SYSTEM_PROMPT = """
You are a sales agent generating a plausible 3-turn sales dialogue for testing purposes. 
The conversation must include the AI Agent referencing the lead's location and course interest. 
Format the output STRICTLY with timestamps and speaker tags: [00:00:XX Speaker: ROLE] Message. Do not include any markdown or extra text.
"""


def generate_dummy_transcript(lead_name: str, course: str, location: str) -> dict:
    """
    Uses Gemini-Flash to generate a plausible mock transcript.
    If Gemini fails or is not initialized, returns a hardcoded mock.
    """

    USER_PROMPT = f"Create a short sales dialogue with Lead: {lead_name}, who inquired about {course} in {location}."

    # --- Hardcoded Fallback ---
    if client is None:
        return {
            "success": True,
            "transcript": f"""
[00:00:01 Speaker: AI Agent] Hello {lead_name}, I'm Alex from IERT.AI. I see your interest in {course} in {location}. Is that correct?
[00:00:07 Speaker: Customer] Yes, but I need to know about the payment plan.
[00:00:12 Speaker: AI Agent] I understand. I will ensure a human counselor calls you back to discuss the installment options. (Call ends).
""",
        }

    # --- Gemini API Call ---
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[USER_PROMPT],
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7
            )
        )
        transcript_text = response.text.strip()
        if transcript_text.startswith("```"):
            transcript_text = transcript_text.split('\n', 1)[-1].strip().strip('```')

        return {"success": True, "transcript": transcript_text}

    except Exception as e:
        return {"success": False, "error": str(e)}


def close_gemini_client():
    """Placeholder for safe client cleanup."""
    global client
    if client:
        pass