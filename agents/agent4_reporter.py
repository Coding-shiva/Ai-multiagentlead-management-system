# agents/agent4_reporter.py
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from db.database import get_lead_by_id, update_lead_status
from datetime import datetime
import logging
from typing import Dict
# 🟢 SMTP Imports
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os  # For accessing environment variables

logger = logging.getLogger(__name__)

# --- Configuration ---
try:
    from agents.gemini_service import client, MODEL_NAME
except ImportError:
    client = None
    MODEL_NAME = 'mock-model'

# --- Email Credentials ---
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")


SYSTEM_INSTRUCTION = """
You are an expert sales automation system. Your task is to generate a highly personalized, professional, and actionable follow-up email in HTML format. 
The email must strictly use the lead's name, reference the summary of the previous call, and emphasize the Next Step provided.
Output MUST be a single JSON object containing only the keys 'subject' and 'body_html'.
"""


class EmailOutput(BaseModel):
    """Defines the structured output format for Agent 4's email generation."""
    subject: str = Field(description="The email subject line.")
    body_html: str = Field(description="The full email body in clean HTML format (using <p> tags, <b>, <a>, etc.).")


# 🟢 NEW FUNCTION: Handles the physical email delivery
def send_email_via_smtp(recipient_email: str, subject: str, body_html: str) -> bool:
    """
    Attempts to send the email using Gmail's SMTP server (Port 465).
    """
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        logger.error("Email credentials missing in .env. Cannot send email.")
        return False

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email

        # Attach HTML body
        msg.attach(MIMEText(body_html, 'html'))

        # Connect to Gmail's secure SMTP server
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())

        logger.info(f"✅ Email SUCCESSFULLY SENT to {recipient_email}.")
        return True

    except Exception as e:
        logger.error(f"🔴 SMTP Email FAILED to send to {recipient_email}. Error: {e}")
        return False


# --- Main Reporting Function (Agent 4) ---
def generate_followup(lead_id: str) -> Dict:
    """
    Generates a personalized follow-up email, updates the status, AND sends the email.
    """
    logger.info(f"Agent 4: Starting report and follow-up generation for Lead {lead_id}...")
    lead = get_lead_by_id(lead_id)

    # 1. Prerequisite Check
    if not lead or lead.get('interaction', {}).get('call_status') != "Analyzed - Ready for Follow-up":
        return {"success": False, "error": "Lead not ready for Agent 4 (Analysis not complete)."}

    # Extract data safely
    analysis = lead.get('analysis', {})
    personal = lead.get('personal', {})
    enrollment = lead.get('enrollment', {})

    lead_name = personal.get('name', 'Valued Customer')
    course_interest = enrollment.get('course_interest', 'General Inquiry')
    next_steps_text = analysis.get('next_steps', 'schedule a follow-up call')
    recipient_email = personal.get('email')  # Get recipient email

    if not analysis.get('summary'):
        return {"success": False, "error": "Analysis summary is missing in the database."}

    # 1. Generate Email Content using Gemini
    USER_PROMPT = f"""
    Generate a follow-up email.
    Lead Name: {lead_name}
    Analysis Summary: {analysis.get('summary')}
    Next Step: {next_steps_text}
    Course: {course_interest}
    """

    # ... (Gemini call or mock logic to get email_result dictionary) ...
    if client is None or MODEL_NAME == 'mock-model':
        email_result = {
            "subject": f"Follow-up regarding your {course_interest} consultation (Mock)",
            "body_html": f"<html><body><p>Dear {lead_name},</p><p>Thank you for the call! As discussed, the next step is: <b>{next_steps_text}</b>.</p><p>Best regards, [Your Name/Team Name]</p></body></html>"
        }
    else:
        try:
            # Code to call Gemini API
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[USER_PROMPT],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=EmailOutput,
                    temperature=0.4
                )
            )
            email_result = response.parsed.model_dump()
        except Exception as e:
            logger.error(f"Gemini Email Generation Failed: {e}")
            return {"success": False, "error": f"Email generation failed: {e}"}

    # 2. Update MongoDB (Save report and set FINAL status)
    update_data = {
        "$set": {
            "interaction.call_status": "Follow-up Sent - Complete",
            "interaction.last_activity": datetime.now(),
            "analysis.report_subject": email_result['subject'],
            "analysis.report_body_html": email_result['body_html']
        }
    }

    db_result = update_lead_status(lead_id, update_data)

    if db_result.get('acknowledged'):
        logger.info(f"✅ Agent 4: Report saved to DB for {lead_id}.")

        # 3. CRITICAL: SEND EMAIL
        if recipient_email:
            send_status = send_email_via_smtp(recipient_email, email_result['subject'], email_result['body_html'])
            if not send_status:
                logger.error("Email sending failed (Check SMTP logs). Status updated in DB anyway.")

        return {"success": True, "report": email_result}
    else:
        logger.error(f"🔴 Agent 4: Failed to save final report to DB for {lead_id}.")
        return {"success": False, "error": "Failed to save final report to database."}