# agents/agent4_reporter.py
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from db.database import get_lead_by_id, update_lead_status
from datetime import datetime
import logging
from typing import Dict
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os

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

# 🟢 1. NEW: Video Link Mapping Dictionary
# Agent 3 se jo 'course_interest' aayega, uske basis par link select hoga
COURSE_VIDEO_LINKS = {
    "MBA": "https://drive.google.com/file/d/1RD1aueZf_TWU_QwARvlzmlMS3gx2tYvU/view?usp=sharing",
    "Computer Science": "https://drive.google.com/file/d/14LMv5mKseOqg4HvJ9z6D17QROSZ-c_JE/view?usp=sharing"
}

SYSTEM_INSTRUCTION = """
You are an expert sales automation system. Your task is to generate a highly personalized, professional, and actionable follow-up email in HTML format. 
The email must strictly use the lead's name, reference the summary of the previous call, and include a section about a course demo video.
Output MUST be a single JSON object containing only the keys 'subject' and 'body_html'.
"""

class EmailOutput(BaseModel):
    subject: str = Field(description="The email subject line.")
    body_html: str = Field(description="The full email body in clean HTML format.")

def send_email_via_smtp(recipient_email: str, subject: str, body_html: str) -> bool:
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        logger.error("Email credentials missing in .env.")
        return False
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email
        msg.attach(MIMEText(body_html, 'html'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        logger.info(f"✅ Email SUCCESSFULLY SENT to {recipient_email}.")
        return True
    except Exception as e:
        logger.error(f"🔴 SMTP Email FAILED: {e}")
        return False

def generate_followup(lead_id: str) -> Dict:
    logger.info(f"Agent 4: Starting report generation for Lead {lead_id}...")
    lead = get_lead_by_id(lead_id)

    if not lead or lead.get('interaction', {}).get('call_status') != "Analyzed - Ready for Follow-up":
        return {"success": False, "error": "Lead not ready for Agent 4."}

    analysis = lead.get('analysis', {})
    personal = lead.get('personal', {})
    enrollment = lead.get('enrollment', {})

    lead_name = personal.get('name', 'Valued Customer')
    course_interest = enrollment.get('course_interest', 'General Inquiry')
    next_steps_text = analysis.get('next_steps', 'schedule a follow-up call')
    recipient_email = personal.get('email')

    # 🟢 2. NEW: Identify the correct video link
    # Interest ko match karenge, agar match nahi hua toh empty string
    selected_video_url = COURSE_VIDEO_LINKS.get(course_interest, "")

    # 🟢 3. NEW: Generate Video Section HTML (If link exists)
    video_section_html = ""
    if selected_video_url:
        video_section_html = f"""
        <div style="margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
            <p>Aapne call par <b>{course_interest}</b> mein ruchi dikhayi thi. Hamne aapke liye ek demo video share kiya hai:</p>

            <a href="{selected_video_url}" 
               target="_blank" 
               rel="noopener noreferrer"
               style="background-color: #ff4b4b; color: white; padding: 12px 25px; text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block;">
                📺 Watch {course_interest} Demo Video
            </a>
        </div>
        """

    USER_PROMPT = f"""
    Generate a follow-up email.
    Lead Name: {lead_name}
    Analysis Summary: {analysis.get('summary')}
    Next Step: {next_steps_text}
    Course: {course_interest}
    Video Section HTML: {video_section_html}
    Instruction: Incorporate the Video Section HTML naturally into the body.
    """

    if client is None or MODEL_NAME == 'mock-model':
        email_result = {
            "subject": f"Follow-up: Your {course_interest} Inquiry",
            "body_html": f"<html><body><p>Dear {lead_name},</p><p>Summary: {analysis.get('summary')}</p>{video_section_html}<p>Next Step: {next_steps_text}</p></body></html>"
        }
    else:
        try:
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
            logger.error(f"Gemini Generation Failed: {e}")
            return {"success": False, "error": str(e)}

    # Final DB Update
    update_data = {
        "$set": {
            "interaction.call_status": "Follow-up Sent - Complete",
            "interaction.last_activity": datetime.now(),
            "analysis.report_subject": email_result['subject'],
            "analysis.report_body_html": email_result['body_html']
        }
    }
    update_lead_status(lead_id, update_data)

    if recipient_email:
        send_email_via_smtp(recipient_email, email_result['subject'], email_result['body_html'])

    return {"success": True, "report": email_result}