import requests
import pandas as pd
import streamlit as st
from PIL import Image
import os
from dotenv import load_dotenv
import time
from typing import List, Dict
import base64
import traceback
import numpy as np
from shap import Explanation
import shap
import plotly.graph_objects as go

import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# --- CONFIGURATION & ENVIRONMENT SETUP ---
load_dotenv()
MANAGER_USERNAME = os.getenv("MANAGER_USERNAME")
MANAGER_PASSWORD = os.getenv("MANAGER_PASSWORD")
MAIN_URL = os.getenv("MAIN_URL")
if not MANAGER_USERNAME or not MANAGER_PASSWORD:
    st.error("Missing Environment Variables! Please check your .env file.")
    st.stop()
# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Multi-Agent Lead Manager (Education)",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- CUSTOM CSS INJECTION (omitted for brevity) ---
st.markdown(
    """
    <style>
    /* Global Background */
    .stApp { background-color: #f0f2f6; color: #1e3c72; }
    /* Global Text Color Fix */
    h1, h2, h3, h4, p { color: #1e3c72 !important; }
    /* Navbar Styling */
    .navbar-container {
        background-color: #262626; padding: 10px 20px; border-radius: 30px; 
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3); margin-bottom: 20px;
        display: flex; align-items: center; width: 100%;
    }
    .nav-logo h2 { color: Black !important; margin: 0; }
    /* Always visible border for all input boxes */
    .stTextInput > div > div > input {
        border: 1.5px solid #c3c7d0 !important;
        border-radius: 8px !important;
        padding: 10px !important;
        transition: 0.2s ease;
        background-color: white !important;
    }

    /* Slightly darker border on hover */
    .stTextInput > div > div > input:hover {
        border: 1.5px solid #1e3c72 !important;
    }

    /* Red border on focus (click) like your login theme */
    .stTextInput > div > div > input:focus {
        border: 1.5px solid #ff4b4b !important;
        box-shadow: 0 0 6px rgba(255, 75, 75, 0.4);
    }
    /* Agent Card Styling */
    .agent-box {
        background-color: white; padding: 20px; border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05); text-align: left; 
        transition: transform 0.3s; margin-bottom: 0px !important; 
        padding-bottom: 5px; 
    }
    .agent-box:hover { transform: translateY(-5px); }
    h2 { color: #1e3c72; }
    h3 { color: #6a5acd; }

    /* Image Height Control */
    .short-image-container { max-height: 80px; overflow: hidden; border-radius: 10px; }
    .short-image-container img { width: 100% !important; height: auto !important; object-fit: cover; }
    .short-image-container > div { max-height: 60px !important; padding: 0 !important; }

    </style>
    """,
    unsafe_allow_html=True
)

# =======================================================
# 📌 INITIALIZE SESSION STATE VARIABLES
# =======================================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'fetched_leads' not in st.session_state:
    st.session_state['fetched_leads'] = []
if 'agent2_campaign_results' not in st.session_state:
    st.session_state['agent2_campaign_results'] = None
if 'bulk_followup_results' not in st.session_state:
    st.session_state['bulk_followup_results'] = []
if 'bulk_scoring_results' not in st.session_state:
    st.session_state['bulk_scoring_results'] = []
if 'analysis_target_lead_id' not in st.session_state:
    st.session_state['analysis_target_lead_id'] = None


# =======================================================
# 📌 HELPER FUNCTIONS (DEFINED BEFORE CALLING)
# =======================================================

def check_password(username, password):
    """Mocks secure password check."""
    return username == MANAGER_USERNAME and password == MANAGER_PASSWORD


@st.cache_data
def get_img_as_base64(file):
    """Encodes a local image file to a base64 string for embedding in HTML."""
    try:
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return ""


def handle_login_submit(username, password):
    """Processes login attempt."""
    if check_password(username, password):
        st.session_state.logged_in = True
        st.success("Login Successful! Redirecting to Agent Dashboard...")
        time.sleep(0.5)
        st.session_state.page = 'Dashboard'
        st.rerun()
    else:
        st.error("Invalid Username or Password.")


def safe_post(url: str, json_payload: dict = None, timeout: int = 120):  # 30
    """
    Safely perform a POST request and return (ok: bool, payload_or_error).
    """
    try:
        resp = requests.post(url, json=json_payload or {}, timeout=timeout)
        if 200 <= resp.status_code < 300:
            try:
                return True, resp.json()
            except Exception:
                return True, {"raw_text": resp.text}
        else:
            try:
                return False, resp.json()
            except Exception:
                return False, {"status_code": resp.status_code, "text": resp.text}
    except Exception as e:
        return False, {"exception": str(e), "trace": traceback.format_exc()}


def safe_get(url: str, timeout: int = 120):  # 10
    """
    Safely perform a GET request and return (ok: bool, payload_or_error).
    """
    try:
        resp = requests.get(url, timeout=timeout)
        if 200 <= resp.status_code < 300:
            try:
                return True, resp.json()
            except Exception:
                return True, {"raw_text": resp.text}
        else:
            try:
                return False, resp.json()
            except Exception:
                return False, {"status_code": resp.status_code, "text": resp.text}
    except Exception as e:
        return False, {"exception": str(e), "trace": traceback.format_exc()}


def fetch_analyzed_leads_from_db(max_leads: int = 50) -> List[Dict]:
    """
    Calls the FastAPI backend (Agent 1 endpoint) to fetch leads that have transcripts.
    This function retrieves leads that have a final transcript saved (i.e., completed calls).
    """
    # This prompt tells Agent 1 to look for ANY lead that has a transcript (mock or real)
    special_prompt = "Fetch leads completed mock and awaiting analysis"
    API_URL = f"{MAIN_URL}/api/v1/agent1/fetch_leads"

    try:
        response = requests.post(
            API_URL,
            json={"manager_prompt": special_prompt, "max_leads": max_leads},
            timeout=60  # 5-60
        )
        if response.status_code == 200:
            all_fetched_leads = response.json()
            # Filter ensures only leads with a non-null transcript are returned
            processed_leads = [
                lead for lead in all_fetched_leads
                if lead.get('analysis', {}).get('transcript') is not None
            ]
            return processed_leads
        return []
    except Exception:
        return []


def render_footer():
    """Renders the standard footer."""
    st.markdown("---")
    st.subheader("Connect with Us")
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        st.markdown("CONTACT INFO")
        st.write("✉ scighd@gmail.com")
        st.write("📞 +91-7665224106")
    with col_f2:
        st.markdown("FEATURES")
        st.write("CRM with Dialer")
        st.write("WhatsApp CRM")
    with col_f3:
        st.markdown("INDUSTRIES")
        st.write("CRM For Education")
        st.write("Real Estate CRM")
    with col_f4:
        st.markdown("RESOURCES")
        st.write("Pricing")
        st.write("Case studies")
        st.button("Request a Demo", key="footer_demo")
    st.markdown(
        "<p style='text-align: center; margin-top: 20px;'>@IERT AI multiagent sales lead management</p>",
        unsafe_allow_html=True)


# --- RENDER NAV BAR ---
def render_navbar(logged_in=False):
    """Renders the custom dark, pill-shaped navigation bar."""

    st.markdown('<div class="navbar-container">', unsafe_allow_html=True)

    col_logo, col_home, col_about, col_contact, col_search, col_login = st.columns([1, 0.5, 0.7, 0.7, 1.5, 1])

    with col_logo:
        logo_base64 = get_img_as_base64("assets/logo.png")
        if logo_base64:
            st.markdown(f'<img src="data:image/png;base64,{logo_base64}" width="100">', unsafe_allow_html=True)
        else:
            st.markdown("<div class='nav-logo'><h2>🎓InspireEd</h2></div>", unsafe_allow_html=True)

    with col_home:
        if st.button("Home", key="nav_home_r"): st.session_state.page = 'Home'
    with col_about:
        if st.button("About Agents", key="nav_about_r"): st.session_state.page = 'About'
    with col_contact:
        if st.button("Contact", key="nav_contact_r"): st.session_state.page = 'Contact'

    with col_login:
        if not logged_in:
            if st.button("🔑 Login/Signup", key="trigger_login", width='stretch'):
                st.session_state.page = 'Login'
        else:
            if st.button("Logout", key="nav_logout_r", width='stretch'):
                st.session_state.logged_in = False
                st.session_state.page = 'Home'
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def render_home_page():
    """Renders the 6-section Homepage (unlocked content)."""
    try:
        banner = Image.open("assets/main.png")
        resized_banner = banner.resize((1200, 460))  # width x height (adjust height freely)
        st.image(resized_banner, use_container_width=True, caption="Real-time Lead Dashboard")

    except Exception as e:
        st.error(f"⚠ Unable to load home image: {e}")
        st.info("Home banner not available.")
    # Placeholder
    st.markdown("""
    <div style="text-align: center; padding: 40px 0; background-color: #f2f2f2; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
        <h1 style="color: #6a5acd;">India's Simplest Lead Management System for Education</h1>
        <h3 style="color: #1e3c72;">AI-Driven Multi-Agent System to Boost Enrollment and Sales Efficiency.</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<h2 style='text-align:center;'>🎯 Unlock Enrollment Growth with AI Automation</h2>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📞 Real-Time Voice Engagement")
        st.info(
            "Agent 2 autonomously conducts human-like conversations over voice, qualifying leads instantly with low-latency barge-in support.")
    with col2:
        st.markdown("### 💡 Intelligent Analysis")
        st.info(
            "Agent 3 provides deep NLP summaries, capturing sentiment, intent, and next steps for every interaction, turning raw data into actionable insights.")
    with col3:
        st.markdown("### 🎯 Dynamic Lead Scoring")
        st.info(
            "Agent 5 continuously updates lead scores using ML, prioritizing the hottest leads based on behavior and voice interaction data.")

    st.markdown("---")
    center_col_s3_1, center_col_s3_2, center_col_s3_3 = st.columns([1, 3, 1])

    with center_col_s3_2:
        st.markdown(
            f"""
            <h2 style="text-align: center; color: #1e3c72;">
                Book a Live CRM Demo to See InspireEd/Our System in Action
            </h2>
            <p style="text-align: center; color: #1e3c72;">
                Spend 1 hour and learn how to:
            </p>
            """,
            unsafe_allow_html=True
        )

    demo_col1, demo_col2 = st.columns([2, 2])

    with demo_col1:
        st.markdown("""### Key Benefits Overview ...""")
        st.subheader("Watch a Quick Demo Video")
        st.markdown("[▶ Watch Demo Video](https://youtu.be/jX4dLxiso6A)")

    with demo_col2:
        with st.form("demo_form"):
            team_size = st.radio("Select team size:", ("Up to 2", "3-5", "6-10", "11-20", "20 plus"), index=4,
                                 key="team_size_radio", label_visibility="collapsed")
            submitted = st.form_submit_button("Continue", type="primary")
            if submitted:
                st.success(f"Demo requested for team size: {team_size}. We'll contact you shortly!")

    st.markdown("---")
    st.markdown(
        "<h2 style='text-align:center;'>✨ Comprehensive Lead Management Benefits</h2>",
        unsafe_allow_html=True
    )

    st.markdown('<div class="short-image-container">', unsafe_allow_html=True)
    try:
        banner = Image.open("assets/benefit.png")
        resized_banner = banner.resize((1200, 460))  # width x height (adjust height freely)
        st.image(resized_banner, use_container_width=True)

    except Exception as e:
        st.error(f"⚠ Unable to load benefit image: {e}")
        st.info("Dashboard banner not available.")
    st.markdown('</div>', unsafe_allow_html=True)
    render_footer()


def render_about_page():
    """Renders the About page with multi-agent information and image."""
    st.title("💡 About the Multi-Agent System")
    st.subheader("AI-Driven Multi-Agent System to Boost Enrollment and Sales Efficiency.")
    st.image("assets/img_2.png", width='stretch')  # Placeholder
    st.markdown("""
    This system utilizes five specialized AI agents working in concert to automate the entire sales lead lifecycle, ensuring speed, personalization, and data-driven follow-up for the education domain.
    """)
    st.markdown("""
    ### Agent Specializations
    1. Agent 1 (Data Fetcher): Interprets natural language requests ("Call all high-scoring leads in Mumbai") and queries MongoDB for the required data.
    2. Agent 2 (Voice Initiator): Orchestrates the real-time voice conversation via bolna.ai, handling STT, LLM response generation, and low-latency TTS.
    3. Agent 3 (Conversation Analyzer): Processes call transcripts post-call to generate structured summaries, sentiment scores, and identify key discussion points.
    4. Agent 4 (Reporter/Follow-up): Formats the analysis into a concise report and uses Generative AI to draft and send highly personalized follow-up emails.
    5. Agent 5 (Lead Scorer): Runs continuous ML models to dynamically re-rank leads based on interaction history and new data from call analysis.
    """)
    render_footer()


def render_contact_page():
    """Renders the Contact page with 4 team members and uniform circular photos."""
    st.title("📞 Contact Us")
    st.subheader("The Team Behind the System")
    st.markdown("---")

    # 🟢 CSS Fix: Sabhi images ko ek jaisa circular aur fixed size banane ke liye
    st.markdown("""
        <style>
        .team-img {
            width: 110px !important;
            height: 110px !important;
            border-radius: 50% !important;
            object-fit: cover !important;
            display: block;
            margin-left: auto;
            margin-right: auto;
            border: 2px solid #1e3c72;
        }
        </style>
    """, unsafe_allow_html=True)

    team_members = [
        {
            "name": "Anant Singh",
            "role": "AI Interaction & Voice Engineer",
            "phone": "+91 7665224106",
            "email": "scighd@gmail.com",
            "linkedin": "#",
            "img": "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
        },
        {
            "name": "Shivanand Sharma",
            "role": "Lead Developer/ML Specialist",
            "phone": "+91 8756315251",
            "email": "shivanandsharma7322@gmail.com",
            "linkedin": "https://www.linkedin.com/in/shivanand-sharma4a28b0257",
            "img": "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
        },
        {
            "name": "Ritesh Kumar Jaiswal",
            "role": "System Architect",
            "phone": "+91 6391954439",
            "email": "Rits14688@gmail.com",
            "linkedin": "https://www.linkedin.com/in/ritesh-jaiswal5",
            "img": "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
        },
        {
            "name": "Saksham Singh",
            "role": "Frontend / UI/UX Designer",
            "phone": "+91 9140397954",
            "email": "xiia3sakshamsingh49@gmail.com",
            "linkedin": "#",
            "img": "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
        }
    ]

    # Displaying in a Grid (2x2)
    for i in range(0, len(team_members), 2):
        col1, col2 = st.columns(2)
        for idx, member in enumerate(team_members[i:i + 2]):
            current_col = col1 if idx == 0 else col2
            with current_col:
                inner_col_img, inner_col_txt = st.columns([1, 2])
                with inner_col_img:
                    st.markdown(f'<img src="{member["img"]}" class="team-img">', unsafe_allow_html=True)
                with inner_col_txt:
                    st.subheader(member["name"])
                    st.markdown(f"""
                    - **Role:** {member['role']}
                    - **Phone:** {member['phone']}
                    - **Email:** {member['email']}
                    - [LinkedIn Profile]({member['linkedin']})
                    """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

    render_footer()


def render_login_page():
    """Renders a centered login/signup form with reduced width."""
    st.markdown("<h1 style='text-align:center;'>Manager Login / Signup</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Create 3 columns → center column becomes the login box
    col_left, col_center, col_right = st.columns([1, 1, 1])

    with col_center:
        login_container = st.container(border=True)
        with login_container:
            # ---- ADD USER ICON HERE ----
            st.markdown(
                """
                <div style="text-align:center; margin-top:-10px;">
                    <span style="font-size:70px;">👤</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            # --------------------------------

            st.markdown(
                "<h3 style='text-align:center; margin-bottom:20px;'>Enter Manager Credentials</h3>",
                unsafe_allow_html=True
            )

            login_username = st.text_input("Username", key="login_u")
            login_password = st.text_input("Password", type="password", key="login_p")

            if st.button("Secure Login", key="do_login", type="primary", use_container_width=True):
                handle_login_submit(login_username, login_password)


def render_agent1_page():
    """Renders the dedicated page for Agent 1 Lead Pull (Fetch Phase)."""

    st.header("⿡ Agent 1: Lead Data Fetcher")
    # st.subheader("Targeted Lead Retrieval for Voice Campaign Initiation")
    try:
        banner = Image.open("assets/database.jpg")  # Place your banner img here
        resized_banner = banner.resize((1200, 460))  # reduce height
        st.image(resized_banner, use_container_width=True)

        st.markdown("""
               <div style='text-align:center; font-size:20px; font-weight:bold; margin-top:10px;'>
                   🚀 <i>"Right leads. Right time. Right impact — Let Agent 1 find your perfect prospects."</i> 🚀
               </div>
           """, unsafe_allow_html=True)

    except:
        st.info("🎯 Fetch targeted leads and get ready to launch automated calling via Agent 2.")

    st.button("⬅ Back to Dashboard", key="back_to_dash", on_click=lambda: st.session_state.update(page='Dashboard'))
    st.markdown("---")
    st.subheader("📂 Google Sheets Automation")
    sync_col, status_col = st.columns([2, 1])

    with sync_col:
        st.write("Sync leads directly from your connected Google Sheet.")
        sheet_name = st.text_input("Google Sheet Name", value="Sales_Leads_2026")

        # 1. Sirf button click hone par hi niche ka logic chalna chahiye
        if st.button("🔄 Sync New Leads from Sheets", type="primary", width='stretch'):

            import sys
            import os
            import time

            # Path Setup
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, "db")

            if db_path not in sys.path:
                sys.path.append(db_path)

            try:
                # 2. Lazy Import: Sirf click hone par import karein
                from database import sync_google_sheets_to_db

                with st.spinner("Fetching latest data from Google Sheets..."):
                    count = sync_google_sheets_to_db(sheet_name)

                    if count > 0:
                        st.success(f"🔥 Dhamaka! {count} naye leads database mein add ho gaye.")
                        time.sleep(1.5)
                        st.rerun()
                    elif count == 0:
                        st.info("✅ Sab kuch up-to-date hai. Koi naya lead nahi mila.")
                    else:
                        st.error("❌ Sync fail! Check credentials.json or Sheet sharing.")

            except Exception as e:
                st.error(f"⚠️ Import Error: 'database.py' nahi mili. Details: {e}")

    st.markdown("---")
    col_form, col_info = st.columns([2, 1])

    with col_form:
        st.info(
            "🎯 Instruct the AI: Enter a natural language request to filter leads by location, score, or status.")
        with st.form("agent1_form_page", border=False):
            prompt = st.text_input("Manager Prompt",
                                   value="Call all pending leads in Gurugram with a high-scoring profile",
                                   key="manager_prompt_page")
            max_leads_to_fetch = st.slider("Max Leads to Fetch", min_value=1, max_value=50, value=10,
                                           key="max_leads_slider_fetch")

            max_calls_to_initiate = st.slider(
                "Max Calls to Initiate (Campaign Size)",
                min_value=1,
                max_value=max_leads_to_fetch,  # Limit max calls to max fetched
                value=5,
                key="max_calls_slider_initiate"
            )

            submitted = st.form_submit_button("▶ Run Agent 1 & Fetch Leads", type="primary", width='stretch')

        # --- API CALL LOGIC (Fetch Leads) ---
        if submitted:
            API_URL = f"{MAIN_URL}/api/v1/agent1/fetch_leads"
            with st.spinner(f"Agent 1 interpreting prompt and fetching leads from MongoDB..."):
                try:
                    response = requests.post(API_URL, json={"manager_prompt": prompt, "max_leads": max_leads_to_fetch},
                                             timeout=140)  # 60
                    if response.status_code == 200:
                        st.session_state['fetched_leads'] = response.json()
                        st.success(
                            f"Agent 1 successfully fetched {len(st.session_state['fetched_leads'])} leads! Ready for Agent 2.")
                    elif response.status_code == 404:
                        st.warning(
                            f"Agent 1 found no leads matching the request. Check the database or try a simpler prompt.")
                        st.session_state['fetched_leads'] = []
                    else:
                        st.error(
                            f"Error connecting to backend API. Status: {response.status_code}. Detail: {response.text}")
                        st.session_state['fetched_leads'] = []
                except Exception as e:
                    st.error(f"Connection/Unexpected Error: {e}")

    with col_info:
        st.markdown("#### Agent 1 Role Details")
        st.markdown("""
        - Input: Manager's natural language text prompt.
        - Process: Filters database based on Location and Score.
        - Output: Structured list of leads ready for voice outreach.
        """, unsafe_allow_html=True)
        # st.markdown(f"Database URI: {os.getenv('MONGO_URI', 'localhost:27017')}")

    st.markdown("---")

    # --- DISPLAY FETCHED LEADS (Full Width) ---
    if st.session_state['fetched_leads']:
        st.subheader(f"✅ Fetched Leads ({len(st.session_state['fetched_leads'])})")

        df_data = []
        for lead in st.session_state['fetched_leads']:
            df_data.append({
                "ID": lead['lead_id'], "Name": lead['personal']['name'], "Location": lead['personal']['location'],
                "Course": lead['enrollment']['course_interest'], "Score": lead['score']['current_score'],
                "Status": lead['interaction']['call_status'], "Phone": lead['personal']['phone_number']
            })
        df = pd.DataFrame(df_data)
        st.dataframe(df, width='stretch')

        # --- NEXT AGENT BUTTON (Agent 2 Trigger) ---
        if st.button(f"📞 Send {max_calls_to_initiate} Leads to Agent 2 (Initiate Call Campaign)", type="secondary",
                     width='stretch'):
            AGENT2_API_URL = f"{MAIN_URL}/api/v1/agent2/initiate_call"
            campaign_list = st.session_state['fetched_leads'][:max_calls_to_initiate]

            with st.spinner(f"Agent 2 initiating {len(campaign_list)} voice calls via bolna.ai..."):
                try:
                    response = requests.post(
                        AGENT2_API_URL,
                        json={"leads_to_call": campaign_list, "max_calls": len(campaign_list)},
                        timeout=60
                    )
                    if response.status_code == 200:
                        st.session_state['agent2_campaign_results'] = response.json()
                        st.session_state.page = 'Agent2Page'
                        st.rerun()
                    else:
                        st.error(f"Agent 2 API Error. Status: {response.status_code}. Detail: {response.text}")
                except Exception as e:
                    st.error(f"Connection/Unexpected Error: {e}")
    else:
        st.info("No leads fetched yet. Run the query above to populate the table.")

    render_footer()


def render_dashboard_page():
    """Renders the Agent Dashboard (only accessible when logged in)."""
    st.title("Welcome, Sales Manager! 📊")
    st.subheader("AI Agent Control Panel")

    try:
        banner = Image.open("assets/dashboard.jpg")
        resized_banner = banner.resize((1200, 560))  # width x height (adjust height freely)
        st.image(resized_banner, use_container_width=True, caption="Real-time Lead Dashboard")

    except Exception as e:
        st.error(f"⚠ Unable to load dashboard image: {e}")
        st.info("Dashboard banner not available.")

    st.markdown("""
         <p style='font-size:1.1rem; text-align:center; margin-top:10px;'>
             🚀 Manage leads, calls, follow-ups & scoring — all in one AI-powered dashboard!
         </p>
         """, unsafe_allow_html=True)
    # Placeholder removed

    st.markdown("""
    <p style='font-size:1.1rem;'>Click on any agent below to initiate its function or view its status.
    <a href="javascript:void(0)">Learn More about Agent Workflow</a></p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Agent Workflow Status")

    agent_cols = st.columns(5)

    agent_info = {
        1: ("Data Fetcher", "Retrieves leads from MongoDB.", "agent1_data_fetcher.py", "Agent 1: Initiate Lead Pull"),
        2: ("Voice Initiator", "Orchestrates real-time voice calls (bolna.ai).", "agent2_initiator.py",
            "Agent 2: Start Calling Campaign"),
        3: ("Conv. Analyzer", "Summarizes call transcripts & sentiment.", "agent3_analyzer.py",
            "Agent 3: View Analysis Dashboard"),
        4: ("Follow-up", "Generates reports & personalized emails.", "agent4_reporter.py",
            "Agent 4: Send Follow-ups/Reports"),
        5: ("Lead Scorer", "Continuously ranks leads using ML.", "agent5_scorer.py",
            "Agent 5: Review Scoring Model"),
    }

    for i in range(5):
        agent_id = i + 1
        name, desc, file, action = agent_info[agent_id]

        with agent_cols[i]:
            st.markdown(f"""
            <div class="agent-box">
                <h3>Agent {agent_id}: {name}</h3>
                <p style="min-height: 50px; font-weight: bold; color: #6a5acd;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

            button_key = f"dash_btn_{agent_id}"
            button_type = "primary" if agent_id == 1 or agent_id == 2 else "secondary"

            if agent_id == 1:
                if st.button(action, key=button_key, type=button_type, width='stretch'):
                    st.session_state.page = 'Agent1Page';
                    st.rerun()

            elif agent_id == 2:
                # Direct navigation to the Live Monitor is the best UX for real-time Agent 2 stage
                if st.button("🎙 Real-Time Call Monitor", key=button_key, type=button_type, width='stretch'):
                    st.session_state.page = 'LiveMonitor';
                    st.rerun()

            elif agent_id == 3:
                # Direct navigation to the Analysis Page for Agent 3 stage
                if st.button("💡 View Analysis Dashboard", key=button_key, type="primary", width='stretch'):
                    st.session_state.page = 'CompletedLeadsPage';  # Directs to the review list
                    st.rerun()

            elif agent_id == 5:
                # Direct navigation to Agent 3 Page to run scoring
                if st.button("🎯 Review Scoring Model", key=button_key, type="secondary", width='stretch'):
                    st.session_state.page = 'Agent3Page';  # We can reuse Agent3Page to display final scores
                    st.rerun()

            else:
                st.button(action, key=button_key, type="secondary", width='stretch', disabled=True)
    render_footer()


def render_agent2_page():
    """
    Renders the dedicated page for Agent 2 execution, displaying campaign metrics
    and providing navigation to the live monitor and post-call review.
    """
    st.title("📞 Agent 2: Voice Campaign Results")
    st.subheader("Reviewing leads after call initiation.")

    st.button("⬅ Back to Lead Fetcher (Agent 1)", key="back_to_fetcher",
              on_click=lambda: st.session_state.update(page='Agent1Page'))

    st.button("▶ Go to Live Call Monitor", key="go_to_monitor_from_A2",
              on_click=lambda: st.session_state.update(page='LiveMonitor'))

    # 🟢 NEW: Dedicated button to go to the conversation review section
    if st.button("👁 View Completed Conversations for Analysis", key="view_completed_reviews", type="primary"):
        st.session_state.page = 'CompletedLeadsPage'
        st.rerun()

    st.markdown("---")

    campaign_results = st.session_state.get('agent2_campaign_results')

    if campaign_results:
        st.header(f"Campaign Initiated: {campaign_results['total_leads']} leads targeted")

        col_s, col_f = st.columns(2)
        col_s.metric("✅ Successful Calls Initiated (Awaiting Webhook)",
                     value=campaign_results.get('successful_calls', 0))
        col_f.metric("❌ Failed Calls (Mock Transcript Generated)", value=campaign_results.get('failed_calls', 0))
        st.markdown("---")

    st.info("The conversation review has been moved to the 'View Completed Conversations' page for clarity.")

    render_footer()


def render_completed_reviews_page():
    """
    Dedicated page to display the final conversation transcripts for all completed leads
     and provides the button to trigger Agent 3 Analysis.
    """
    st.title("📚 Conversation Review & Agent 3 Trigger")
    st.subheader("Review the full, final transcript for every completed lead.")
    try:
        banner = Image.open("assets/conversation.jpg")  # Make sure image exists in assets folder
        resized_banner = banner.resize((1200, 460))  # Width x Height for smaller UI space
        st.image(resized_banner, use_container_width=True)

        st.markdown("""
                <div style='text-align: center; font-size: 20px; font-weight: bold; margin-top: 10px;'>
                    💡 <i>"Great conversations deserve meaningful insights — review and analyze with precision."</i> 💡
                </div>
                """,
                    unsafe_allow_html=True
                    )

    except Exception:
        st.info("📍 Review completed conversations and run analysis to extract insights.")
    st.button("⬅ Back to Campaign Results", key="back_to_agent2_results_c",
              on_click=lambda: st.session_state.update(page='Agent2Page'))
    st.markdown("---")

    st.header("💬 Completed Conversations (Post-Call)")

    # CRITICAL: Fetch leads that received a transcript (Mock or Real)
    leads_with_transcripts = fetch_analyzed_leads_from_db(max_leads=50)

    if leads_with_transcripts:
        st.info(f"Found {len(leads_with_transcripts)} conversations in the database.")

        # 🟢 NEW: BULK ACTION BUTTON
        # This button triggers the pipeline hub for the first unanalyzed lead
        if st.button("📧 Generate Follow-up Emails for ALL Completed Leads", key="trigger_bulk_followup",
                     type="secondary", width='stretch'):
            st.session_state.page = 'Agent4Page'  # Move to the Agent 4 Bulk Executor page
            st.rerun()

        st.markdown("---")

        for lead in leads_with_transcripts:
            name = lead['personal']['name']
            lead_id = lead['lead_id']
            transcript = lead.get('analysis', {}).get('transcript', 'Transcript not found.')

            # Check if analysis is already complete
            is_analyzed = lead.get('analysis', {}).get('summary') is not None
            current_status = lead['interaction']['call_status']

            expander_label = f"{name}** ({lead_id}) - Status: {current_status} {'(✅ Analyzed)' if is_analyzed else ''}"

            with st.expander(expander_label):

                # --- Display Analysis (if available) ---
                if is_analyzed:
                    st.success("Analysis Complete!")
                    st.markdown(
                        f"Sentiment: {lead['analysis']['sentiment']} | Next Step: {lead['analysis']['next_steps']}")
                    st.markdown("### 📝 Summary")
                    st.info(lead['analysis']['summary'])
                    st.markdown("---")

                st.markdown("#### Full Call Transcript (Source: MongoDB)")

                # --- Display Transcript using chat elements ---
                if transcript and transcript != 'Transcript not found.':
                    lines = transcript.strip().split('\n')

                    for line in lines:
                        if 'assistant:' in line.lower() or 'agent:' in line.lower():
                            role, avatar, content = "assistant", "🤖", line.split(':', 1)[-1].strip()
                        elif 'user:' in line.lower() or 'customer:' in line.lower():
                            role, avatar, content = "user", "👤", line.split(':', 1)[-1].strip()
                        else:
                            # Handle lines that don't match the simple speaker format
                            continue

                        with st.chat_message(role, avatar=avatar):
                            st.write(content)
                else:
                    st.warning("Transcript is missing or call data was lost.")

                if not is_analyzed:
                    # Button to start analysis for this specific lead
                    st.button(f"🧠 Analyze Conversation for {lead_id}", key=f"review_analyze_{lead_id}",
                              on_click=lambda id=lead_id: st.session_state.update(
                                  page='Agent3Page', analysis_target_lead_id=id),
                              type="secondary")

    else:
        st.info("No completed conversations are available for review.")

    render_footer()


def render_agent3_page():
    """
    Handles the Agent 3 Analysis trigger and displays the results.
    """
    st.title("💡 Agent 3: Conversation Analysis")
    st.subheader("Process Transcript and Generate Structured Insights")

    st.button("⬅ Back to Dashboard", key="back_to_dash_a3",
              on_click=lambda: st.session_state.update(page='Dashboard'))

    # 🟢 Added button to go back to the list of completed leads
    st.button("↩ Review Completed Leads", key="back_to_completed_list",
              on_click=lambda: st.session_state.update(page='CompletedLeadsPage'))

    st.markdown("---")

    # --- 1. TARGET LEAD SELECTION & STATUS CHECK ---
    st.header("1. Target Lead for Processing")

    # Use the lead ID passed from Agent 2/Completed Page, or default to a test ID
    target_lead_id = st.session_state.get('analysis_target_lead_id') or st.text_input(
        "Enter Lead ID:", value='LMS-101', key='a3_manual_lead_id')

    st.session_state['analysis_target_lead_id'] = target_lead_id

    # 2. Fetch Lead Data (Runs constantly to check status)
    lead_data = {}
    if target_lead_id:
        try:
            response = requests.get(f"{MAIN_URL}/api/v1/lead/{target_lead_id}", timeout=3)
            if response.status_code == 200:
                lead_data = response.json()
        except Exception:
            st.error("Could not fetch lead data from FastAPI.")
            return

    latest_analysis = lead_data.get('analysis', {})
    current_status = lead_data.get('interaction', {}).get('call_status', 'N/A')

    # --- Display Current Status Cards (Metrics) ---
    col_status, col_score, col_sentiment, col_next = st.columns(4)

    col_status.metric("Status", current_status)
    col_score.metric("Score", lead_data.get('score', {}).get('current_score', 'N/A'))
    col_sentiment.metric("Sentiment", str(latest_analysis.get('sentiment', 'N/A')).upper())
    col_next.metric("Next Action", str(latest_analysis.get('next_steps', '---')))

    st.markdown("---")

    # ----------------------------------------------------------------------
    # 🟢 AGENT 3 ACTION BUTTON (SEQUENTIAL TRIGGER)
    # ----------------------------------------------------------------------
    a3_can_run = current_status == 'Transcript Received'
    is_analyzed = current_status == 'Analyzed - Ready for Follow-up'  # For status check

    # 1. RUN AGENT 3 (Analyze)
    if a3_can_run:
        if st.button("🧠 Run Agent 3 (Analyze)", key='btn_a3', type='primary', use_container_width=True):
            API_URL = f"{MAIN_URL}/api/v1/agent3/analyze_lead/{target_lead_id}"
            with st.spinner("Agent 3 running NLP analysis..."):
                response = requests.post(API_URL, timeout=90)
                if response.status_code == 200:
                    st.success("✅ Analysis Complete! Redirecting to Follow-up.")
                    st.session_state.page = 'Agent4Page'  # Move to Agent 4 page
                    st.rerun()
                else:
                    st.error(f"Analysis Failed: {response.json().get('detail', 'API Error')}")

    # 2. Sequential Button Check (If A3 is done, show A4 button)
    elif is_analyzed:
        st.info("Analysis is ready. Proceed to follow-up generation.")
        if st.button("📧 Proceed to Agent 4 (Follow-up)", key='btn_a4_redirect', type='secondary',
                     width='stretch'):
            st.session_state.page = 'Agent4Page'
            st.rerun()

    # 3. Final State Message
    elif current_status.startswith('Scored'):
        st.success("🎉 Pipeline Complete!")
    else:
        st.warning("Lead is not yet ready. Ensure the call status is 'Transcript Received' to begin.")

    st.markdown("---")
    st.header("Results & Transcript")

    # --- 3. DISPLAY ANALYSIS AND TRANSCRIPT ---
    if latest_analysis.get('summary'):
        col_s, col_n = st.columns(2)
        col_s.metric("Overall Sentiment", str(latest_analysis.get('sentiment', 'N/A')).upper())
        col_n.metric("Next Action", str(latest_analysis.get('next_steps', 'N/A')))

        st.markdown("### 📝 Summary")
        st.info(latest_analysis.get('summary'))

    with st.expander("View Source Transcript"):
        st.code(latest_analysis.get('transcript', 'Transcript not available.'), language='text')

    render_footer()


# ======================================================
# AGENT 4 PAGE: BULK FOLLOW-UP EXECUTION
# ======================================================
def render_agent4_followup_page():
    if not st.session_state.logged_in:
        st.warning("Login required.")
        return

    st.title("📧 Agent 4: Bulk Follow-Up Generation")
    st.subheader("Generating personalized emails for session-fetched leads only.")

    from PIL import Image
    try:
        banner = Image.open("assets/follow-ban.jpg")
        resized_banner = banner.resize((1200, 460))
        st.image(resized_banner, use_container_width=True)

        st.markdown("""
            <div style='text-align: center; font-size: 20px; font-weight: bold; margin-top: 10px;'>
                ✨ <i>"Every follow-up is a new opportunity — let's convert conversations into success."</i> ✨
            </div>
            """, unsafe_allow_html=True)
    except:
        st.info("🚀 Let's begin follow-up email generation for your session leads!")

    st.button("⬅ Back to Analysis Hub", key="back_to_a3_hub_4",
              on_click=lambda: st.session_state.update(page='Agent3Page'))

    st.markdown("---")

    # 🟢 CHANGE: Pura database fetch karne ki jagah, sirf session mein fetch kiye gaye leads lein
    leads_to_process = st.session_state.get('fetched_leads', [])

    if not leads_to_process:
        st.warning("⚠️ Koi leads fetch nahi kiye gaye hain. Pehle Agent 1 ka upyog karke leads fetch karein.")
        if st.button("Go to Agent 1"):
            st.session_state.page = 'Agent1Page'
            st.rerun()
        return

    st.markdown(
        f"""
        <div style="
            text-align:center; 
            font-size:15px; 
            font-weight:bold; 
            padding:12px; 
            background-color:#e8f0fe;
            border-radius:8px;
            border:1px solid #b9ccee;
            margin-top:5px;
            margin-bottom:5px;
        ">
            Ready to generate follow-up emails for {len(leads_to_process)} session leads 🚀
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- EXECUTION ---
    if st.button("▶ START BULK FOLLOW-UP (Current Session Only)", type="primary", width='stretch'):

        # Reset results container before new run
        st.session_state.bulk_followup_results = []

        progress_bar = st.progress(0, text="Starting session-based email generation...")

        for i, lead in enumerate(leads_to_process, start=1):
            lead_id = lead["lead_id"]

            # 1. Run Agent 3 Analysis
            safe_post(f"{MAIN_URL}/api/v1/agent3/analyze_lead/{lead_id}", json_payload={}, timeout=60)

            # 2. Run Agent 4 Follow-up
            API_URL = f"{MAIN_URL}/api/v1/agent4/generate_followup/{lead_id}"
            ok, resp = safe_post(API_URL, json_payload={}, timeout=60)

            # Fetch updated data to display result
            ok_final, lead_final = safe_get(f"{MAIN_URL}/api/v1/lead/{lead_id}")

            if ok and ok_final:
                report = resp.get('report', {})
                st.session_state.bulk_followup_results.append({
                    "Lead ID": lead_id,
                    "Name": lead['personal'].get('name', 'N/A'),
                    "Status": "SUCCESS",
                    "Email Subject": report.get("report_subject", "N/A"),
                    "Email Body": lead_final.get('analysis', {}).get('report_body_html', 'N/A')
                })
            else:
                st.session_state.bulk_followup_results.append({
                    "Lead ID": lead_id,
                    "Name": lead['personal'].get('name', 'N/A'),
                    "Status": "FAIL",
                    "Email Subject": "ERROR",
                    "Email Body": f"API Failed for lead {lead_id}"
                })

            progress_bar.progress(i / len(leads_to_process), text=f"Processing: {lead_id}")

        st.success("🎉 Session-based Bulk Follow-up Complete!")
        progress_bar.empty()
        st.rerun()

        # --- DISPLAY RESULTS ---
    if st.session_state.get('bulk_followup_results'):
        st.markdown("## 📬 Generated Email Previews")
        for email in st.session_state['bulk_followup_results']:
            with st.expander(f"📧 {email['Name']} ({email['Lead ID']}) - Status: {email['Status']}"):
                st.markdown(f"### Subject: {email['Email Subject']}")
                if email["Status"] == "FAIL":
                    st.error(email["Email Body"])
                else:
                    st.components.v1.html(email["Email Body"], height=250, scrolling=True)

    st.markdown("---")

    # 🎯 Proceed to Agent 5
    if st.session_state.get('bulk_followup_results'):
        if st.button("🎯 Proceed To Agent 5 Scoring", type="primary", width='stretch'):
            st.session_state.page = 'Agent5Page'
            st.rerun()

    render_footer()


def render_agent5_scoring_page():
    """
    Triggers Agent 5 scoring for the target lead and displays the final score and tag.
    Updated to show all eligible leads for scoring.
    """
    st.title("🎯 Agent 5: Bulk Lead Scoring")
    st.subheader("Calculating Final Priority Scores for Follow-up Leads")

    try:
        banner = Image.open("assets/score.jpg")
        resized_banner = banner.resize((1200, 360))
        st.image(resized_banner, use_container_width=True)

        st.markdown("""
               <div style='text-align: center; font-size: 20px; font-weight: bold; margin-top: 10px;'>
                   💡 <i>"Scoring insights that drive smarter decisions — every number tells a story."</i> 💡
               </div>
               """, unsafe_allow_html=True)
    except Exception:
        st.info("📍 Ready to calculate final lead scores and prioritize follow-ups efficiently!")

    st.button("⬅ Back to Follow-up Page", key="back_to_a4_hub_score",
              on_click=lambda: st.session_state.update(page='Agent4Page'))
    st.markdown("---")

    # 🟢 UPDATE 1: Fetch leads and allow multiple statuses
    leads = fetch_analyzed_leads_from_db(50)

    # Hum un sabhi leads ko allow karenge jinpar analysis ho chuka hai ya follow-up ho chuka hai
    allowed_statuses = ["Follow-up Sent - Complete", "Analyzed - Ready for Follow-up", "Transcript Received"]

    leads_for_scoring = [
        lead for lead in leads
        if any(status in lead.get('interaction', {}).get('call_status', '') for status in allowed_statuses)
           or "Scored" in lead.get('interaction', {}).get('call_status', '')
    ]

    if not leads_for_scoring and not st.session_state.get('bulk_scoring_results'):
        st.warning("⚠️ No leads found ready for scoring. Ensure calls have been analyzed by Agent 3.")
        render_footer()
        return

    # Check session state for running status
    if 'bulk_scoring_ran' not in st.session_state:
        st.session_state['bulk_scoring_ran'] = False

    st.info(f"Found {len(leads_for_scoring)} leads eligible for final scoring.")

    # --- SCORING EXECUTION ---
    # Sirf unhi leads ko process karenge jo abhi tak score nahi hui hain
    unscored_leads = [l for l in leads_for_scoring if "Scored" not in l.get('interaction', {}).get('call_status', '')]

    if unscored_leads and st.session_state['bulk_scoring_ran'] == False:
        if st.button(f"▶ Run Agent 5: CALCULATE SCORE ({len(unscored_leads)} New Leads)", type="primary",
                     width='stretch'):
            st.session_state.bulk_scoring_results = []
            progress_bar = st.progress(0, text="Starting bulk scoring...")

            for i, lead in enumerate(unscored_leads, start=1):
                lead_id = lead.get("lead_id")

                # API Call to Agent 5 Backend
                url = f"{MAIN_URL}/api/v1/agent5/run_scoring/{lead_id}"
                ok, resp = safe_post(url)

                # Fetch updated lead for the table
                ok_final, lead_final = safe_get(f"{MAIN_URL}/api/v1/lead/{lead_id}")

                if ok_final:
                    st.session_state.bulk_scoring_results.append({
                        "Lead ID": lead_id,
                        "Name": lead_final['personal'].get('name', 'N/A'),
                        "Final Score": lead_final.get('score', {}).get('current_score', 'N/A'),
                        "Priority Tag": lead_final.get('score', {}).get('priority_tag', 'N/A'),
                        "Sentiment": lead_final.get('analysis', {}).get('sentiment', 'N/A'),
                        "Status": lead_final.get('interaction', {}).get('call_status', 'N/A')
                    })

                progress_bar.progress(i / len(unscored_leads))

            st.session_state['bulk_scoring_ran'] = True
            st.success("🎉 Bulk Scoring Completed!")
            progress_bar.empty()
            st.balloons()
            st.rerun()

    # --- 🟢 UPDATE 2: DISPLAY RESULTS (Current Session + Already Scored) ---
    st.markdown("## 📊 Final Scoring Summary")

    # Table ke liye data prepare karein
    display_data = []
    for lead in leads_for_scoring:
        display_data.append({
            "Lead ID": lead['lead_id'],
            "Name": lead['personal'].get('name', 'N/A'),
            "Final Score": lead.get('score', {}).get('current_score', 'N/A'),
            "Priority Tag": lead.get('score', {}).get('priority_tag', 'N/A'),
            "Sentiment": lead.get('analysis', {}).get('sentiment', 'N/A'),
            "Last Status": lead.get('interaction', {}).get('call_status', 'N/A')
        })

    if display_data:
        df = pd.DataFrame(display_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No scored leads to display yet.")

    st.markdown("---")

    # --- SET TARGET LEAD FOR XAI ---
    if display_data:
        st.markdown("### 🔍 Explainable AI Analysis")

        # Dropdown to select lead for XAI
        lead_ids = [item["Lead ID"] for item in display_data]
        selected_lead_xai = st.selectbox("Select Lead to Explain Score:", lead_ids)

        if st.button("🧠 View XAI Explanation", key="trigger_xai", type="secondary", width='stretch'):
            st.session_state.analysis_target_lead_id = selected_lead_xai
            st.session_state.page = "Agent6XaiPage"
            st.rerun()

    render_footer()


def render_agent6_xai_page():
    if not st.session_state.logged_in:
        st.warning("Login required.")
        return

    st.title("💡  Explainable AI (XAI)")
    st.subheader("Understanding Why Agent 5 Assigned the Final Lead Score")

    selected_id = st.session_state.get("analysis_target_lead_id", "")

    if st.button("⬅ Back to Scoring Results", key="back_xai_to_score"):
        st.session_state.page = "Agent5Page"
        st.rerun()

    st.markdown("---")

    st.markdown("### 🔍 Search or Change Lead")
    manual_id = st.text_input("Enter Lead ID", value=selected_id)

    if st.button("Load Explanation", key="xai_loader"):
        st.session_state.analysis_target_lead_id = manual_id
        st.rerun()

    target_lead_id = st.session_state.get("analysis_target_lead_id")

    if not target_lead_id:
        st.info("Please select a lead from dropdown or search box.")
        return

    st.header(f"Explanation for Lead: **{target_lead_id}**")
    st.markdown("---")

    with st.spinner("Calculating Feature Contributions (SHAP)..."):
        ok, response = safe_get(f"{MAIN_URL}/api/v1/agent5/explain/{target_lead_id}")

    if ok and response.get("success"):
        st.success("Score Explanation Loaded Successfully!")

        col_pred, spacer, col_impact = st.columns([1, 0.5, 2])

        with col_pred:
            st.markdown("### Final Prediction")
            st.metric("Predicted Score", f"{response.get('prediction', 0.0):.3f}")
            st.metric("Priority Tag", response.get("tag", "N/A"))

            # Plotly Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=response["prediction"],
                title={"text": "Lead Score Gauge", "font": {"size": 22}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 40], "color": "red"},
                        {"range": [40, 70], "color": "yellow"},
                        {"range": [70, 100], "color": "green"},
                    ]
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

        with col_impact:
            st.markdown("### 🧠 Top Feature Contributions (SHAP Impact)")
            impacts = response["feature_impacts"]

            impact_rows = []
            for item in impacts[:10]:
                color = "green" if item["impact"] > 0 else "red"
                impact_rows.append({
                    "Feature": item["feature"].replace('.', ' '),
                    "Impact Value": f"<span style='color:{color}; font-weight:bold;'>{item['impact']:+.3f}</span>"
                })

            df_impact = pd.DataFrame(impact_rows)
            st.markdown(df_impact.to_html(escape=False, index=False), unsafe_allow_html=True)
            st.markdown("---")
            st.info("Positive impacts increase score; negative impacts decrease score.")

        # =============================
        # 🎯 SHAP Waterfall Plot Section
        # =============================
        st.markdown("---")
        import shap
        import numpy as np
        import matplotlib.pyplot as plt

        try:
            st.markdown("### 📊 SHAP Waterfall Visualization (Compact Mode)")

            shap_values = np.array(response["shap_values"], dtype=float)
            base_value = float(response["base_value"])
            feature_values = np.array(response["feature_values"], dtype=float)
            feature_names = response["feature_names"]

            exp = shap.Explanation(
                values=shap_values,
                base_values=base_value,
                data=feature_values,
                feature_names=feature_names
            )

            # Create plot
            fig = plt.figure()
            shap.plots.waterfall(exp, max_display=10, show=False)

            # Render smaller using CSS scaling
            st.markdown("""
                <style>
                .shap-plot-scale {
                    transform: scale(0.70);     /* change 0.70 to smaller/larger */
                    transform-origin: top left;
                    margin-bottom: -100px;      /* remove extra whitespace */
                }
                </style>
            """, unsafe_allow_html=True)

            st.markdown('<div class="shap-plot-scale">', unsafe_allow_html=True)
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Waterfall plot unavailable: {e}")

    st.button("↩ Finish and Review Completed Leads",
              on_click=lambda: st.session_state.update(page="CompletedLeadsPage"))
    render_footer()


def render_admin_dashboard():
    st.title("📊 Strategic Analytics Dashboard")
    st.markdown("---")

    # 1. Fetch Data
    leads = fetch_analyzed_leads_from_db(100)
    if not leads:
        st.warning("Data unavailable for analytics.")
        return

    # Data Processing for Charts
    # Yahan hum 'LastActivity' ko process karte waqt hi datetime mein convert kar rahe hain
    df_list = []
    for l in leads:
        last_act_raw = l.get('interaction', {}).get('last_activity')
        try:
            last_act_dt = pd.to_datetime(last_act_raw) if last_act_raw else None
        except:
            last_act_dt = None

        df_list.append({
            "Status": l.get('interaction', {}).get('call_status', 'Unknown'),
            "Location": l.get('personal', {}).get('location', 'Unknown'),
            "Tag": l.get('score', {}).get('priority_tag', 'N/A'),
            "LastActivity": last_act_dt
        })

    df = pd.DataFrame(df_list)

    # --- ROW 1: Key Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Leads", len(leads))
    col2.metric("Hot Leads", len(df[df['Tag'] == 'Hot']))
    col3.metric("Pending Calls", len(df[df['Status'] == 'Pending']))

    # Lead Aging Calculation (Red Flags) - FIXED Logic
    current_time = datetime.now()
    stuck_leads = []
    for l in leads:
        last_act_raw = l.get('interaction', {}).get('last_activity')
        status = l.get('interaction', {}).get('call_status', '')

        if last_act_raw:
            try:
                # String to Datetime conversion fix
                last_act_dt = pd.to_datetime(last_act_raw)
                # Check if stuck > 2 days and not yet fully scored
                if (current_time - last_act_dt).days >= 2 and "Scored" not in status:
                    stuck_leads.append(l)
            except:
                continue

    col4.metric("Red Flags 🚩", len(stuck_leads), delta_color="inverse")

    st.markdown("---")

    # --- ROW 2: Conversion Funnel & Location Analysis ---
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("📉 Conversion Funnel")
        funnel_stages = {
            "Total": len(leads),
            "Interaction Done": len([l for l in leads if l.get('analysis', {}).get('transcript')]),
            "Follow-up Sent": len(df[df['Status'].str.contains("Follow-up", na=False)]),
            "Final Scored": len(df[df['Status'].str.contains("Scored", na=False)])
        }
        fig_funnel = go.Figure(go.Funnel(
            y=list(funnel_stages.keys()),
            x=list(funnel_stages.values()),
            textinfo="value+percent initial",
            marker={"color": ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]}
        ))
        # Updated width to 'stretch'
        st.plotly_chart(fig_funnel)

    with right_col:
        st.subheader("📍 Leads by Location (Hot Focus)")
        hot_df = df[df['Tag'] == 'Hot'].groupby('Location').size().reset_index(name='Counts')
        if not hot_df.empty:
            fig_map = px.bar(hot_df, x='Location', y='Counts', color='Counts',
                             color_continuous_scale='Reds', template="plotly_white")
            st.plotly_chart(fig_map)
        else:
            st.info("No Hot leads to map yet.")

    # --- ROW 3: Lead Aging Alerts ---
    st.markdown("---")
    st.subheader("🚩 Aging Alerts (Stuck > 48 Hours)")
    if stuck_leads:
        for sl in stuck_leads:
            name = sl['personal'].get('name', 'Unknown')
            loc = sl['personal'].get('location', 'Unknown')
            curr_status = sl['interaction'].get('call_status', 'N/A')

            with st.expander(f"⚠️ {name} ({loc})"):
                st.write(f"**Current Status:** {curr_status}")

                # Display date safely
                last_act_val = sl['interaction'].get('last_activity')
                if last_act_val:
                    formatted_date = pd.to_datetime(last_act_val).strftime('%Y-%m-%d %H:%M')
                    st.write(f"**Last Activity:** {formatted_date}")

                st.button(f"Notify Sales Team for {sl['lead_id']}", key=sl['lead_id'], width='stretch')
    else:
        st.success("High efficiency! No leads are stuck in the pipeline.")
        # --- ROW 4: Export Data ---

    st.markdown("---")
    st.subheader("📥 Export Analytics Report")

    if not df.empty:
        # CSV mein convert karein
        csv_data = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Lead Analytics as CSV",
            data=csv_data,
            file_name=f"Lead_Analytics_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            width='stretch'
        )
    else:
        st.info("No data available to export.")


def render_live_monitor_page():
    """
    Renders the dedicated page for monitoring real-time transcription from Bolna.
    """
    st.title("🎙 Live Call & Transcription Monitor")
    st.subheader("Watch the conversation between the AI Agent and the Lead in real-time.")

    # --- Navigation Buttons ---
    col_nav1, col_nav2 = st.columns([0.4, 6])

    with col_nav1:
        st.button("⬅ Back to Dashboard", key="back_to_dash_live",
                  on_click=lambda: st.session_state.update(page='Dashboard'))

    with col_nav2:
        st.button("👁 View Completed Conversations for Analysis", key="view_completed_from_monitor",
                  on_click=lambda: st.session_state.update(page='CompletedLeadsPage'))

    st.markdown("---")

    if 'monitor_lead_id' not in st.session_state:
        st.session_state['monitor_lead_id'] = ""

    lead_id = st.text_input(
        "Enter Lead ID to Monitor (e.g., LMS-101) 👇",
        value=st.session_state.get('monitor_lead_id', 'LMS-101'),
        key="live_monitor_lead_id_input"
    )
    st.session_state['monitor_lead_id'] = lead_id

    if st.button("Start/Refresh Monitoring") and lead_id:

        API_URL_BASE = "f{MAIN_URL}/api/v1/lead/"
        st.warning(f"Monitoring Lead {lead_id}. Status will refresh every 2 seconds.")

        placeholder = st.empty()

        try:
            while st.session_state.page == 'LiveMonitor':
                response = requests.get(f"{API_URL_BASE}{lead_id}", timeout=3)

                if response.status_code == 200:
                    data = response.json()
                    current_status = data.get('interaction', {}).get('call_status', 'N/A')
                    transcript = data.get("analysis", {}).get("live_transcript", [])

                    with placeholder.container():
                        st.markdown(f"Current Call Status: {current_status}")
                        st.markdown("---")

                        # Determine if the call is officially finished based on final statuses
                        is_call_finished = current_status not in ['Calling', 'Ringing', 'in-progress',
                                                                  'In Progress - Live']

                        # Only show waiting message if call is ACTIVE AND transcript is empty
                        if not transcript and not is_call_finished:
                            st.info("Waiting for first interaction chunk...")

                        # Display all chunks in order
                        if transcript:
                            for t in transcript:
                                speaker_role = t['speaker'].lower()
                                avatar = "🤖" if speaker_role == "agent" else "👤"

                                with st.chat_message(role, avatar=avatar):
                                    st.write(t['text'])

                    # Check for call completion to stop polling automatically
                    if is_call_finished:
                        st.success(f"Call finished (Status: {current_status}). Live monitoring stopped.")
                        break

                elif response.status_code == 404:
                    st.error(f"Lead ID {lead_id} not found in the database. Ensure Agent 2 has initiated the call.")
                    break
                else:
                    st.error(f"Error fetching lead data: {response.status_code}")
                    break

                time.sleep(2)

        except requests.exceptions.ConnectionError:
            st.error(
                "Connection Error: Ensure the FastAPI server is running (uvicorn main:app --reload) at {MAIN_URL}")
        except Exception as e:
            st.error(f"An unexpected error occurred during monitoring: {e}")

    render_footer()


if __name__ == '__main__':
    # 1. Render Navigation Bar on every page load
    render_navbar(st.session_state.logged_in)
    if st.session_state.logged_in:
        with st.sidebar:
            st.title("⚙️ Management")
            if st.button("📊 Admin Analytics Dashboard", use_container_width=True):
                st.session_state.page = 'DashboardPage'
                st.rerun()
            st.markdown("---")

    # 2. Page Routing Logic

    if st.session_state.logged_in:
        if st.session_state.page == 'Home' or st.session_state.page == 'Login':
            st.session_state.page = 'Dashboard'
    else:
        if st.session_state.page in ['Dashboard', 'Agent1Page', 'Agent2Page', 'LiveMonitor', 'Agent3Page',
                                     'CompletedLeadsPage', 'Agent4Page', 'Agent5Page', 'Agent6XaiPage',
                                     'DashboardPage']:
            st.session_state.page = 'Home'

    # Render the correct page based on state
    if st.session_state.page == 'Home':
        render_home_page()
    elif st.session_state.page == 'About':
        render_about_page()
    elif st.session_state.page == 'Contact':
        render_contact_page()
    elif st.session_state.page == 'Login':
        render_login_page()
    elif st.session_state.page == 'Dashboard' and st.session_state.logged_in:
        render_dashboard_page()
    elif st.session_state.page == 'Agent1Page' and st.session_state.logged_in:
        render_agent1_page()
    elif st.session_state.page == 'Agent2Page' and st.session_state.logged_in:
        render_agent2_page()
    elif st.session_state.page == 'LiveMonitor' and st.session_state.logged_in:
        render_live_monitor_page()
    elif st.session_state.page == 'CompletedLeadsPage' and st.session_state.logged_in:
        render_completed_reviews_page()
    elif st.session_state.page == 'Agent3Page' and st.session_state.logged_in:
        render_agent3_page()
    elif st.session_state.page == 'Agent4Page' and st.session_state.logged_in:
        render_agent4_followup_page()
    elif st.session_state.page == 'Agent5Page' and st.session_state.logged_in:
        render_agent5_scoring_page()
    elif st.session_state.page == 'Agent6XaiPage' and st.session_state.logged_in:
        render_agent6_xai_page()  # 🟢 NEW ROUTE MAPPING
    elif st.session_state.page == 'DashboardPage' and st.session_state.logged_in:
        render_admin_dashboard()
    else:
        render_home_page()

