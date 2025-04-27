import os
import json
import flask
from flask import Flask, redirect, request, session, url_for, render_template, flash, jsonify
import google.generativeai as genai
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request as GoogleAuthRequest
from googleapiclient.discovery import build
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone 
from googleapiclient.errors import HttpError
import google.auth.exceptions
import requests
import base64
import email
import math
from email.header import decode_header
import time
import gspread
import logging
import pandas as pd
from werkzeug.utils import secure_filename # For potential future file uploads
import dotenv
import uuid
import redis
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, Text
dotenv.load_dotenv()

# Initialize Redis connection (make sure this is defined)
REDIS_URL = os.environ.get("REDIS_URL")
print(f"DEBUG: Connecting to Redis at {REDIS_URL}")
try:
    redis_client = redis.Redis.from_url(REDIS_URL)
    # Test the connection
    redis_client.ping()
    print("DEBUG: Redis connection successful")
except Exception as e:
    print(f"DEBUG: Redis connection failed: {str(e)}")
    # Fallback to a dummy Redis client that logs but doesn't fail
    class DummyRedis:
        def hset(self, key, mapping=None, **kwargs):
            print(f"DUMMY REDIS: Would set {key} to {mapping or kwargs}")
        def hgetall(self, key):
            print(f"DUMMY REDIS: Would get all for {key}")
            return {b'status': b'dummy', b'processed': b'0', b'total': b'100'}
        def ping(self):
            return True
    redis_client = DummyRedis()
    print("DEBUG: Using dummy Redis client")

# --- Configuration ---
# Load from environment variables for security in a web app context
# Users will need to set these before running the app.
# DO NOT HARDCODE THESE HERE IN A REAL APPLICATION
# --- Configuration ---
CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "super-secret-key-for-dev-only")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
# REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0") # Keep if using Redis/RQ

DATABASE_URI = os.environ.get("DATABASE_URL", "sqlite:///./instance/app_data.db")
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "./instance/uploads")
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

# --- Flask App Setup ---
app = Flask(__name__) # Create Flask app instance first
app.secret_key = SECRET_KEY
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true' # Control via env var

# --- IMPORTANT: Ensure Instance/Upload Folders Exist EARLY ---
instance_path = os.path.join(app.instance_path) # Use Flask's built-in instance_path property
upload_path = os.path.join(instance_path, 'uploads') # Define upload path relative to instance path

try:
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)
        logging.info(f"Created instance folder: {instance_path}")
    if not os.path.exists(upload_path):
        os.makedirs(upload_path)
        logging.info(f"Created upload folder: {upload_path}")
except OSError as e:
     # Log error but allow app to continue; SQLAlchemy will likely fail if instance path creation failed
     logging.error(f"Error creating instance or upload folder: {e}")

# --- Configure App Settings AFTER creating folders if paths depend on them ---
# Adjust DB URI to use absolute path to instance folder for robustness
db_file_path = os.path.join(instance_path, 'app_data.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_file_path}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = upload_path # Use the defined absolute upload path
# app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS # This isn't a standard Flask config, handle in code

# --- Initialize Extensions AFTER app config is set ---
db = SQLAlchemy(app)

@app.context_processor
def inject_now():
    """Injects the current UTC datetime into the template context."""
    return {'now': datetime.now(timezone.utc)} # Use timezone.utc for consistency


# --- Google OAuth Configuration ---
# This file is downloaded from Google Cloud Console (OAuth Client ID for Web application)
# It's better practice to load CLIENT_ID/SECRET from env vars directly,
# but using the file path is also common for Flask examples.
# Ensure this file exists and is configured correctly.
CLIENT_SECRETS_FILE = 'credentials_web_2.json' # Needs client_id, client_secret, redirect_uris etc.

# Scopes required by the application
SCOPES = [
    'openid', # Basic profile info
    'https://www.googleapis.com/auth/userinfo.email', # Email address
    'https://www.googleapis.com/auth/userinfo.profile', # Profile info
    'https://www.googleapis.com/auth/gmail.readonly', # Read emails
    'https://www.googleapis.com/auth/spreadsheets', # Read/write sheets
    'https://www.googleapis.com/auth/drive.file' # Create/find sheets in Drive
]
# Redirect URI specified in Google Cloud Console for your Web Application credentials
# Must match exactly! For local dev, often http://127.0.0.1:5000/callback
REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "http://127.0.0.1:5000/callback")

# --- Database Model ---
class UserProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    google_id = db.Column(db.String(100), unique=True, nullable=False)
    name = db.Column(db.String(150))
    email = db.Column(db.String(150))
    phone = db.Column(db.String(50))
    address = db.Column(db.String(300))
    linkedin_url = db.Column(db.String(255))
    portfolio_url = db.Column(db.String(255))
    resume_text = db.Column(db.Text) # For pasted resume content
    resume_filename = db.Column(db.String(255))
    resume_filepath = db.Column(db.String(512))
    extracted_resume_json = db.Column(Text) # Store extracted data as JSON string in Text field
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<UserProfile {self.email or self.google_id}>'

with app.app_context():
    try:
        inspector = inspect(db.engine)
        if not inspector.has_table(UserProfile.__tablename__):
            logging.info(f"Creating database table '{UserProfile.__tablename__}'...")
            db.create_all()
            logging.info("Database table created.")
        else:
            logging.info(f"Database table '{UserProfile.__tablename__}' already exists.")
    except Exception as e:
        logging.error(f"Error during database table check/creation: {e}")
# --- Helper Functions ---

def call_serpapi_google_jobs(query, location="None", start=0):
    """Calls the SerpApi Google Jobs endpoint."""
    if not SERPAPI_API_KEY:
        logging.error("SERPAPI_API_KEY environment variable not set.")
        return {"error": "Job search API key is not configured."}

    base_url = "https://serpapi.com/search"
    params = {
        "engine": "google_jobs",
        "q": query,
        "hl": "en",
        "gl": "us",
        "location": location,
        # "start": start, # For pagination
        "api_key": SERPAPI_API_KEY
    }

    try:
        logging.info(f"Calling SerpApi Google Jobs: q='{query}', location='{location}', start={start}")
        response = requests.get(base_url, params=params, timeout=20) # Add timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        logging.info(f"SerpApi returned {len(data.get('jobs_results', []))} jobs.")
        return data
    except requests.exceptions.RequestException as e:
        logging.error(f"SerpApi request failed: {e}")
        return {"error": f"Failed to connect to job search service: {e}"}
    except json.JSONDecodeError:
        logging.error(f"Failed to decode SerpApi JSON response. Status: {response.status_code}, Body: {response.text[:200]}...")
        return {"error": "Received invalid response from job search service."}
    except Exception as e:
        logging.error(f"An unexpected error occurred during SerpApi call: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while searching for jobs: {e}"}

def parse_serpapi_jobs(api_response):
    """Parses the jobs_results from the SerpApi response into a standard format."""
    if "error" in api_response:
        logging.warning(f"Cannot parse jobs, API response contained an error: {api_response['error']}")
        return [] # Return empty list on error

    jobs_results = api_response.get("jobs_results", [])
    parsed_jobs = []

    for job in jobs_results:
        # Extract relevant fields, providing defaults if keys are missing
        title = job.get("title", "N/A")
        company = job.get("company_name", "N/A")
        location = job.get("location", "N/A")
        description = job.get("description", "")
        # Try to find a direct link, fallback to google jobs link if needed
        job_link = "#" # Default placeholder
        related_links = job.get('related_links', [])
        if related_links and isinstance(related_links, list) and len(related_links) > 0:
             job_link = related_links[0].get('link', '#') # Take the first related link if available
        elif job.get('job_id'): # Fallback to constructing a google jobs link maybe? Requires more info.
             # For simplicity, we'll just use '#' if no direct link found
             pass

        # Add highlights if available
        highlights = job.get('job_highlights', [])
        if highlights and isinstance(highlights, list):
            highlight_text = "\n".join([h.get('title', '') + (": " + ", ".join(h.get('items',[])) if h.get('items') else "") for h in highlights if h.get('title')])
            if highlight_text:
                description += "\n\nHighlights:\n" + highlight_text

        parsed_jobs.append({
            'id': job.get("job_id", f"serp_{uuid.uuid4()}"), # Use SerpApi job_id or generate one
            'title': title,
            'company': company,
            'location': location,
            'description': description.strip(),
            'url': job_link, # Use the extracted link
            'source': job.get('via', 'Google Jobs') # Indicate the source
        })

    return parsed_jobs

def extract_resume_data_with_gemini(resume_text):
    """
    Uses Gemini to extract structured data (skills, experience, education) from resume text.

    Args:
        resume_text: The raw text content of the resume.

    Returns:
        A JSON string containing the extracted data, or None if an error occurs or text is empty.
    """
    if not resume_text or not resume_text.strip():
        logging.info("Resume text is empty, skipping Gemini processing.")
        return None

    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        logging.error("Gemini API Key is missing for resume processing.")
        # Return None or raise an error depending on desired handling
        return None

    # Define the desired JSON structure
    # Be very specific about the format you want Gemini to return.
    prompt = f"""
    Analyze the following resume text and extract key information into a structured JSON format.

    **Instructions:**
    1.  Parse the text to identify sections like Contact Info, Summary/Objective, Work Experience, Education, Skills, Projects, etc.
    2.  Extract relevant details for each section.
    3.  Format the output STRICTLY as a single JSON object containing the following keys:
        * `summary`: (String) The summary or objective statement, if present. Null if not found.
        * `skills`: (List of Strings) A list of identified skills (technical, soft, languages). Empty list if none found.
        * `work_experience`: (List of Objects) Each object should have:
            * `company`: (String) Company name.
            * `job_title`: (String) Job title.
            * `dates`: (String) Employment dates (e.g., "Jan 2020 - Present", "2019-2021").
            * `description`: (String) Responsibilities and achievements (combine bullet points into a single string with newlines '\\n').
        * `education`: (List of Objects) Each object should have:
            * `institution`: (String) Name of the school/university.
            * `degree`: (String) Degree obtained (e.g., "B.S. Computer Science").
            * `dates`: (String) Attendance dates or graduation date.
            * `details`: (String, Optional) Any relevant details like GPA, honors, relevant coursework.
        * `projects`: (List of Objects, Optional) Each object should have:
            * `name`: (String) Project name.
            * `description`: (String) Brief description of the project.
            * `technologies`: (List of Strings, Optional) Technologies used.

    4.  If a section (like projects) is not found, omit the key or set its value to an empty list/null as appropriate based on the schema defined above.
    5.  Ensure the output is a single, valid JSON object and nothing else. Do not include explanations or markdown formatting.

    **Resume Text:**
    ---
    {resume_text[:15000]}
    ---

    **JSON Output:**
    """

    try:
        genai.configure(api_key=gemini_api_key)
        # Use a model suitable for structured data extraction, like Pro or potentially Flash
        # Consider Gemini 1.5 Pro if available and needing larger context or better JSON adherence
        model = genai.GenerativeModel('gemini-1.5-flash') # Or 'gemini-pro'

        logging.info("Calling Gemini API to process resume text...")
        response = model.generate_content(
            prompt,
            # Enforce JSON output if the model/API supports it reliably
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )

        # Attempt to parse the JSON response directly
        response_text = response.text.strip()
        logging.debug(f"Raw Gemini Resume Response: {response_text[:200]}...") # Log beginning of response

        # Basic cleanup (sometimes models wrap in markdown even with mime type set)
        if response_text.startswith("```json"):
            response_text = response_text[len("```json"):].strip()
        if response_text.endswith("```"):
            response_text = response_text[:-len("```")].strip()

        # Validate that it's parsable JSON
        parsed_json = json.loads(response_text)

        # Return the valid JSON string
        logging.info("Successfully processed resume text with Gemini.")
        return json.dumps(parsed_json, indent=2) # Store pretty-printed JSON string

    except json.JSONDecodeError as json_err:
        logging.error(f"Failed to parse JSON response from Gemini for resume: {json_err}")
        logging.error(f"Gemini Raw Response causing JSON error: {response.text}")
        return json.dumps({"error": "Failed to parse Gemini response as JSON.", "raw_response": response.text})
    except Exception as e:
        # Catch other potential errors (API connection, rate limits, etc.)
        logging.error(f"Error calling Gemini API for resume processing: {e}", exc_info=True)
        # Store an error indicator in the JSON
        return json.dumps({"error": f"Gemini API call failed: {str(e)}"})


def get_google_credentials():
    """Gets Google credentials from the session.
        Testing if the Credentials class can handle its own serialized format directly.
    """
    if 'credentials' not in session:
        print("DEBUG: No 'credentials' found in session.")
        return None

    try:
        # Load the dictionary directly from the JSON string stored in the session
        stored_credentials_dict = json.loads(session['credentials'])
        print(f"DEBUG: Loaded credentials dict from session: {stored_credentials_dict}") # Debugging

        # --- ALTERNATIVE FIX: Remove explicit datetime conversion ---
        # Let the Credentials class constructor handle the dictionary directly,
        # including the 'expiry' string as it was saved by to_json().
        # The library might be better equipped to handle its own format.

        # Check the type before creating Credentials object
        expiry_value = stored_credentials_dict.get('expiry')

        print(f"DEBUG: Type of expiry loaded from JSON: {type(expiry_value)}")
        print(f"DEBUG: Value of expiry loaded from JSON: {expiry_value}")


        # Create Credentials object from the dictionary
        # Ensure all necessary fields are present for Credentials object creation
        required_keys = ['token', 'refresh_token', 'token_uri', 'client_id', 'client_secret', 'scopes']
        # Note: expiry is handled above, refresh_token might be None initially but should exist if offline access was granted
        if all(key in stored_credentials_dict for key in required_keys if key != 'refresh_token') and \
            stored_credentials_dict.get('token'): # Ensure token exists
                print("DEBUG: Creating Credentials object directly from loaded dict...")
                # Pass the dictionary directly, assuming Credentials can handle the expiry string format
                creds = Credentials(**stored_credentials_dict)
                print("DEBUG: Credentials object created.")
                # Verify expiry type *after* object creation by the library
                if hasattr(creds, 'expiry'):
                    creds.expiry = datetime.fromisoformat(creds.expiry.replace("Z", "+00:00"))
                    print(f"DEBUG: Type of creds.expiry after creation: {type(creds.expiry)}")
                    if isinstance(creds.expiry, datetime):
                        print(f"DEBUG: Timezone info of creds.expiry: {creds.expiry.tzinfo}")
                        creds.expiry = creds.expiry.replace(tzinfo=None)
                return creds
        else:
            print("Warning: Incomplete credentials data found in session before creating Credentials object.")
            session.pop('credentials', None) # Clear incomplete data
            return None

    except json.JSONDecodeError:
        print("Error: Could not decode credentials from session.")
        session.pop('credentials', None) # Clear potentially corrupt data
        return None
    except TypeError as te:
            # Catch potential TypeError during Credentials creation if format is wrong
            print(f"TypeError during Credentials object creation: {te}")
            print("DEBUG: Credentials dictionary causing error:", stored_credentials_dict)
            session.pop('credentials', None) # Clear potentially problematic data
            return None
    except Exception as e:
        print(f"Error loading/creating credentials from session: {e}")
        session.pop('credentials', None)
        return None

def save_google_credentials(credentials):
    """Saves Google credentials to the session."""
    session['credentials'] = credentials.to_json()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def build_google_service(service_name, version):
    """Builds and returns a Google API service."""
    try:
        print(f"DEBUG: Building {service_name} service v{version}")
        credentials = get_google_credentials()
        if not credentials:
            print(f"DEBUG: No valid credentials for {service_name}")
            return None
            
        # Print token info for debugging
        print(f"DEBUG: Token valid: {credentials.valid}")
        print(f"DEBUG: Token expired: {credentials.expired}")
        
        # Force token refresh if expired
        if credentials.expired:
            print("DEBUG: Refreshing expired token")
            credentials.refresh(GoogleAuthRequest())
            
        try:
            service = build(service_name, version, credentials=credentials)
            return service
        except Exception as e:
            print(f"DEBUG: Error building {service_name} service: {str(e)}")
            # Handle specific auth errors if possible
            if isinstance(e, HttpError) and e.resp.status in [401, 403]:
                flash("Authentication error accessing Google Services. Please log in again.", "danger")
                session.clear() # Clear session on auth failure
            return None
    
    except Exception as e:
        print(f"DEBUG: Error building {service_name} service: {str(e)}")
        return None
    
def generate_job_titles_with_gemini(keywords):
    """
    Uses Gemini to suggest relevant job titles based on input keywords.

    Args:
        keywords: A list of strings (skills, technologies, etc.).

    Returns:
        A list of suggested job title strings, or an empty list if error/no results.
    """
    if not keywords:
        return []

    if not GEMINI_API_KEY:
        logging.error("Gemini API Key is missing for job title generation.")
        return []

    # Limit keywords sent to Gemini
    keywords_str = ", ".join(keywords[:15]) # Send up to 15 keywords

    prompt = f"""
    Analyze the following list of skills, technologies, and potential keywords extracted from a user's resume:
    Keywords: {keywords_str}

    Based ONLY on these keywords, suggest 2-3 common and relevant job titles that someone with these skills might search for.
    Focus on standard industry titles suitable for querying a job search engine (like Google Jobs).
    Avoid overly niche or extremely specific titles unless strongly implied by multiple keywords.

    Output ONLY a comma-separated list of the suggested job titles. Do not include explanations, numbering, or any other text.

    Example Output:
    Software Engineer, Backend Developer, Python Developer

    Output:
    """

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash') # Use Flash for this task

        logging.info(f"Calling Gemini to generate job titles from keywords: {keywords_str}")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.2) # Slightly creative but focused
        )

        response_text = response.text.strip()
        logging.info(f"Gemini suggested titles raw response: {response_text}")

        # Simple parsing: split by comma, strip whitespace
        suggested_titles = [title.strip() for title in response_text.split(',') if title.strip()]

        # Basic validation/cleanup (optional)
        valid_titles = [t for t in suggested_titles if len(t) > 3 and len(t) < 50] # Filter out very short/long titles

        logging.info(f"Parsed suggested job titles: {valid_titles}")
        return valid_titles[:3] # Return max 3 titles

    except Exception as e:
        logging.error(f"Error calling Gemini API for job title generation: {e}", exc_info=True)
        return []

def decode_email_part(part):
    """Decodes email subject or sender."""
    if part is None: return ""
    decoded_parts = []
    try:
        for text, encoding in decode_header(str(part)):
            if isinstance(text, bytes):
                charset = encoding if encoding else 'utf-8'
                try: decoded_parts.append(text.decode(charset, errors='replace'))
                except LookupError: decoded_parts.append(text.decode('utf-8', errors='replace'))
            elif isinstance(text, str): decoded_parts.append(text)
        return ''.join(decoded_parts)
    except Exception: return str(part) # Fallback

def get_email_body(message_payload):
    """Extracts plain text body from Gmail API message payload."""
    body = ""
    if 'parts' in message_payload:
        for part in message_payload['parts']:
            if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                body_data = part['body']['data']
                body = base64.urlsafe_b64decode(body_data).decode('utf-8', errors='replace')
                return body.strip()
            # Recursive call for multipart/alternative or multipart/related
            elif part['mimeType'].startswith('multipart/'):
                nested_body = get_email_body(part)
                if nested_body:
                    return nested_body # Return first plain text found
    elif message_payload.get('mimeType') == 'text/plain' and 'body' in message_payload and 'data' in message_payload['body']:
         body_data = message_payload['body']['data']
         body = base64.urlsafe_b64decode(body_data).decode('utf-8', errors='replace')

    return body.strip()

def extract_job_info_with_gemini(gemini_key, subject, body, date_received):
    """Uses Gemini to extract job application details."""
    if not gemini_key:
        return {"Status": "Error: Gemini API Key Missing"}
    if not body and not subject:
        return None

    truncated_body = body if body else ""

    # Use the same robust prompt as before
    prompt = f"""
        **Objective:** Analyze the provided email content (Subject and Body) related to a job application. Your goal is to accurately extract the Company Name, Job Title, and classify the email's status based on its primary purpose.

        **Instructions:**

        1.  **Read Thoroughly:** Carefully examine the *entire* provided email content, including the Subject line and the full Body text. Pay attention to keywords, phrases, and the overall context.
        2.  **Reason Step-by-Step (Internal Thought Process):**
            * First, look for explicit mentions of a company name. Prioritize the company sending the email or the company the application is clearly directed towards.
            * Second, scan for mentions of a specific job title related to the application.
            * Third, analyze the email's core message to determine its status. Look for specific keywords associated with each status category. If multiple keywords/themes are present (e.g., rejection *and* keep-in-touch language), apply the prioritization logic defined in the Status Categories section below.
        3.  **Extract Information:**
            * **Company Name:** Identify the primary company name directly mentioned or strongly implied in the email. If multiple are mentioned (e.g., a parent company and a subsidiary), choose the one most relevant to the application context (usually the sender or the one the job is with). If no company name can be reasonably identified, state "Not Mentioned".
            * **Job Title:** Identify the specific job title being discussed for the application. Include any clarifying details if present (e.g., "Software Engineer (L4)"). If no specific job title is mentioned, or if it refers only generally to "the position" or "your application" without naming the role, state "Not Mentioned".
            * **Status:** Classify the email's main purpose using **EXACTLY ONE** category from the list below. **Prioritize based on keywords and the most definitive action/decision conveyed.** The order below implies a general priority (e.g., a clear rejection overrides simple acknowledgment phrases):
                * **Rejection**: Choose if keywords like "regret", "unfortunately", "not selected", "other candidates", "will not be moving forward", "position has been filled", "unable to offer you the position" are present and represent the core message.
                * **Offer**: Choose if keywords like "offer", "employment offer", "job offer", "compensation", "salary", "benefits", "start date", "welcome aboard", "joining us", "extend an offer" are present and form the main purpose.
                * **Interview Request**: Choose if keywords like "interview", "schedule time", "schedule a call", "speak with", "talk further", "next steps involve a call/meeting", "discussion with the team" clearly indicate a request to schedule or conduct an interview.
                * **Assessment Request**: Choose if keywords like "assessment", "test", "coding challenge", "technical screen", "assignment", "Hackerrank", "Codility", "online assessment" indicate the next step is a test or skills evaluation.
                * **Keep In Touch / Future Consideration**: Choose ONLY if the *primary message* is about keeping the application/resume on file for *future* roles, often in the absence of strong rejection keywords for the *current* role (e.g., "keep your resume on file", "consider you for future opportunities", "reach out if a suitable role opens"). If clear rejection language for the *current* role is present, choose 'Rejection' instead, even if future consideration is mentioned secondarily.
                * **Application Acknowledgment**: Choose ONLY if the email *solely* confirms receipt of the application and indicates it's under review, without providing any further status update (e.g., "received your application", "thank you for applying", "application is under review", "will be in touch if"). If other status keywords are present, prioritize those categories.
                * **Informational / General Company Email**: Choose if the email is clearly a newsletter, marketing communication, company update, job alert notification (for *new* jobs, not a status update on an *existing* application), or other general communication not tied to the status of a specific, active application process.
                * **Other**: Choose if the email's purpose related to the application process is clear but does not fit neatly into any of the above categories (e.g., a request for more information/documents *from* the applicant, notification of a delay in the process, system error message).
                * **Unable to Determine**: Choose ONLY if the email content is extremely ambiguous, lacks sufficient context, or is corrupted/incomplete, making it impossible to confidently determine the company, title, or status.

        4.  **Output Format:**
            * Produce **ONLY** the requested information.
            * Do **NOT** include any introductory phrases, explanations, confidence scores, or the reasoning process in the final output.
            * Use the following exact format, replacing bracketed placeholders with the extracted information:

            ```text
            Company Name: [Extracted Company Name or "Not Mentioned"]
            Job Title: [Extracted Job Title or "Not Mentioned"]
            Status: [Chosen Status Category]
            ```

        --- Email Content ---
        Date Received: {date_received}
        Subject: {subject}
        Body:
        {truncated_body}
        --- End Email Content ---

        Output:
    """
    try:
        genai.configure(api_key=gemini_key)
        # Ensure a valid model name is used
        gemini_model = genai.GenerativeModel('gemini-2.0-flash') # Or 'gemini-pro'
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.1)
        )
        extracted_data = {"Company Name": "Parsing Error", "Job Title": "Parsing Error", "Status": "Parsing Error"}
        lines = response.text.strip().split('\n')
        found_keys = set()
        for line in lines:
            if ':' in line:
                 key, value = line.split(':', 1)
                 key = key.strip(); value = value.strip()
                 if key in extracted_data:
                     extracted_data[key] = value
                     found_keys.add(key)
        if len(found_keys) != len(extracted_data):
             extracted_data["Status"] = "LLM Format Error"
        return extracted_data
    except Exception as e:
        print(f"Error calling Gemini API: {e}") # Log error server-side
        # Provide a user-friendly error status
        error_status = "LLM Error"
        if "API key not valid" in str(e): error_status = "LLM Error: Invalid API Key"
        elif "API_KEY_SERVICE_BLOCKED" in str(e): error_status = "LLM Error: API Blocked"
        elif "RESOURCE_EXHAUSTED" in str(e): error_status = "LLM Error: Quota Exceeded"
        return {"Company Name": "LLM Error", "Job Title": "LLM Error", "Status": error_status}

# --- Flask Routes ---

@app.route('/')
def index():
    """Home page: Shows login status and form to start processing."""
    user_info = session.get('user_info')
    gemini_key_set = GEMINI_API_KEY in session
    return render_template('index.html', user_info=user_info, gemini_key_set=gemini_key_set, now=datetime.now())

@app.route('/login')
def login():
    """Initiates the Google OAuth 2.0 login flow."""
    # Ensure client_secrets.json exists
    if not os.path.exists(CLIENT_SECRETS_FILE):
         flash("Configuration error: Client secrets file not found.", "danger")
         return redirect(url_for('index'))

    try:
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE, scopes=SCOPES, redirect_uri=REDIRECT_URI)

        # Generate the authorization URL and store the state
        authorization_url, state = flow.authorization_url(
            access_type='offline', # Request refresh token
            include_granted_scopes='true',
            prompt='consent' # Force consent screen for refresh token
        )
        session['oauth_state'] = state # Store state to prevent CSRF
        return redirect(authorization_url)
    except Exception as e:
        flash(f"Error starting authentication flow: {e}", "danger")
        return redirect(url_for('index'))


@app.route('/callback')
def callback():
    """Handles the redirect from Google after user authorization."""
    # Verify the state parameter to prevent CSRF

    state = session.pop('oauth_state', None)
    if state is None or state != request.args.get('state'):
        flash('Invalid state parameter. Authentication failed.', 'danger')
        return redirect(url_for('index'))

    # Ensure client_secrets.json exists
    if not os.path.exists(CLIENT_SECRETS_FILE):
         flash("Configuration error: Client secrets file not found during callback.", "danger")
         return redirect(url_for('index'))

    try:
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE, scopes=SCOPES, redirect_uri=REDIRECT_URI)

        # Exchange the authorization code for credentials (access & refresh tokens)
        flow.fetch_token(authorization_response=request.url)
        credentials = flow.credentials
        save_google_credentials(credentials) # Save credentials in session

        # Get user info
        try:
            user_info_service = build('oauth2', 'v2', credentials=credentials)
            user_info = user_info_service.userinfo().get().execute()
            # --- Store Google ID in session ---
            session['user_info'] = {
                'email': user_info.get('email'),
                'name': user_info.get('name'),
                'picture': user_info.get('picture'),
                'google_id': user_info.get('id') # IMPORTANT: Store the unique Google ID
            }
            flash(f"Successfully logged in as {user_info.get('email', 'Unknown')}", "success")
        except HttpError as e:
            flash(f"Could not fetch user info: {e}", "warning")
            session['user_info'] = {'email': 'Error fetching email'} # Store placeholder

        return redirect(url_for('index'))

    except Exception as e:
        flash(f"Error during authentication callback: {e}", "danger")
        return redirect(url_for('index'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_info' not in session or not session['user_info'].get('google_id'):
        flash("Please log in to view or edit your profile.", "warning")
        return redirect(url_for('login'))

    google_id = session['user_info']['google_id']
    user_profile = UserProfile.query.filter_by(google_id=google_id).first()

    if request.method == 'POST':
        # --- Process form submission ---
        try:
            if not user_profile:
                # Create new profile if one doesn't exist
                user_profile = UserProfile(google_id=google_id, email=session['user_info'].get('email'))
                db.session.add(user_profile)

            # Update fields from form
            user_profile.name = request.form.get('name', user_profile.name)
            user_profile.phone = request.form.get('phone', user_profile.phone)
            user_profile.address = request.form.get('address', user_profile.address)
            user_profile.linkedin_url = request.form.get('linkedin_url', user_profile.linkedin_url)
            user_profile.portfolio_url = request.form.get('portfolio_url', user_profile.portfolio_url)
            user_profile.resume_text = request.form.get('resume_text', user_profile.resume_text)
            # Ensure email from session is stored if profile was just created
            if not user_profile.email:
                 user_profile.email = session['user_info'].get('email')

            # --- Process Resume Text ---
            new_resume_text = request.form.get('resume_text')
            # Check if text has actually changed or if it's the first time saving text
            process_resume = (new_resume_text and new_resume_text != user_profile.resume_text) or \
                             (new_resume_text and not user_profile.extracted_resume_json)

            user_profile.resume_text = new_resume_text # Update the raw text field

            if process_resume:
                flash("Processing resume text with AI... This may take a moment.", "info")
                # Call Gemini function - This might take a few seconds!
                extracted_data_json = extract_resume_data_with_gemini(new_resume_text)
                if extracted_data_json:
                    user_profile.extracted_resume_json = extracted_data_json
                    logging.info(f"Stored extracted resume JSON for user {google_id}")
                else:
                    # Handle case where Gemini processing failed or returned None
                    logging.warning(f"Gemini resume processing returned no data for user {google_id}")
                    # Optionally clear old data or keep it? Let's clear it if processing was attempted.
                    user_profile.extracted_resume_json = json.dumps({"error": "Resume processing failed or text was empty."})
            elif not new_resume_text:
                 # If resume text was cleared, clear the extracted data too
                 user_profile.extracted_resume_json = None

            # --- Handle File Upload ---
            
            resume_file = request.files.get('resume_file')
            if resume_file and resume_file.filename != '':
                if allowed_file(resume_file.filename):
                    filename = secure_filename(f"{google_id}_{resume_file.filename}") # Add google_id prefix for uniqueness
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                    # --- Optional: Delete old file if replacing ---
                    if user_profile.resume_filepath and os.path.exists(user_profile.resume_filepath):
                         try:
                             os.remove(user_profile.resume_filepath)
                             logging.info(f"Deleted old resume file: {user_profile.resume_filepath}")
                         except OSError as e:
                             logging.warning(f"Could not delete old resume file {user_profile.resume_filepath}: {e}")

                    # --- Save new file ---
                    resume_file.save(filepath)
                    user_profile.resume_filename = resume_file.filename # Store original name
                    user_profile.resume_filepath = filepath # Store full path
                    logging.info(f"Saved new resume file: {filepath} for user {google_id}")
                else:
                    flash("Invalid file type for resume. Allowed types: txt, pdf, doc, docx", "warning")
            elif request.form.get('delete_resume') == 'true':
                 # --- Handle Deletion Request ---
                 if user_profile.resume_filepath and os.path.exists(user_profile.resume_filepath):
                      try:
                          os.remove(user_profile.resume_filepath)
                          logging.info(f"Deleted resume file via request: {user_profile.resume_filepath}")
                          user_profile.resume_filename = None
                          user_profile.resume_filepath = None
                      except OSError as e:
                          logging.warning(f"Could not delete resume file {user_profile.resume_filepath}: {e}")
                          flash("Could not delete the stored resume file.", "danger")
                 else:
                      # Clear DB fields even if file didn't exist
                      user_profile.resume_filename = None
                      user_profile.resume_filepath = None



            db.session.commit()
            flash("Profile updated successfully!", "success")
            # Stay on profile page after saving
            return redirect(url_for('profile'))

        except Exception as e:
            db.session.rollback() # Rollback changes on error
            logging.error(f"Error updating profile for {google_id}: {e}")
            flash(f"An error occurred while updating your profile: {e}", "danger")

    extracted_data = None
    if user_profile and user_profile.extracted_resume_json:
        try:
            extracted_data = json.loads(user_profile.extracted_resume_json)
        except json.JSONDecodeError:
            logging.warning(f"Could not parse stored JSON for user {google_id}")
            extracted_data = {"error": "Stored resume data is corrupted."} # Provide error feedback

    # --- GET Request: Render the form ---
    # If profile doesn't exist yet, pass an empty object or None
    if not user_profile:
         # Pre-populate email and name from Google session info if available
         user_profile = UserProfile(
             google_id=google_id,
             email=session['user_info'].get('email'),
             name=session['user_info'].get('name')
         )

    return render_template('profile.html', profile=user_profile, user_info=session.get('user_info'), now=datetime.now(), extracted_data=extracted_data) # Pass parsed data to template

@app.route('/jobs')
def jobs():
    """Displays recommended and general job postings using SerpApi."""
    if 'user_info' not in session or not session['user_info'].get('google_id'):
        flash("Please log in to view job recommendations.", "warning")
        return redirect(url_for('login'))

    if not SERPAPI_API_KEY:
         flash("Job search functionality is currently unavailable (API key missing).", "warning")
         return render_template('jobs.html', recommended_jobs=[], general_jobs=[], user_info=session.get('user_info'), api_error=True)

    try:
        page = int(request.args.get('page', 1))
    except ValueError:
        page = 1
    if page < 1: page = 1
    results_per_page = 6 # How many results to SHOW per page


    google_id = session['user_info']['google_id']
    user_profile = UserProfile.query.filter_by(google_id=google_id).first()

    recommended_jobs_all = []
    recommended_jobs_page = []
    general_jobs_all = [] # Store all fetched general jobs here
    general_jobs_page = [] # Store the slice for the current page
    profile_keywords = []
    user_location = "United States" # Default location

    if user_profile:
        # Get location
        if user_profile.address:
             parts = user_profile.address.split(',')
             if len(parts) >= 2:
                 potential_loc = f"{parts[-2].strip()}, {parts[-1].strip()}"
                 if potential_loc: user_location = potential_loc

        # Extract and filter keywords
        if user_profile.extracted_resume_json:
            try:
                extracted_data = json.loads(user_profile.extracted_resume_json)
                skills = extracted_data.get('skills', [])
                titles = [exp.get('job_title', '') for exp in extracted_data.get('work_experience', [])]

                # Filter keywords: prioritize skills, check length, avoid overly generic/problematic terms
                potential_keywords = list(set(
                    [s.strip() for s in skills if s and isinstance(s, str)] +
                    [t.strip() for t in titles if t and isinstance(t, str)]
                ))

                # --- Keyword Filtering Logic ---
                filtered_keywords = []
                # Avoid very short terms unless known acronyms/langs (customize this list)
                allowed_short_terms = {'ai', 'ml', 'go', 'aws', 'api', 'qa', 'ux', 'ui', 'c#', 'f#', 'r'}
                ignore_terms = {'and', 'or', 'the', 'for', 'with', 'llc', 'inc', 'corp'} # Example ignore list

                for k in potential_keywords:
                    k_lower = k.lower()
                    # Check length or if it's an allowed short term
                    if len(k) >= 3 or k_lower in allowed_short_terms:
                         # Check against ignore list
                        if k_lower not in ignore_terms:
                            # Avoid terms that are likely just parts of titles or too abstract for job search
                            if 'engineer' not in k_lower and 'developer' not in k_lower and 'manager' not in k_lower and 'specialist' not in k_lower and 'completeness' not in k_lower:
                                filtered_keywords.append(k) # Keep original casing for quoting if needed

                if filtered_keywords:
                    profile_keywords = filtered_keywords
                    suggested_titles = generate_job_titles_with_gemini(profile_keywords)

                    logging.info(f"Using FILTERED keywords from profile: {profile_keywords}")
                else:
                     logging.info("No suitable keywords found after filtering profile JSON.")

            except json.JSONDecodeError:
                logging.warning(f"Could not parse extracted_resume_json for user {google_id}")
            except Exception as e:
                 logging.error(f"Error processing profile data for keywords: {e}")
        else:
             logging.info("No extracted resume JSON found for keywords.")

    # --- Prepare Search Queries ---
    if not profile_keywords:
        profile_keywords = ["software engineer", "python developer"] # More targeted fallback
        flash("No specific skills found or extracted from your profile. Showing general software jobs. Update your profile's resume text for better recommendations.", "info")

    # Create query string: Quote phrases/multi-word terms, limit count
    # Use only the first 5-7 filtered keywords
    # Limit to 6 keywords to avoid overly complex queries
    query_terms = []
    for k in profile_keywords[:6]:
        if ' ' in k:
            query_terms.append(f'"{k}"') # Quote multi-word terms
        else:
            query_terms.append(k)
   
    general_query = "All Jobs" # Simpler general query

    # --- Prepare Search Queries ---
    recommended_query = ""
    if suggested_titles:
        # Use titles generated by Gemini
        recommended_query = " OR ".join(f'"{t}"' for t in suggested_titles) # Quote titles
        logging.info(f"Using Gemini suggested titles for query: '{recommended_query}'")
    elif profile_keywords:
        # Fallback 1: Use filtered keywords if title generation failed/empty
        query_terms = []
        for k in profile_keywords[:6]: # Limit to 6 keywords
            if ' ' in k: query_terms.append(f'"{k}"')
            else: query_terms.append(k)
        recommended_query = " OR ".join(query_terms)
        logging.info(f"Using filtered keywords for query (Gemini fallback): '{recommended_query}'")
        flash("Could not generate specific job titles from profile, using keywords instead.", "info")
    else:
        # Fallback 2: Use generic default if no keywords found at all
        recommended_query = "software engineer OR python developer"
        logging.info(f"Using default keywords for query: '{recommended_query}'")
        flash("No specific skills found in profile. Showing general software jobs. Update your profile for better recommendations.", "info")
    
    # --- Call SerpApi ---
    logging.info(f"Attempting recommended search with query: '{recommended_query}' and location: '{user_location}'")
    recommended_response = call_serpapi_google_jobs(recommended_query, user_location)
    recommended_jobs_all = parse_serpapi_jobs(recommended_response)
    print(f"DEBUG: number of recommended jobs: {len(recommended_jobs_all)}")
    if "error" in recommended_response:
         flash(f"Could not fetch recommended jobs: {recommended_response['error']}", "warning")
         logging.warning(f"SerpApi error for recommended jobs: {recommended_response['error']}")

    else:
        # --- Perform Pagination on the Fetched Results ---
        total_fetched_results = len(recommended_jobs_all)
        logging.info(f"Total recommended jobs fetched for pagination: {total_fetched_results}")

        if total_fetched_results > 0:
            start_slice_index = (page - 1) * results_per_page
            end_slice_index = start_slice_index + results_per_page
            recommended_jobs_page = recommended_jobs_all[start_slice_index:end_slice_index] # Slice the list

            total_pages = math.ceil(total_fetched_results / results_per_page)

            recommended_pagination_info = {
                "current_page": page,
                "total_pages": total_pages,
                "has_prev": page > 1,
                "has_next": page < total_pages,
                "prev_num": page - 1 if page > 1 else None,
                "next_num": page + 1 if page < total_pages else None,
                "total_results": total_fetched_results # Add total count for info
            }
            logging.info(f"Pagination info (manual slicing): {recommended_pagination_info}")
        else:
             logging.info("No recommended jobs returned by API.")
             recommended_jobs_page = [] # Ensure it's an empty list
             # Set pagination info to indicate no results/pages
             recommended_pagination_info = { "current_page": 1, "total_pages": 0, "has_prev": False, "has_next": False, "total_results": 0 }

    # Fetch General Jobs
    logging.info(f"Attempting general search with query: '{general_query}' and location: '{user_location}'")
    general_response = call_serpapi_google_jobs(general_query, user_location)
    general_jobs_all = parse_serpapi_jobs(general_response)
    print(f"DEBUG: number of general jobs: {len(general_jobs_all)}")
    if "error" in general_response:
         flash(f"Could not fetch general jobs: {general_response['error']}", "warning")
         logging.warning(f"SerpApi error for general jobs: {general_response['error']}")

    else:
        # --- Perform Pagination on the Fetched Results ---
        total_fetched_results = len(general_jobs_all)
        logging.info(f"Total general jobs fetched for pagination: {total_fetched_results}")

        if total_fetched_results > 0:
            start_slice_index = (page - 1) * results_per_page
            end_slice_index = start_slice_index + results_per_page
            general_jobs_page = general_jobs_all[start_slice_index:end_slice_index] # Slice the list

            total_pages = math.ceil(total_fetched_results / results_per_page)

            general_pagination_info = {
                "current_page": page,
                "total_pages": total_pages,
                "has_prev": page > 1,
                "has_next": page < total_pages,
                "prev_num": page - 1 if page > 1 else None,
                "next_num": page + 1 if page < total_pages else None,
                "total_results": total_fetched_results # Add total count for info
            }
            logging.info(f"Pagination info (manual slicing): {general_pagination_info}")
        else:
            logging.info("No general jobs returned by API.")
            general_jobs_page = [] # Ensure it's an empty list
            # Set pagination info to indicate no results/pages
            general_pagination_info = { "current_page": 1, "total_pages": 0, "has_prev": False, "has_next": False, "total_results": 0 }

    return render_template('jobs.html',
                           recommended_jobs=recommended_jobs_page,
                           general_jobs=general_jobs_page,
                           user_info=session.get('user_info'),
                           api_error=False,
                           recommended_pagination=recommended_pagination_info,
                           general_pagination=general_pagination_info) # API was attempted
                           

@app.route('/logout')
def logout():
    """Logs the user out by clearing the session."""
    # Optional: Revoke the token (more secure, but user needs to re-authorize)
    # credentials = get_google_credentials()
    # if credentials and credentials.token:
    #     try:
    #         requests.post('https://oauth2.googleapis.com/revoke',
    #             params={'token': credentials.token},
    #             headers = {'content-type': 'application/x-www-form-urlencoded'})
    #     except Exception as e:
    #         print(f"Warning: Failed to revoke token: {e}") # Log error

    session.clear() # Clear all session data
    flash("You have been logged out.", "info")
    return redirect(url_for('index'))

def update_progress(job_id, status, processed, total, current_email=None, included=None, error=None):
    """Updates the progress of the job in Redis."""
    try:
        print(f"DEBUG update_progress: Updating job {job_id} - status: {status}, processed: {processed}, total: {total}")
        mapping = {'status': status, 'processed': str(processed), 'total': str(total)}
        
        # Add optional parameters to mapping if provided
        if current_email is not None:
            mapping['current_email'] = current_email
        if included is not None:
            mapping['included'] = str(included)
        if error is not None:
            mapping['error'] = error
            
        redis_client.hset(job_id, mapping=mapping)
        print(f"DEBUG update_progress: Update successful")
    except Exception as e:
        print(f"DEBUG update_progress: Error updating progress in Redis: {str(e)}")

@app.route('/process_emails', methods=['POST'])
def process_emails():
    """Handles the form submission to start email processing."""
    if 'user_info' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('index'))

    # Get Gemini Key from form and store in session
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        flash("Gemini API Key is required.", "danger")
        return redirect(url_for('index'))
    session[GEMINI_API_KEY] = gemini_key # Store key in session

    # Get other parameters from form (add more as needed)
    sheet_name = request.form.get('sheet_name', 'Job Application Tracker')
    # Define statuses to include (could make this configurable via form later)
    included_statuses = ["Interview Request", "Assessment Request", "Offer", "Rejection"]
    max_emails_str = request.form.get('max_emails', '50') # Default to 50 for safety
    try:
        max_emails = int(max_emails_str) if max_emails_str else None
    except ValueError:
        flash("Invalid value for maximum emails.", "danger")
        return redirect(url_for('index'))

    # Create a unique job ID and store it in Redis
    job_id = str(uuid.uuid4())
    session['job_id'] = job_id
    # Force session to be saved immediately
    session.modified = True
    print(f"DEBUG: Set job_id in session: {job_id}")
    
    # Store job_id in Redis as well as a backup
    redis_client.set('current_job_id', job_id)
    
    # Store processing parameters in Redis
    redis_client.hset(f"{job_id}:params", mapping={
        'sheet_name': sheet_name,
        'max_emails': str(max_emails or 50),
        'included_statuses': json.dumps(included_statuses)
    })
    
    # Initialize progress in Redis with 'connecting' status
    update_progress(job_id, 'connecting', 0, max_emails or 100)
    
    # Immediately redirect to results page
    return redirect(url_for('results'))

# Update the progress route to use Redis instead of session
@app.route('/progress')
def progress():
    """Returns the current progress of email processing as JSON."""
    job_id = request.args.get('job_id')
    print(f"DEBUG /progress: job_id from request: {job_id}")
    
    if not job_id:
        job_id = session.get('job_id')
        print(f"DEBUG /progress: job_id from session: {job_id}")
        
        if not job_id:
            try:
                job_id = redis_client.get('current_job_id')
                if job_id:
                    job_id = job_id.decode('utf-8')
                    print(f"DEBUG /progress: Retrieved job_id from Redis: {job_id}")
            except Exception as e:
                print(f"DEBUG /progress: Error retrieving job_id from Redis: {str(e)}")
    
    if not job_id:
        print("DEBUG /progress: No job_id found in request, session or Redis")
        return jsonify({
            'current': 0,
            'total': 0,
            'processed': 0,
            'included': 0,
            'current_email': 'Not started',
            'status': 'Not started',
            'error': 'No job ID found',
            'live_results': []
        })
    
    try:
        # Get progress data from Redis
        progress_data = redis_client.hgetall(job_id)
        # print(f"DEBUG /progress: Progress data from Redis: {progress_data}")
        
        if not progress_data:
            # print(f"DEBUG /progress: No progress data found for job_id: {job_id}")
            return jsonify({
                'current': 0,
                'total': 0,
                'processed': 0,
                'included': 0,
                'current_email': 'Initializing...',
                'status': 'initializing',
                'message': 'Waiting for processing to start',
                'live_results': []
            })
        
        # Extract data from Redis (convert from bytes)
        status = progress_data.get(b'status', b'initializing').decode('utf-8')
        processed = int(progress_data.get(b'processed', b'0').decode('utf-8'))
        total = int(progress_data.get(b'total', b'100').decode('utf-8'))
        included = int(progress_data.get(b'included', b'0').decode('utf-8'))
        current_email = progress_data.get(b'current_email', b'Processing...').decode('utf-8')
        error = progress_data.get(b'error')
        if error:
            error = error.decode('utf-8')
        
        # Get live results from Redis
        live_results = []
        
        # Fetch the most recent results (up to 20)
        for i in range(included):
            result_key = f"{job_id}:result:{i}"
            result_data = redis_client.hgetall(result_key)
            if result_data:
                # Convert bytes to strings
                result = {k.decode('utf-8'): v.decode('utf-8') for k, v in result_data.items()}
                live_results.append({
                    'id': i,
                    'company': result.get('Company Name', 'Unknown Company'),
                    'job_title': result.get('Job Title', 'Unknown Position'),
                    'status': result.get('Status', 'Unknown Status'),
                    'subject': result.get('Email Subject', 'No Subject')
                })
        
        # Limit to most recent 20 results
        live_results = live_results[-20:] if len(live_results) > 20 else live_results
        
        # Build response
        response = {
            'current': processed,
            'total': total,
            'processed': processed,
            'included': included,
            'current_email': current_email,
            'status': status,
            'live_results': live_results
        }
        
        if error:
            response['error'] = error
        
        # print(f"DEBUG /progress: Returning response with {len(live_results)} live results")
        return jsonify(response)
    except Exception as e:
        # print(f"DEBUG /progress: Error occurred: {str(e)}")
        # Return a fallback response in case of error
        return jsonify({
            'current': 0,
            'total': 0,
            'processed': 0,
            'included': 0,
            'current_email': f'Error: {str(e)}',
            'status': 'error',
            'error': str(e),
            'live_results': []
        })

@app.route('/get_results')
def get_results():
    """Returns the latest results for a job."""
    job_id = request.args.get('job_id')
    if not job_id:
        job_id = session.get('job_id')
        if not job_id:
            try:
                job_id = redis_client.get('current_job_id')
                if job_id:
                    job_id = job_id.decode('utf-8')
            except:
                pass
    
    if not job_id:
        return jsonify({'status': 'error', 'message': 'No job ID found'})
    
    # Get the latest results
    included_count = int(redis_client.hget(job_id, 'included') or b'0')
    results = []
    
    for i in range(included_count):
        result_key = f"{job_id}:result:{i}"
        result_data = redis_client.hgetall(result_key)
        if result_data:
            result = {}
            for k, v in result_data.items():
                result[k.decode('utf-8')] = v.decode('utf-8')
            results.append(result)
    
    # Get errors
    errors = []
    error_list = redis_client.lrange(f"{job_id}:errors", 0, -1)
    for error in error_list:
        errors.append(error.decode('utf-8'))
    
    return jsonify({
        'results': results,
        'errors': errors
    })

@app.route('/start_processing')
def start_processing():
    """Background route that actually processes the emails."""
    job_id = request.args.get('job_id')
    if not job_id:
        job_id = session.get('job_id')
        if not job_id:
            try:
                job_id = redis_client.get('current_job_id')
                if job_id:
                    job_id = job_id.decode('utf-8')
            except:
                pass
    
    if not job_id:
        return jsonify({'status': 'error', 'message': 'No job ID found'})
    
    # Check if user is authenticated
    if 'credentials' not in session:
        update_progress(job_id, 'error', 0, 0, error='Not authenticated with Google. Please log in again.')
        return jsonify({'status': 'error', 'message': 'Not authenticated with Google'})
    
    # Get parameters from Redis
    params = redis_client.hgetall(f"{job_id}:params")
    if not params:
        update_progress(job_id, 'error', 0, 0, error='No parameters found for job')
        return jsonify({'status': 'error', 'message': 'No parameters found for job'})
    
    # Convert parameters from bytes
    sheet_name = params.get(b'sheet_name', b'Job Application Tracker').decode('utf-8')
    max_emails_str = params.get(b'max_emails', b'50').decode('utf-8')
    included_statuses_json = params.get(b'included_statuses', b'["Interview Request", "Assessment Request", "Offer", "Rejection"]').decode('utf-8')
    
    try:
        max_emails = int(max_emails_str)
        included_statuses = json.loads(included_statuses_json)
    except (ValueError, json.JSONDecodeError) as e:
        update_progress(job_id, 'error', 0, 0, error=f'Invalid parameters: {str(e)}')
        return jsonify({'status': 'error', 'message': f'Invalid parameters: {str(e)}'})

    # --- Build API Services ---
    try:
        # Update progress to show we're connecting
        update_progress(job_id, 'connecting', 0, max_emails or 100)
        
        # Build Gmail service
        gmail_service = build_google_service('gmail', 'v1')
        if not gmail_service:
            update_progress(job_id, 'error', 0, 0, error='Failed to connect to Gmail. Please check your authentication.')
            return jsonify({'status': 'error', 'message': 'Failed to connect to Gmail'})
            
        # Test Gmail connection
        try:
            profile = gmail_service.users().getProfile(userId='me').execute()
            print(f"DEBUG: Successfully connected to Gmail for {profile.get('emailAddress')}")
        except Exception as e:
            update_progress(job_id, 'error', 0, 0, error=f'Gmail connection test failed: {str(e)}')
            return jsonify({'status': 'error', 'message': f'Gmail connection test failed: {str(e)}'})
        
        # Build other services
        sheets_service = build_google_service('sheets', 'v4')
        drive_service = build_google_service('drive', 'v3')
        
        # Get gspread client
        credentials = get_google_credentials()
        if not credentials:
            update_progress(job_id, 'error', 0, 0, error='No valid credentials for Google Sheets')
            return jsonify({'status': 'error', 'message': 'No valid credentials for Google Sheets'})
            
        gspread_client = gspread.authorize(credentials)
        
        print(f"DEBUG: All services built successfully")
        
    except Exception as e:
        error_msg = f'Error building Google services: {str(e)}'
        print(f"DEBUG: {error_msg}")
        update_progress(job_id, 'error', 0, 0, error=error_msg)
        return jsonify({'status': 'error', 'message': error_msg})

    # --- Processing Logic ---
    results = []
    processed_count = 0
    included_count = 0
    errors = []
    sheet_url = "#"

    try:
        # Update progress to 'searching' status
        update_progress(job_id, 'searching', 0, max_emails or 100)
        
        # 1. Find or Create Spreadsheet
        try:
            # Search for existing sheet by name
            response = drive_service.files().list(
                q=f"name='{sheet_name}' and mimeType='application/vnd.google-apps.spreadsheet' and trashed=false",
                spaces='drive', fields='files(id, name, webViewLink)').execute()
            files = response.get('files', [])
            print(f"DEBUG: Found {len(files)} matching spreadsheets")
            
            if files:
                spreadsheet_id = files[0]['id']
                sheet_url = files[0]['webViewLink']
                spreadsheet = gspread_client.open_by_key(spreadsheet_id)
                print(f"DEBUG: Found existing sheet: {sheet_name} (ID: {spreadsheet_id})")
            else:
                # Create sheet if not found
                print(f"DEBUG: Creating new sheet: {sheet_name}")
                spreadsheet = gspread_client.create(sheet_name)
                spreadsheet_id = spreadsheet.id
                sheet_url = spreadsheet.url
                print(f"DEBUG: Created sheet: {sheet_name} (ID: {spreadsheet_id})")

            # Get or create worksheet
            worksheet_title = "Job Applications"
            try:
                worksheet = spreadsheet.worksheet(worksheet_title)
            except gspread.exceptions.WorksheetNotFound:
                worksheet = spreadsheet.add_worksheet(title=worksheet_title, rows=1000, cols=10)

            # Ensure headers
            headers = ['Date Received', 'Company Name', 'Job Title', 'Status', 'Email Subject', 'Sender', 'Email ID']
            existing_headers = worksheet.row_values(1) if worksheet.row_count > 0 else []
            if not existing_headers or existing_headers != headers:
                if existing_headers: worksheet.delete_rows(1)
                worksheet.insert_row(headers, 1)
                worksheet.format('A1:G1', {'textFormat': {'bold': True}})

        except Exception as e:
            error_msg = f"Error accessing/creating Google Sheet: {str(e)}"
            print(f"DEBUG: {error_msg}")
            errors.append(error_msg)
            update_progress(job_id, 'error', 0, 0, error=error_msg)
            raise

        # 2. Search Emails
        # Use a broader query to ensure we get results
        query = 'subject:(application OR interview OR offer OR rejection OR assessment OR "thank you" OR position OR job)'
        
        print(f"DEBUG: Searching emails with query: {query}")
        try:
            response = gmail_service.users().messages().list(userId='me', q=query, maxResults=max_emails or 100).execute()
            messages = response.get('messages', [])
            print(f"DEBUG: Found {len(messages)} potentially relevant messages")
            
            if not messages:
                error_msg = "No matching emails found. Try a different search query."
                update_progress(job_id, 'error', 0, 0, error=error_msg)
                return jsonify({'status': 'error', 'message': error_msg})
                
            # Update progress with total count
            update_progress(job_id, 'analyzing', 0, len(messages))
            
            # Store sheet URL in Redis
            redis_client.hset(job_id, 'sheet_url', sheet_url)
            redis_client.hset(job_id, 'sheet_name', sheet_name)
        except Exception as e:
            error_msg = f"Error searching emails: {str(e)}"
            print(f"DEBUG: {error_msg}")
            update_progress(job_id, 'error', 0, 0, error=error_msg)
            return jsonify({'status': 'error', 'message': error_msg})
        
        # 3. Process Each Email
        for i, message_info in enumerate(messages):
            if max_emails is not None and processed_count >= max_emails:
                print(f"DEBUG: Reached processing limit of {max_emails}")
                break

            msg_id = message_info['id']
            processed_count += 1
            print(f"DEBUG: Processing email {i+1}/{len(messages)} (ID: {msg_id})")

            try:
                # Fetch full email content
                msg = gmail_service.users().messages().get(userId='me', id=msg_id, format='full').execute()
                payload = msg.get('payload', {})
                headers = payload.get('headers', [])

                subject = next((decode_email_part(h['value']) for h in headers if h['name'].lower() == 'subject'), 'No Subject')
                sender = next((decode_email_part(h['value']) for h in headers if h['name'].lower() == 'from'), 'Unknown Sender')
                date_str = next((h['value'] for h in headers if h['name'].lower() == 'date'), None)

                if date_str:
                    date_received = parsedate_to_datetime(date_str).strftime('%Y-%m-%d')
                else:
                    date_received = "Unknown Date"

                body = get_email_body(payload)
                print(f"DEBUG: Email subject: {subject}")
                print(f"DEBUG: Email body length: {len(body) if body else 0}")

                if not body and not subject: 
                    print("DEBUG: Skipping email with no body and subject")
                    continue

                # Update progress with current email info
                update_progress(job_id, 'analyzing', processed_count, len(messages), current_email=subject)

                # Call Gemini
                extracted_info = extract_job_info_with_gemini(GEMINI_API_KEY, subject, body, date_received)
                print(f"DEBUG: Extracted info: {extracted_info}")
                
                if extracted_info:
                    status_val = extracted_info.get('Status', 'Unable to Determine')
                    if status_val in included_statuses:
                        record = {
                            'Date Received': date_received,
                            'Company Name': extracted_info.get('Company Name', 'Not Found'),
                            'Job Title': extracted_info.get('Job Title', 'Not Found'),
                            'Status': status_val,
                            'Email Subject': subject,
                            'Sender': sender,
                            'Email ID': msg_id
                        }
                        
                        # Store result in Redis for live updates
                        result_key = f"{job_id}:result:{included_count}"
                        redis_client.hset(result_key, mapping=record)
                        included_count += 1
                        redis_client.hset(job_id, 'included', str(included_count))
                        
                        # Update progress
                        update_progress(job_id, 'writing', processed_count, len(messages), included=included_count)
                        
                        # Append to Google Sheet
                        headers_to_append = ['Date Received', 'Company Name', 'Job Title', 'Status', 'Email Subject', 'Sender', 'Email ID']
                        row_to_append = [record[h] for h in headers_to_append]
                        try:
                            worksheet.append_row(row_to_append, value_input_option='USER_ENTERED')
                            print(f"DEBUG: Added row to sheet for {record['Company Name']}")
                        except Exception as sheet_err:
                            error_msg = f"Error writing row for email {msg_id} to Sheet: {str(sheet_err)}"
                            errors.append(error_msg)
                            print(f"DEBUG: {error_msg}")
                            redis_client.lpush(f"{job_id}:errors", error_msg)

                time.sleep(1.1) # Rate limit Gemini calls

            except Exception as e:
                error_msg = f"Error processing email {msg_id}: {str(e)}"
                errors.append(error_msg)
                print(f"DEBUG: {error_msg}")
                redis_client.lpush(f"{job_id}:errors", error_msg)
                continue

    except Exception as e:
        error_msg = f"Major error during processing: {str(e)}"
        print(f"DEBUG: {error_msg}")
        errors.append(error_msg)
        redis_client.lpush(f"{job_id}:errors", error_msg)
        update_progress(job_id, 'error', processed_count, len(messages) if 'messages' in locals() else 0, error=error_msg)
        return jsonify({'status': 'error', 'message': error_msg})

    # Update progress to 'finishing' status
    update_progress(job_id, 'finishing', processed_count, len(messages) if 'messages' in locals() else 0)
    
    # Store final results count in Redis
    redis_client.hset(job_id, 'processed_count', str(processed_count))
    redis_client.hset(job_id, 'included_count', str(included_count))
    
    # Final update to 'complete' status
    update_progress(job_id, 'complete', processed_count, len(messages) if 'messages' in locals() else 0)
    
    return jsonify({
        'status': 'complete',
        'processed_count': processed_count,
        'included_count': included_count,
        'sheet_url': sheet_url
    })

@app.route('/results')
def results():
    """Displays the results page with live processing updates."""
    job_id = request.args.get('job_id')
    if not job_id:
        job_id = session.get('job_id')
        if not job_id:
            try:
                job_id = redis_client.get('current_job_id')
                if job_id:
                    job_id = job_id.decode('utf-8')
            except:
                pass
    
    # If no job_id found, redirect to index
    if not job_id:
        flash("No processing job found. Please start a new processing job.", "warning")
        return redirect(url_for('index'))
    
    # Check if processing has started
    progress_data = redis_client.hgetall(job_id)
    processing_started = bool(progress_data)
    
    # Get sheet info if available
    sheet_url = progress_data.get(b'sheet_url', b'#').decode('utf-8') if progress_data else "#"
    sheet_name = progress_data.get(b'sheet_name', b'Job Application Tracker').decode('utf-8') if progress_data else "Job Application Tracker"
    
    # Render the results template
    return render_template('results.html',
                          results=[],
                          processed_count=0,
                          included_count=0,
                          errors=[],
                          sheet_url=sheet_url,
                          sheet_name=sheet_name,
                          is_processing=True,
                          job_id=job_id,
                          processing_started=processing_started)

# --- Run the App ---
if __name__ == '__main__':
    # Make sure necessary environment variables are set
    if not CLIENT_ID or not CLIENT_SECRET:
        print("ERROR: GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables must be set.")
    else:
        # This is important for OAuth state handling in some environments
        os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1' # Allow HTTP for local testing ONLY
        app.run(debug=True) # Run in debug mode for development
        # For production, use a proper WSGI server like Gunicorn or Waitress
        # And set debug=False, SESSION_COOKIE_SECURE=True (if using HTTPS)
