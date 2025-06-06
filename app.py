import os
import json
import flask
from flask import Flask, redirect, request, session, url_for, render_template, flash, jsonify, send_file
from flask_cors import CORS  # Add this import
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

# Allow OAuth 2.0 to work in development mode
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

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
CORS(app)  # Enable CORS for all routes
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
app.config['SESSION_REFRESH_EACH_REQUEST'] = True

# Add session check middleware
@app.before_request
def before_request():
    if 'user_info' in session:
        # Refresh session on each request if user is logged in
        session.modified = True

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

class SavedEmail(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_google_id = db.Column(db.String(100), db.ForeignKey('user_profile.google_id'), nullable=False, index=True)
    gmail_message_id = db.Column(db.String(100), nullable=False) # Gmail's unique ID for the message
    subject = db.Column(db.String(500)) # Increased length for potentially long subjects
    sender = db.Column(db.String(255))
    date_received = db.Column(db.Date) # Store only the date
    body_snippet = db.Column(db.Text) # Store a snippet of the body
    extracted_company = db.Column(db.String(255))
    extracted_job_title = db.Column(db.String(255))
    extracted_status = db.Column(db.String(100)) # The status assigned by Gemini
    saved_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Define relationship (optional)
    user = db.relationship('UserProfile', backref=db.backref('saved_emails', lazy='dynamic', cascade="all, delete-orphan"))

    # Unique constraint per user per email message
    __table_args__ = (db.UniqueConstraint('user_google_id', 'gmail_message_id', name='_user_email_message_uc'),)

    def __repr__(self):
        return f'<SavedEmail {self.user_google_id} - {self.subject}>'
    
with app.app_context():
    try:
        inspector = inspect(db.engine)
        tables_to_create = []
        table_models = [UserProfile, SavedEmail] # List all your models here

        for model in table_models:
            table_name = model.__tablename__
            if not inspector.has_table(table_name):
                # Add the actual table object from the model's metadata
                tables_to_create.append(model.__table__)
                logging.info(f"Table '{table_name}' marked for creation.")
            else:
                logging.info(f"Table '{table_name}' already exists.")

        # Create tables if any are marked
        if tables_to_create:
            logging.info(f"Creating database tables: {[t.name for t in tables_to_create]}")
             # Ensure db.metadata contains all tables before calling create_all
            db.metadata.create_all(bind=db.engine, tables=tables_to_create)
            logging.info("Database tables checked/created.")
        else:
             logging.info("All required database tables already exist.")

    except Exception as e:
        # Log the full traceback for database errors
        logging.error(f"Error during database table check/creation: {e}", exc_info=True)

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_google_id = db.Column(db.String(100), db.ForeignKey('user_profile.google_id'), nullable=False)
    title = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with messages
    messages = db.relationship('Message', backref='conversation', lazy='dynamic', cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Conversation {self.id} - {self.title}>'

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    is_user = db.Column(db.Boolean, default=True)  # True for user messages, False for AI responses
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Message {self.id} - {"User" if self.is_user else "AI"}>'

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
        model = genai.GenerativeModel('gemini-2.0-flash') # Or 'gemini-pro'

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
    """Gets Google credentials from the session."""
    if 'credentials' not in session:
        logging.debug("No 'credentials' found in session.")
        return None

    try:
        credentials_json_str = session['credentials']
        credentials_info = json.loads(credentials_json_str)
        # The 'expiry' field in credentials_info (from to_json()) is typically an ISO 8601 string.
        # Credentials.from_authorized_user_info should handle this.
        creds = Credentials.from_authorized_user_info(credentials_info)

        # The library should handle expiry internally. If it's already a datetime, ensure it's UTC.
        # If it needs to be naive UTC for some reason (less common with this lib):
        if creds.expiry and creds.expiry.tzinfo:
            creds.expiry = creds.expiry.astimezone(timezone.utc).replace(tzinfo=None)

        logging.debug(f"Successfully loaded credentials from session. Valid: {creds.valid}")
        return creds
    except json.JSONDecodeError:
        logging.error("Error: Could not decode credentials from session.", exc_info=True)
        session.pop('credentials', None)
        return None
    except Exception as e:
        logging.error(f"Error loading/creating credentials from session: {e}", exc_info=True)
        session.pop('credentials', None)
        return None

def save_google_credentials(credentials):
    """Saves Google credentials to the session."""
    try:
        # credentials.to_json() returns a JSON string.
        # No need to json.loads and then json.dumps if you store the string directly.
        session['credentials'] = credentials.to_json()
        logging.debug("Successfully saved credentials to session")
    except Exception as e:
        logging.error(f"Error saving credentials to session: {e}", exc_info=True)
        session.pop('credentials', None)

def build_google_service(service_name, version):
    credentials = get_google_credentials() # Use the refactored getter
    if not credentials:
        logging.error(f"No valid credentials for {service_name}")
        flash("Your session may have expired or credentials are not found. Please log in again.", "warning")
        return None # Or redirect to login

    # logging.debug(f"Building {service_name} service v{version}")
    # logging.debug(f"Using credentials. Valid: {credentials.valid}, Expired: {credentials.expired}")


    if credentials.expired and credentials.refresh_token:
        logging.info("Refreshing expired token")
        try:
            credentials.refresh(GoogleAuthRequest())
            save_google_credentials(credentials) # Re-save refreshed credentials
            logging.info("Token refreshed successfully.")
        except google.auth.exceptions.RefreshError as e:
            logging.error(f"Error refreshing token: {str(e)}", exc_info=True)
            if "invalid_grant" in str(e).lower() or "token has been revoked" in str(e).lower():
                logging.warning("Token has been revoked or is invalid, clearing session.")
                session.clear()
                flash("Your authorization has expired or been revoked. Please log in again.", "danger")
            else:
                flash("Could not refresh your session. Please try logging in again.", "warning")
            return None
        except Exception as e: # Catch any other refresh exception
            logging.error(f"Unexpected error refreshing token: {str(e)}", exc_info=True)
            flash("An unexpected error occurred while refreshing your session. Please try logging in again.", "warning")
            return None
    elif credentials.expired and not credentials.refresh_token:
        logging.warning("Token expired and no refresh token available. Clearing session.")
        session.clear()
        flash("Your session has expired and cannot be refreshed. Please log in again.", "danger")
        return None


    if not credentials.valid:
        logging.warning(f"Credentials are not valid for {service_name}. Clearing session.")
        session.clear() # Or handle as appropriate
        flash("Your session is invalid. Please log in again.", "warning")
        return None

    try:
        service = build(service_name, version, credentials=credentials)
        logging.debug(f"Successfully built {service_name} service.")
        return service
    except HttpError as e:
        logging.error(f"HttpError building {service_name} service: {e.resp.status} - {e.content}", exc_info=True)
        if e.resp.status in [401, 403]:
            flash("Authentication error accessing Google Services. Your session might be invalid. Please log in again.", "danger")
            session.clear() # Potentially clear session if auth fails critically
        else:
            flash(f"Error accessing Google Service ({service_name}): {e.resp.status}. Please try again.", "warning")
        return None
    except Exception as e:
        logging.error(f"Unexpected error building {service_name} service: {str(e)}", exc_info=True)
        flash(f"An unexpected error occurred while accessing Google Service ({service_name}).", "danger")
        return None
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def build_google_service(service_name, version):
#     try:
#         credentials_dict = session.get('credentials')
#         if not credentials_dict:
#             print(f"DEBUG: No credentials in session for {service_name}")
#             return None
            
#         print(f"DEBUG: Building {service_name} service v{version}")
#         print(f"DEBUG: Loaded credentials dict from session: {credentials_dict}")
        
#         # Create credentials object from dict
#         try:
#             print("DEBUG: Creating Credentials object directly from loaded dict...")
#             print(credentials_dict)
#             credentials = Credentials.from_authorized_user_info(credentials_dict)
#             print("DEBUG: Credentials object created.")
#             print(f"DEBUG: Type of creds.expiry after creation: {type(credentials.expiry)}")
#             print(f"DEBUG: Timezone info of creds.expiry: {credentials.expiry.tzinfo}")
#             print(credentials)
#         except Exception as e:
#             print(f"DEBUG: Error creating credentials object: {str(e)}")
#             session.clear()
#             return None
            
#         if not credentials:
#             print(f"DEBUG: No valid credentials for {service_name}")
#             return None
            
#         print(f"DEBUG: Token valid: {credentials.valid}")
#         print(f"DEBUG: Token expired: {credentials.expired}")
        
#         # Force token refresh if expired
#         if credentials.expired:
#             print("DEBUG: Refreshing expired token")
#             try:
#                 credentials.refresh(GoogleAuthRequest())
#             except Exception as e:
#                 print(f"DEBUG: Error refreshing token: {str(e)}")
#                 if "invalid_grant" in str(e).lower():
#                     print("DEBUG: Token has been revoked, clearing session")
#                     session.clear()
#                     flash("Your session has expired. Please log in again.", "warning")
#                 return None
            
#         try:
#             service = build(service_name, version, credentials=credentials)
#             return service
#         except Exception as e:
#             print(f"DEBUG: Error building {service_name} service: {str(e)}")
#             if isinstance(e, HttpError) and e.resp.status in [401, 403]:
#                 flash("Authentication error accessing Google Services. Please log in again.", "danger")
#                 session.clear()
#             return None
    
#     except Exception as e:
#         print(f"DEBUG: Error building {service_name} service: {str(e)}")
#         return None

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
    """
    Uses Gemini to extract job application details with advanced error handling and response validation.
    
    Args:
        gemini_key: The Gemini API key
        subject: Email subject line
        body: Email body text
        date_received: Date the email was received
        
    Returns:
        Dictionary containing extracted job information or error details
    """
    if not gemini_key:
        logging.error("Gemini API key missing for job info extraction")
        return {"Status": "Error: Gemini API Key Missing", "Company Name": "Not Available", "Job Title": "Not Available"}
    
    if not body and not subject:
        logging.info("Both email subject and body are empty, skipping extraction")
        return None

    # Prepare email content, prioritizing meaningful content
    email_content = body if body else ""
    if len(email_content) > 12000:  # Truncate very long emails to stay within context limits
        email_content = email_content[:12000] + "... [Content truncated due to length]"

    # Enhanced structured prompt with clear extraction guidelines
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
                * **Rejection**: Choose if keywords like "regret", "unfortunately", "not selected", "other candidates", "will not be moving forward", "position has been filled", "unable to offer you the position" are present and represent the core message. Sometimes even "thank you for applying" is a rejection.
                * **Offer**: Choose if keywords like "offer", "employment offer", "job offer", "compensation", "salary", "benefits", "start date", "welcome aboard", "joining us", "extend an offer" are present and form the main purpose.
                * **Interview Request**: Choose if keywords like "interview", "schedule time", "schedule a call", "speak with", "talk further", "next steps involve a call/meeting", "discussion with the team" clearly indicate a request to schedule or conduct an interview.
                * **Assessment Request**: Choose if keywords like "assessment", "test", "coding challenge", "technical screen", "assignment", "Hackerrank", "Codility", "online assessment" indicate the next step is a test or skills evaluation.
                * **Keep In Touch / Future Consideration**: Choose ONLY if the *primary message* is about keeping the application/resume on file for *future* roles, often in the absence of strong rejection keywords for the *current* role (e.g., "keep your resume on file", "consider you for future opportunities", "reach out if a suitable role opens"). If clear rejection language for the *current* role is present, choose 'Rejection' instead, even if future consideration is mentioned secondarily.
                * **Application Acknowledgment**: Choose ONLY if the email *solely* confirms receipt of the application and indicates it's under review, without providing any further status update (e.g., "received your application", "application is under review", "will be in touch if"). If other status keywords are present, prioritize those categories.
                * **Informational / General Company Email**: Choose if the email is clearly a newsletter, marketing communication, company update, job alert notification (for *new* jobs, not a status update on an *existing* application), or other general communication not tied to the status of a specific, active application process.
                * **Other**: Choose if the email's purpose related to the application process is clear but does not fit neatly into any of the above categories (e.g., a request for more information/documents *from* the applicant, notification of a delay in the process, system error message).
                * **Unable to Determine**: Choose ONLY if the email content is extremely ambiguous, lacks sufficient context, or is corrupted/incomplete, making it impossible to confidently determine the company, title, or status.

        4.  **Output Format:**
            * Produce **ONLY** the requested information.
            * Do **NOT** include any introductory phrases, explanations, confidence scores, or the reasoning process in the final output.
            * Use the following exact format, replacing bracketed placeholders with the extracted information:

            ```json
            {{
                "Company Name": "[Extracted Company Name or 'Not Mentioned']",
                "Job Title": "[Extracted Job Title or 'Not Mentioned']",
                "Status": "[Chosen Status Category]",
                "Reasoning": "[Brief explanation of why this status was chosen]"
            }}
            ```

        --- Email Content ---
        Date Received: {date_received}
        Subject: {subject}
        Body:
        {email_content}
        --- End Email Content ---

        Output:
    """
    
    try:
        # Configure Gemini API with proper error handling
        genai.configure(api_key=gemini_key)
        
        # Use the most appropriate model based on content length and complexity
        model_name = 'gemini-2.0-flash'  # Default to faster model
        if len(email_content) > 8000 or subject and "interview" in subject.lower():
            model_name = 'gemini-2.0-pro'  # Use more capable model for complex or long content
            
        gemini_model = genai.GenerativeModel(model_name)
        
        # Request structured JSON response with low temperature for consistency
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        # Process the response with robust error handling
        response_text = response.text.strip()
        logging.debug(f"Raw Gemini response for job info: {response_text[:200]}...")  # Log beginning for debugging
        
        # Clean up response if it contains markdown code blocks
        if response_text.startswith("```json"):
            response_text = response_text[7:].strip()
        if response_text.endswith("```"):
            response_text = response_text[:-3].strip()
            
        # Try to parse as JSON first
        try:
            extracted_data = json.loads(response_text)
            # Validate required fields
            required_fields = ["Company Name", "Job Title", "Status"]
            for field in required_fields:
                if field not in extracted_data:
                    extracted_data[field] = "Not Mentioned"
            
            # Ensure reasoning field exists
            if "Reasoning" not in extracted_data:
                extracted_data["Reasoning"] = "No reasoning provided"
                
            # Normalize status values for consistency
            if extracted_data["Status"] in ["Unknown", "Unclear", "Not clear"]:
                extracted_data["Status"] = "Unable to Determine"
                
            return extracted_data
            
        except json.JSONDecodeError:
            # Fallback to line-by-line parsing if JSON parsing fails
            extracted_data = {"Company Name": "Not Mentioned", "Job Title": "Not Mentioned", "Status": "Parsing Error", "Reasoning": "Failed to parse response"}
            lines = response_text.split('\n')
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key in extracted_data:
                        extracted_data[key] = value
            
            logging.warning(f"Failed to parse JSON response, used line-by-line fallback: {extracted_data}")
            return extracted_data
            
    except Exception as e:
        logging.error(f"Error calling Gemini API for job info extraction: {str(e)}", exc_info=True)
        
        # Provide detailed error status for better debugging and user feedback
        error_status = "LLM Error"
        error_details = str(e).lower()
        
        if "api key not valid" in error_details or "invalid api key" in error_details:
            error_status = "LLM Error: Invalid API Key"
        elif "api_key_service_blocked" in error_details:
            error_status = "LLM Error: API Blocked"
        elif "resource_exhausted" in error_details or "quota exceeded" in error_details:
            error_status = "LLM Error: Quota Exceeded"
        elif "deadline exceeded" in error_details or "timeout" in error_details:
            error_status = "LLM Error: Request Timeout"
        elif "internal" in error_details:
            error_status = "LLM Error: Service Unavailable"
        
        return {
            "Company Name": "Not Available", 
            "Job Title": "Not Available", 
            "Status": error_status,
            "Reasoning": f"Technical error: {str(e)[:100]}"
        }

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
        recommended_page = int(request.args.get('page_recommended', 1))
        general_page = int(request.args.get('page_general', 1))
    except ValueError:
        recommended_page = 1
        general_page = 1
    if recommended_page < 1: recommended_page = 1
    if general_page < 1: general_page = 1
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
            start_slice_index = (recommended_page - 1) * results_per_page
            end_slice_index = start_slice_index + results_per_page
            recommended_jobs_page = recommended_jobs_all[start_slice_index:end_slice_index] # Slice the list

            total_pages = math.ceil(total_fetched_results / results_per_page)

            recommended_pagination_info = {
                "current_page": recommended_page,
                "total_pages": total_pages,
                "has_prev": recommended_page > 1,
                "has_next": recommended_page < total_pages,
                "prev_num": recommended_page - 1 if recommended_page > 1 else None,
                "next_num": recommended_page + 1 if recommended_page < total_pages else None,
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
            start_slice_index = (general_page - 1) * results_per_page
            end_slice_index = start_slice_index + results_per_page
            general_jobs_page = general_jobs_all[start_slice_index:end_slice_index] # Slice the list

            total_pages = math.ceil(total_fetched_results / results_per_page)

            general_pagination_info = {
                "current_page": general_page,
                "total_pages": total_pages,
                "has_prev": general_page > 1,
                "has_next": general_page < total_pages,
                "prev_num": general_page - 1 if general_page > 1 else None,
                "next_num": general_page + 1 if general_page < total_pages else None,
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
    
    user_google_id = session['user_info']['google_id'] # <<-- Get google_id


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
    
    # Create a unique job ID and store it in Redis
    job_id = str(uuid.uuid4())
    session['job_id'] = job_id
    session.modified = True
    print(f"DEBUG: Set job_id in session: {job_id} for user {user_google_id}")

    # Store job_id in Redis as well as a backup
    redis_client.set(f'user:{user_google_id}:current_job_id', job_id, ex=3600) # Link job to user, add expiry

    # Store processing parameters in Redis
    redis_client.hset(f"{job_id}:params", mapping={
        "user_google_id": user_google_id,
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
        # Attempt to find job_id based on logged-in user if possible (less reliable for background)
        if 'user_info' in session and session['user_info'].get('google_id'):
            user_google_id = session['user_info']['google_id']
            try:
                job_id_bytes = redis_client.get(f'user:{user_google_id}:current_job_id')
                if job_id_bytes:
                    job_id = job_id_bytes.decode('utf-8')
                    print(f"DEBUG start_processing: Found job_id {job_id} via user {user_google_id} from Redis.")
            except Exception as e:
                print(f"DEBUG start_processing: Error fetching job_id for user {user_google_id} from Redis: {e}")

    
    if not job_id:
        return jsonify({'status': 'error', 'message': 'No job ID found'})
    
    # Check if user is authenticated
    if 'credentials' not in session:
        update_progress(job_id, 'error', 0, 0, error='Not authenticated with Google. Please log in again.')
        return jsonify({'status': 'error', 'message': 'Not authenticated with Google'})
    
    # Get parameters from Redis
    params = redis_client.hgetall(f"{job_id}:params")
    user_google_id = params.get(b'user_google_id') # <<--- GET google_id
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
        emails_saved_count = 0

        for i, message_info in enumerate(messages):
            # Check if processing has been stopped
            current_status = redis_client.hget(job_id, 'status')
            if current_status and current_status.decode('utf-8') == 'stopped':
                print(f"DEBUG: Processing stopped by user for job {job_id}")
                update_progress(job_id, 'stopped', processed_count, len(messages), error='Process stopped by user')
                return jsonify({
                    'status': 'stopped',
                    'processed_count': processed_count,
                    'included_count': included_count,
                    'sheet_url': sheet_url,
                    'emails_saved_db': emails_saved_count
                })

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
                date_received_obj = None
                date_received_str = "Unknown Date" # Default for sheet/logging if parsing fails
                if date_str:
                    try:
                        date_received_obj = parsedate_to_datetime(date_str)
                        date_received_str = date_received_obj.strftime('%Y-%m-%d') # Format for sheet
                    except Exception as date_err:
                        logging.warning(f"Could not parse date string '{date_str}' for email {msg_id}: {date_err}")
                        # date_received_obj remains None

                body = get_email_body(payload)
                body_snippet_to_save = (body[:1000] + '...') if body and len(body) > 1000 else body

                # <<< ADDED LOGGING >>>
                logging.debug(f"Email {msg_id}: Subject='{subject}', DateStr='{date_received_str}'")

                if not body and not subject:
                    logging.debug(f"Skipping email {msg_id} with no body and subject")
                    continue

                update_progress(job_id, 'analyzing', processed_count, len(messages), current_email=subject)

                extracted_info = extract_job_info_with_gemini(GEMINI_API_KEY, subject, body, date_received_str) # Pass formatted string

                logging.debug(f"Email {msg_id}: Gemini Extracted Info = {extracted_info}")

                if extracted_info and "Status" in extracted_info and "LLM Error" not in extracted_info.get("Status"):
                    status_val = extracted_info.get('Status', 'Unable to Determine')

                    status_match = status_val in included_statuses
                    logging.debug(f"Email {msg_id}: Status='{status_val}', IncludedStatuses={included_statuses}, Match={status_match}")

                    if status_match:
                        # --- Attempt to SAVE TO DATABASE ---
                        try:
                            logging.debug(f"Email {msg_id}: Checking DB for user {user_google_id}, message {msg_id}")

                            existing_saved_email = SavedEmail.query.filter_by(
                                user_google_id=user_google_id,
                                gmail_message_id=msg_id
                            ).first()

                            if not existing_saved_email:
                                logging.debug(f"Email {msg_id}: Attempting to save to DB.")
                                new_saved_email = SavedEmail(
                                    id=None,
                                    user_google_id=user_google_id,
                                    gmail_message_id=msg_id,
                                    subject=subject,
                                    sender=sender,
                                    date_received=date_received_obj.date() if date_received_obj else None, # Save date object
                                    body_snippet=body_snippet_to_save,
                                    extracted_company=extracted_info.get('Company Name', 'Not Found'),
                                    extracted_job_title=extracted_info.get('Job Title', 'Not Found'),
                                    extracted_status=status_val
                                )

                                db.session.add(new_saved_email)
                                db.session.commit()
                                emails_saved_count += 1
                                logging.info(f"Successfully saved email {msg_id} to database for user {user_google_id}.") # Changed to info
                            else:
                                logging.debug(f"Email {msg_id}: Already exists in database, skipping DB save.")

                        except Exception as db_err:
                            db.session.rollback()
                            # <<< MODIFIED LOGGING >>>
                            error_msg = f"DATABASE ERROR saving email {msg_id} for user {user_google_id}: {str(db_err)}"
                            errors.append(error_msg)
                            logging.error(error_msg, exc_info=True) # Log full traceback for DB errors
                            redis_client.lpush(f"{job_id}:errors", error_msg)
                            # Continue to next email even if DB save fails

                        # --- Prepare data and add to Google Sheet ---
                        # Correctly define the record *once* for the sheet
                        sheet_record = {
                            'Date Received': date_received_str, # Use formatted string
                            'Company Name': extracted_info.get('Company Name', 'Not Found'),
                            'Job Title': extracted_info.get('Job Title', 'Not Found'),
                            'Status': status_val,
                            'Email Subject': subject,
                            'Sender': sender,
                            'Email ID': msg_id
                        }

                        # Store result in Redis for live updates on /results page
                        result_key = f"{job_id}:result:{included_count}" # included_count tracks items added to sheet/redis results
                        try:
                            # Ensure all values going into Redis are strings
                            redis_record = {k: str(v) for k, v in sheet_record.items()}
                            redis_client.hset(result_key, mapping=redis_record)
                        except Exception as redis_err:
                            logging.warning(f"Could not set Redis result key {result_key}: {redis_err}")

                        included_count += 1 # Increment count for items matching criteria
                        redis_client.hset(job_id, 'included', str(included_count)) # Update sheet/redis results count

                        update_progress(job_id, 'writing', processed_count, len(messages), included=included_count)

                        # Append to Google Sheet
                        headers_to_append = ['Date Received', 'Company Name', 'Job Title', 'Status', 'Email Subject', 'Sender', 'Email ID']
                        # Ensure order matches headers
                        row_to_append = [sheet_record.get(h, '') for h in headers_to_append]
                        try:
                            worksheet.append_row(row_to_append, value_input_option='USER_ENTERED')
                            logging.debug(f"Added row to sheet for email {msg_id}")
                        except Exception as sheet_err:
                            error_msg = f"SHEET ERROR writing row for email {msg_id}: {str(sheet_err)}"
                            errors.append(error_msg)
                            logging.error(error_msg, exc_info=True) # Log full traceback
                            redis_client.lpush(f"{job_id}:errors", error_msg)
                    # End of 'if status_match:'
                # End of 'if extracted_info:'
                else:
                    logging.warning(f"Email {msg_id}: Skipping save/sheet add due to missing/invalid Gemini data: {extracted_info}")


                time.sleep(1.1) # Rate limit Gemini calls

            except Exception as e:
                error_msg = f"PROCESSING ERROR for email {msg_id}: {str(e)}"
                errors.append(error_msg)
                logging.error(error_msg, exc_info=True) # Log full traceback
                redis_client.lpush(f"{job_id}:errors", error_msg)
                # Update progress even if one email fails catastrophically
                update_progress(job_id, 'error_processing', processed_count, len(messages), error=f"Error on email {msg_id}")
                continue # Move to the next email

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
    update_progress(job_id, 'complete', processed_count, len(messages) if 'messages' in locals() else 0, included=included_count) # included_count is still relevant for sheet
    
    redis_client.hset(job_id, 'emails_saved_db', str(emails_saved_count))

    return jsonify({
        'status': 'complete',
        'processed_count': processed_count,
        'included_count': included_count,
        'sheet_url': sheet_url,
        'emails_saved_db': emails_saved_count
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

@app.route('/saved_emails')
def saved_emails():
    if 'user_info' not in session or not session['user_info'].get('google_id'):
        flash("Please log in to view or edit your profile.", "warning")
        return redirect(url_for('login'))
    
    user_google_id = session["user_info"]["google_id"]
    saved_emails = SavedEmail.query.filter_by(user_google_id=user_google_id).order_by(SavedEmail.saved_at.desc()).all()
    
    return render_template("saved_emails.html", saved_emails=saved_emails)

@app.route('/remove_email/<int:email_db_id>', methods=['POST'])
def remove_email(email_db_id):
    if "user_info" not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    try:
        saved_email = SavedEmail.query.get_or_404(email_db_id)
        if saved_email.user_google_id != session["user_info"]["google_id"]:
            return jsonify({'success': False, 'message': 'Unauthorized'}), 403
        
        # Delete from Gmail if possible
        try:
            credentials = get_google_credentials()
            if credentials:
                service = build_google_service('gmail', 'v1')
                service.users().messages().delete(userId='me', id=saved_email.gmail_message_id).execute()
        except Exception as gmail_err:
            logging.error(f"Error deleting email from Gmail: {gmail_err}")
            # Continue with database deletion even if Gmail deletion fails
            # This ensures the email is at least removed from our database

        # Delete from database
        db.session.delete(saved_email)
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error deleting email: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/delete_email/<email_id>', methods=['DELETE'])
def delete_email(email_id):
    if "user_info" not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    try:
        saved_email = SavedEmail.query.filter_by(gmail_message_id=email_id).first()
        if not saved_email:
            return jsonify({'success': False, 'message': 'Email not found'}), 404
        
        if saved_email.user_google_id != session["user_info"]["google_id"]:
            return jsonify({'success': False, 'message': 'Unauthorized'}), 403
        
        # Delete from Gmail if possible
        try:
            credentials = get_google_credentials()
            if credentials:
                service = build_google_service('gmail', 'v1')
                service.users().messages().delete(userId='me', id=email_id).execute()
        except Exception as gmail_err:
            logging.error(f"Error deleting email from Gmail: {gmail_err}")
            # Continue with database deletion even if Gmail deletion fails
            # This ensures the email is at least removed from our database

        # Delete from database
        db.session.delete(saved_email)
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error deleting email: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route("/chat")
def chat():
    if 'user_info' not in session or not session['user_info'].get('google_id'):
        flash("Please log in to view or edit your profile.", "warning")
        return redirect(url_for('login'))

    user_google_id = session["user_info"]["google_id"]
    conversations = Conversation.query.filter_by(user_google_id=user_google_id).order_by(Conversation.updated_at.desc()).all()
    
    conversation_id = request.args.get("conversation_id")
    messages = []
    
    # Create a new conversation if none exist
    if not conversations:
        new_conversation = Conversation(
            user_google_id=user_google_id,
            title="New Conversation"
        )
        db.session.add(new_conversation)
        db.session.commit()
        conversation_id = new_conversation.id
        conversations = [new_conversation]
    
    # If conversation_id is provided, get messages for that conversation
    if conversation_id:
        messages = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.created_at.asc()).all()
    # If no conversation_id but conversations exist, use the first one
    elif conversations:
        conversation_id = conversations[0].id
        messages = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.created_at.asc()).all()
    
    return render_template(
        "chat.html",
        conversations=conversations,
        messages=messages,
        current_conversation_id=conversation_id
    )

@app.route("/chat_with_ai", methods=["POST"])
def chat_with_ai():
    if "user_info" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400

        user_google_id = session["user_info"]["google_id"]
        message_content = data["message"]
        conversation_id = data.get("conversation_id")

        # Get or create conversation
        if conversation_id:
            conversation = Conversation.query.get(conversation_id)
            if not conversation or conversation.user_google_id != user_google_id:
                print("Invalid conversation")
                return jsonify({"error": "Invalid conversation"}), 400
        else:
            conversation = Conversation(
                user_google_id=user_google_id,
                title=message_content[:30] + "..." if len(message_content) > 30 else message_content
            )
            db.session.add(conversation)
            db.session.flush()  # Get the conversation ID without committing

        # Save user message
        user_message = Message(
            conversation_id=conversation.id,
            content=message_content,
            is_user=True
        )
        db.session.add(user_message)
        # Get Gemini API key from session   
        print(session)
        gemini_api_key = GEMINI_API_KEY
        if not gemini_api_key:
            return jsonify({"error": "Gemini API key not found"}), 400
        print("Gemini API key found")
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Get user profile context
        user_profile = UserProfile.query.filter_by(google_id=user_google_id).first()
        context = ""
        if user_profile and user_profile.extracted_resume_json:
            resume_data = json.loads(user_profile.extracted_resume_json)
            skills = resume_data.get("skills", [])
            work_experience = resume_data.get("work_experience", [])
            education = resume_data.get("education", [])

            context = f"""
            User Profile:
            - Skills: {', '.join(skills)}
            - Work Experience: {len(work_experience)} positions
            - Education: {len(education)} entries
            """
        # Generate response, Build prompt with context
        prompt = f"""
            You are an Expert AI Career Strategist & Consultant. Your mission is to provide personalized, insightful, and actionable guidance to empower users in their job search and career development. Adopt a professional, empathetic, and encouraging tone.

            SYSTEM CONTEXT:
            {context}

            USER MESSAGE:
            "{message_content}"

            YOUR TASK:
            Based on the user's message and available context, provide a concise and helpful response. Your advice MUST be:

            1. Directly Relevant: Address the user's specific query or implied need. If the query is vague, offer initial general advice and then ask 1-2 targeted clarifying questions.

            2. Specific & Actionable:
               - Avoid generic platitudes
               - Provide concrete steps and examples
               - Explain how to use recommended resources

            3. Insightful & Strategic:
               - Offer deeper insights about the job market
               - Help think strategically about options and goals

            4. Focus on Key Areas (as applicable):
               - Job Search Strategies
               - Resume/CV & Cover Letter Optimization
               - Interview Preparation
               - Career Development

            FORMATTING RULES:
            - Keep responses concise and to the point
            - Use bullet points with "-" instead of "*"
            - Use numbered lists for steps
            - Use clear section headers
            - Avoid markdown formatting
            - Use line breaks for readability
            - Keep paragraphs short (2-3 sentences max)

            OUTPUT STRUCTURE:
            1. Brief acknowledgment of the user's situation
            2. Main advice in clear, actionable points
            3. Supporting details if needed
            4. Encouraging conclusion or next steps

            Remember: Be concise, clear, and practical. Focus on actionable advice rather than lengthy explanations.
            """

        response = model.generate_content(prompt)
        response_text = response.text

        # Save AI response
        ai_message = Message(
            conversation_id=conversation.id,
            content=response_text,
            is_user=False
        )
        db.session.add(ai_message)

        # Update conversation timestamp
        conversation.updated_at = datetime.utcnow()
        db.session.commit()

        return jsonify({
            "response": response_text,
            "conversation_id": conversation.id
        })

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON"}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route("/delete_conversation/<int:conversation_id>", methods=["DELETE"])
def delete_conversation(conversation_id):
    if "user_info" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    try:
        user_google_id = session["user_info"]["google_id"]
        conversation = Conversation.query.get(conversation_id)

        if not conversation or conversation.user_google_id != user_google_id:
            return jsonify({"error": "Invalid conversation"}), 400

        # Delete all messages in the conversation
        Message.query.filter_by(conversation_id=conversation_id).delete()
        
        # Delete the conversation
        db.session.delete(conversation)
        db.session.commit()

        return jsonify({"success": True})

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    """Stops the current email processing job."""
    if 'user_info' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    job_id = request.json.get('job_id')
    if not job_id:
        return jsonify({'success': False, 'message': 'No job ID provided'}), 400
    
    try:
        # Update the job status in Redis to 'stopped'
        redis_client.hset(job_id, 'status', 'stopped')
        redis_client.hset(job_id, 'error', 'Process stopped by user')
        
        # Clear the job ID from session
        session.pop('job_id', None)
        session.modified = True
        
        return jsonify({'success': True, 'message': 'Processing stopped successfully'})
    except Exception as e:
        logging.error(f"Error stopping processing: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# --- Run the App ---
if __name__ == '__main__':
    with app.app_context():
        # Create all database tables
        db.create_all()
    app.run(host='127.0.0.1', port=5000, debug=True)
