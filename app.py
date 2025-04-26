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
from email.header import decode_header
import time
import gspread
import pandas as pd
from werkzeug.utils import secure_filename # For potential future file uploads
import dotenv
import uuid
import redis

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
# For local testing, you might use a .env file and python-dotenv
# DO NOT HARDCODE THESE HERE IN A REAL APPLICATION
CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "super-secret-key-for-dev-only") # Change for production!
GEMINI_API_KEY_SESSION_KEY = os.environ.get("GEMINI_API_KEY") # Key to store Gemini key in session

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['SESSION_COOKIE_SECURE'] = False # Set to True if using HTTPS

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

# --- Helper Functions ---

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
            
        service = build(service_name, version, credentials=credentials)
        print(f"DEBUG: Successfully built {service_name} service")
        return service
    except Exception as e:
        print(f"DEBUG: Error building {service_name} service: {str(e)}")
        return None

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
    gemini_key_set = GEMINI_API_KEY_SESSION_KEY in session
    return render_template('index.html', user_info=user_info, gemini_key_set=gemini_key_set)

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
            session['user_info'] = user_info # Store user info
            flash(f"Successfully logged in as {user_info.get('email', 'Unknown')}", "success")
        except HttpError as e:
            flash(f"Could not fetch user info: {e}", "warning")
            session['user_info'] = {'email': 'Error fetching email'} # Store placeholder

        return redirect(url_for('index'))

    except Exception as e:
        flash(f"Error during authentication callback: {e}", "danger")
        return redirect(url_for('index'))


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
    session[GEMINI_API_KEY_SESSION_KEY] = gemini_key # Store key in session

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
                extracted_info = extract_job_info_with_gemini(GEMINI_API_KEY_SESSION_KEY, subject, body, date_received)
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
