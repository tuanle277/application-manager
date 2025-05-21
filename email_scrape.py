import os
import base64
import email
from email.utils import parsedate_to_datetime
import re
import imaplib
from email.header import decode_header

import google.generativeai as genai
import pandas as pd
from datetime import datetime
import time # For potential rate limiting delays
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import print as rprint
import sys
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe, get_as_dataframe
import dotenv

dotenv.load_dotenv()

# --- Configuration ---
# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 
          'https://www.googleapis.com/auth/spreadsheets',
          'https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = 'credentials.json' # Downloaded from Google Cloud Console
TOKEN_FILE = 'token.json'
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') # Replace with your actual key
GOOGLE_SHEET_NAME = 'job_application_tracker'
# Keywords to initially filter emails (adjust as needed)
EMAIL_SEARCH_QUERY = '(SUBJECT "application") OR (SUBJECT "interview") OR (SUBJECT "offer") OR (SUBJECT "rejection") OR (SUBJECT "assessment") OR (SUBJECT "keep in touch") OR (SUBJECT "thank you") OR (SUBJECT "thank you for applying") OR (SUBJECT "position") OR (SUBJECT "regret") OR (SUBJECT "unfortunately")'
# Maximum emails to process (set to None to process all matches, use a small number for testing)
MAX_EMAILS_TO_PROCESS = 1000
# Gemini Model Name
GEMINI_MODEL = 'gemini-2.0-flash' # Or another suitable model
# IMAP Configuration
IMAP_SERVER = 'imap.gmail.com'
EMAIL_ADDRESS = 'mtuan.le2024@gmail.com'
EMAIL_PASSWORD = 'sovj jzrt glis wubm'  # Use app password for Gmail
# Status filter - only include these statuses
INCLUDED_STATUSES = ["Interview Request", "Assessment Request", "Offer", "Rejection"]

# Initialize Rich console
console = Console()

# --- Gemini API Configuration ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    console.print(f"[bold green]✓[/bold green] Successfully configured Gemini model: [cyan]{GEMINI_MODEL}[/cyan]")
except Exception as e:
    console.print(f"[bold red]✗[/bold red] Error configuring Gemini: {e}", style="bold red")
    sys.exit(1)

# --- Google Sheets Configuration ---
try:
    creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
    gc = gspread.authorize(creds)
    console.print(f"[bold green]✓[/bold green] Successfully authenticated with Google Sheets API")
except Exception as e:
    console.print(f"[bold red]✗[/bold red] Error authenticating with Google Sheets: {e}", style="bold red")
    sys.exit(1)

# --- IMAP Connection ---
def get_imap_connection():
    """Connect to IMAP server and login."""
    try:
        with console.status("[bold blue]Connecting to IMAP server...[/bold blue]"):
            mail = imaplib.IMAP4_SSL(IMAP_SERVER)
            mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        console.print("[bold green]✓[/bold green] IMAP connection established successfully.")
        return mail
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Error connecting to IMAP server: {e}", style="bold red")
        return None

# --- Email Processing ---
def get_email_body(msg):
    """Extracts the text body from an email message."""
    body = ""
    
    if msg.is_multipart():
        # Handle multipart messages
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            
            # Skip attachments
            if "attachment" in content_disposition:
                continue
                
            # Get plain text content
            if content_type == "text/plain":
                try:
                    body = part.get_payload(decode=True).decode('utf-8', errors='replace')
                    return body.strip()
                except Exception as e:
                    console.print(f"[yellow]Warning:[/yellow] Error decoding email body: {e}")
    else:
        # Handle non-multipart messages
        try:
            body = msg.get_payload(decode=True).decode('utf-8', errors='replace')
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Error decoding email body: {e}")
    
    return body.strip()

def decode_email_subject(subject):
    """Decode email subject that might be encoded."""
    if subject is None:
        return "No Subject"
        
    decoded_parts = []
    for part, encoding in decode_header(subject):
        if isinstance(part, bytes):
            if encoding:
                try:
                    decoded_parts.append(part.decode(encoding))
                except:
                    decoded_parts.append(part.decode('utf-8', errors='replace'))
            else:
                decoded_parts.append(part.decode('utf-8', errors='replace'))
        else:
            decoded_parts.append(part)
    
    return ''.join(decoded_parts)

# --- LLM Interaction ---
def extract_job_info_with_gemini(subject, body, date_received):
    """Uses Gemini to extract job application details."""
    if not body and not subject:
        return None # Skip if no content

    # Limit body length to avoid excessive API usage/cost
    max_body_length = 12000 # Adjust as needed
    truncated_body = body if body else ""

    prompt = f"""
        **Objective:** Analyze the provided email content (Subject and Body) related to a job application. Your goal is to accurately extract the Company Name, Job Title, and classify the email's status based on its primary purpose.

        **Instructions:**

        1.  **Read Thoroughly:** Carefully examine the *entire* provided email content, including the Subject line and the full Body text. Pay attention to keywords, phrases, and the overall context. Explain your thought process to yourself and argue with yourself, give an explaination on why you chose the status.
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
            Reasoning: [Your internal reasoning process]
            ```

        --- Email Content ---
        Date Received: {date_received}
        Subject: {subject}
        Body:
        {truncated_body}
        --- End Email Content ---

        Output:
    """

    # Add safety settings if desired (optional)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    try:
        response = gemini_model.generate_content(
            prompt,
            # safety_settings=safety_settings # Uncomment to enable
            )
        # Basic parsing - assumes the LLM follows the format strictly
        extracted_data = {"Company Name": "Error", "Job Title": "Error", "Status": "Error", "Reasoning": "Error"}
        lines = response.text.strip().split('\n')
        for line in lines:
            if ':' in line:
                 key, value = line.split(':', 1)
                 key = key.strip()
                 value = value.strip()
                 if key in extracted_data:
                     extracted_data[key] = value
        return extracted_data

    except Exception as e:
        console.print(f"[bold red]Error calling Gemini API:[/bold red] {e}")
        # Check for specific errors like blocked prompts
        if hasattr(e, 'response') and e.response.prompt_feedback.block_reason:
             console.print(f"[bold red]Prompt blocked:[/bold red] {e.response.prompt_feedback.block_reason}")
        return {"Company Name": "LLM Error", "Job Title": "LLM Error", "Status": "LLM Error", "Reasoning": "LLM Error"}

# --- Google Sheets Functions ---
def initialize_google_sheet():
    """Create or load the Google Sheet with proper headers"""
    try:
        # Try to open existing sheet
        try:
            sheet = gc.open(GOOGLE_SHEET_NAME)
            console.print(f"[green]Using existing Google Sheet:[/green] {GOOGLE_SHEET_NAME}")

        except gspread.exceptions.SpreadsheetNotFound:
            # Create new sheet if it doesn't exist
            sheet = gc.create(GOOGLE_SHEET_NAME)
            console.print(f"[green]Created new Google Sheet:[/green] {GOOGLE_SHEET_NAME}")
            sheet.share('mtuan.le2024@gmail.com', perm_type='user', role='writer')
            console.print(f"[green]Shared with mtuan.le2024@gmail.com[/green]")
            
        # Get or create the first worksheet
        try:
            worksheet = sheet.worksheet("Job Applications")
        except gspread.exceptions.WorksheetNotFound:
            worksheet = sheet.add_worksheet(title="Job Applications", rows=1000, cols=10)

        # Set headers if sheet is empty
        if worksheet.row_count <= 1:
            headers = ['Date Received', 'Company Name', 'Job Title', 'Status', 'Reasoning', 'Email Subject', 'Sender', 'Email ID']
            worksheet.append_row(headers)
            
        # After creating the sheet, add:
        console.print(f"[green]Spreadsheet URL:[/green] {sheet.url}")
        console.print(f"[green]Spreadsheet ID:[/green] {sheet.id}")
        
        return sheet, worksheet
    except Exception as e:
        console.print(f"[bold red]Error initializing Google Sheet:[/bold red] {e}")
        return None, None

def append_to_google_sheet(worksheet, record):
    """Append a single record to the Google Sheet"""
    try:
        # Convert record values to list in the same order as headers
        row = [
            record['Date Received'],
            record['Company Name'],
            record['Job Title'],
            record['Status'],
            record['Reasoning'],
            record['Email Subject'],
            record['Sender'],
            record['Email ID']
        ]
        worksheet.append_row(row)
        return True
    except Exception as e:
        console.print(f"[bold red]Error appending to Google Sheet:[/bold red] {e}")
        return False

# --- Main Execution ---
def main():
    # Initialize Google Sheet
    sheet, worksheet = initialize_google_sheet()
    if not worksheet:
        return
    
    mail = get_imap_connection()
    if not mail:
        return

    all_job_data = []
    processed_count = 0
    included_count = 0

    try:
        # Select the mailbox (inbox)
        mail.select('INBOX')
        
        with console.status("[bold blue]Searching for emails...[/bold blue]") as status:
            console.print(f"[bold]Search query:[/bold] {EMAIL_SEARCH_QUERY}")
            try:
                # Fix: Use proper IMAP search syntax with OR operators
                search_criteria = []
                search_terms = [
                    'SUBJECT "application"',
                    'SUBJECT "interview"',
                    'SUBJECT "offer"',
                    'SUBJECT "rejection"',
                    'SUBJECT "assessment"',
                    'SUBJECT "keep in touch"',
                    'SUBJECT "thank you"',
                    'SUBJECT "thank you for applying"',
                    'SUBJECT "position"',
                    'SUBJECT "regret"',
                    'SUBJECT "unfortunately"'
                ]
                
                # IMAP doesn't support complex OR queries directly, so we'll do multiple searches
                all_email_ids = set()
                for term in search_terms:
                    status, messages = mail.search(None, term)
                    if status == 'OK' and messages[0]:
                        ids = messages[0].split()
                        all_email_ids.update(ids)
                
                # Convert set back to the format expected by the rest of the code
                email_ids = list(all_email_ids)
                
            except Exception as e:
                console.print(f"[yellow]Error with complex search, trying simpler approach:[/yellow] {e}")
                # Fallback to a simpler search if the complex one fails
                status, messages = mail.search(None, 'ALL')
                email_ids = messages[0].split() if status == 'OK' else []
                
            if not email_ids:
                console.print("[yellow]No emails found matching the search criteria[/yellow]")
                return
        
        console.print(f"[bold green]Found {len(email_ids)} emails[/bold green] matching the search criteria")
        console.print(f"[bold blue]Will only include emails with status:[/bold blue] {', '.join(INCLUDED_STATUSES)}")
        
        # Create a table for results
        results_table = Table(title="Job Application Email Analysis")
        results_table.add_column("Date", style="cyan")
        results_table.add_column("Company", style="green")
        results_table.add_column("Job Title", style="blue")
        results_table.add_column("Status", style="magenta")
        results_table.add_column("Reasoning", style="yellow")
        # Process emails with progress bar
        email_ids_to_process = list(reversed(email_ids))
        # Fix: Remove the min() calculation that might be causing the error
        max_to_process = MAX_EMAILS_TO_PROCESS if MAX_EMAILS_TO_PROCESS else len(email_ids_to_process)
        email_ids_to_process = email_ids_to_process[:max_to_process]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Processing emails...", total=len(email_ids_to_process))
            
            for i, email_id in enumerate(email_ids_to_process):
                # if MAX_EMAILS_TO_PROCESS is not None and processed_count >= MAX_EMAILS_TO_PROCESS:
                #     break
                    
                try:
                    # Fetch email by ID
                    status, msg_data = mail.fetch(email_id, '(RFC822)')
                    
                    if status != 'OK':
                        console.print(f"[yellow]Error fetching email ID: {email_id.decode()}[/yellow]")
                        continue
                        
                    # Parse the email
                    raw_email = msg_data[0][1]
                    msg = email.message_from_bytes(raw_email)
                    
                    # Extract email details
                    subject = decode_email_subject(msg['Subject'])
                    sender = msg['From']
                    
                    # Parse date
                    date_str = msg['Date']
                    date_received = "Unknown Date"
                    if date_str:
                        try:
                            dt_obj = parsedate_to_datetime(date_str)
                            date_received = dt_obj.strftime('%Y-%m-%d')
                        except Exception:
                            pass
                    
                    # Extract email body
                    body = get_email_body(msg)
                    
                    progress.update(task, description=f"[cyan]Processing email {processed_count+1}/{len(email_ids_to_process)}: {subject[:30]}...[/cyan]")
                    
                    if not body and not subject:
                        continue
                    
                    # Use Gemini to extract information
                    extracted_info = extract_job_info_with_gemini(subject, body, date_received)
                    
                    if extracted_info:
                        status = extracted_info.get('Status', 'Not Found')
                        
                        # Only include emails with specified statuses
                        if status in INCLUDED_STATUSES:
                            record = {
                                'Date Received': date_received,
                                'Company Name': extracted_info.get('Company Name', 'Not Found'),
                                'Job Title': extracted_info.get('Job Title', 'Not Found'),
                                'Status': status,
                                'Reasoning': extracted_info.get('Reasoning', 'Not Found'),
                                'Email Subject': subject,
                                'Sender': sender,
                                'Email ID': email_id.decode()
                            }
                            all_job_data.append(record)
                            
                            # Add to results table
                            results_table.add_row(
                                date_received,
                                record['Company Name'],
                                record['Job Title'],
                                record['Status']
                            )
                            
                            # Append to Google Sheet as we go
                            append_to_google_sheet(worksheet, record)
                            included_count += 1
                    
                    processed_count += 1
                    progress.update(task, advance=1)
                    time.sleep(0.5)  # Add a small delay to avoid hitting API rate limits
                    
                except Exception as e:
                    console.print(f"[bold red]Error processing email ID {email_id.decode()}:[/bold red] {e}")
        
        console.print(f"\n[bold green]✓[/bold green] Finished processing [bold]{processed_count}[/bold] emails.")
        console.print(f"[bold green]✓[/bold green] Added [bold]{included_count}[/bold] relevant emails to Google Sheet.")
        
        # Display results table
        console.print(results_table)
        
        # Close the connection
        mail.close()
        mail.logout()
        
        console.print(f"[bold green]✓[/bold green] Data exported to Google Sheet: [cyan]{GOOGLE_SHEET_NAME}[/cyan]")

    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred in main execution:[/bold red] {e}")
        if mail:
            try:
                mail.close()
                mail.logout()
            except:
                pass


if __name__ == '__main__':
    console.print("[bold blue]===== Job Application Email Analyzer =====[/bold blue]")
    main()