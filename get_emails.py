from app import app, db, SavedEmail
from flask import session
with app.app_context():
    try:
        emails = SavedEmail.query.all()
        print(f"\nTotal saved emails found: {len(emails)}\n")

        for email in emails:
            print(f"ID: {email.id}")
            print(f"User Google ID: {email.user_google_id}")
            print(f"Gmail Message ID: {email.gmail_message_id}")
            print(f"Subject: {email.subject}")
            print(f"Status: {email.extracted_status}")
            print(f"Company: {email.extracted_company}")
            print(f"Job Title: {email.extracted_job_title}")
            print(f"Date Received: {email.date_received}")
            print(f"Sender: {email.sender}")
            print("-" * 80)
            
    except Exception as e:
        print(f"Error querying database: {e}") 