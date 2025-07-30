from flask import Flask, render_template
from email_handler import start_email_monitoring
from text_analysis import TextAnalyzer
import threading
import logging

# Reduce Flask's verbose logging
logging.getLogger('werkzeug').setLevel(logging.WARNING)

app = Flask(__name__)

@app.route('/')
def home():
    """Serve the main homepage"""
    return render_template('home.html')

@app.route('/community')
def community():
    """Serve the community submissions page"""
    return render_template('community.html')

if __name__ == "__main__":
    print("🚀 Starting SecuriVoice")
    print("📝 Make sure your .env file is configured")
    
    # Initialize text analysis database (quietly)
    print("💾 Initializing database...")
    try:
        analyzer = TextAnalyzer()
        print("✅ Database ready")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        print("❌ Exiting...")
        exit(1)
    
    print("📧 Starting email monitoring...")
    
    # Start email monitoring in background thread
    email_thread = threading.Thread(target=start_email_monitoring, daemon=True)
    email_thread.start()
    
    print("✅ Email monitoring started")
    print("🌐 Homepage: http://localhost:5000")
    print("🤝 Community: http://localhost:5000/community")
    print("-" * 40)
    
    # Start Flask with minimal logging
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)