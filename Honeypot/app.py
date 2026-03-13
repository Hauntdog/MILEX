import sys
import os
import threading
import time
import csv
import io
from flask import Flask, render_template, jsonify, Response, request

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.logger import initialize_db, get_recent_attacks, get_stats, log_attack, get_all_attacks, get_captured_creds
from services.http_honeypot import app as http_app
from services.ssh_honeypot import run_ssh_honeypot

app = Flask(__name__)

# --- Dashboard Routes ---
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/logs')
def logs():
    return render_template('logs.html')

@app.route('/registered-admins')
def registered_admins():
    creds = get_captured_creds()
    return render_template('registered_admins.html', creds=creds)


@app.route('/api/stats')
def api_stats():
    # Fetch data from logger
    limit = int(request.args.get('limit', 50))
    recent = get_recent_attacks(limit)
    stats = get_stats()
    return jsonify({
        'recent': recent,
        'stats': stats
    })


@app.route('/export/csv')
def export_csv():
    """Generates a CSV file of all attacks."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(['ID', 'Timestamp', 'IP', 'Service', 'Payload', 'Username', 'Password', 'City', 'Country', 'Lat', 'Lon'])
    
    # Data
    rows = get_all_attacks()
    writer.writerows(rows)
    
    output.seek(0)
    return Response(
        output.read(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=honeypot_attacks.csv"}
    )

# --- Background Service Runner ---
def generate_mock_data():
    """Seeds the DB with a few initial 'attacks' for demonstration."""
    print("[*] Generating demonstration data...")
    # Delay to ensure DB is initialized
    time.sleep(1)
    # Target IPs (Common sources)
    mock_ips = [
        ("185.156.177.21", {"city": "Dublin", "country": "Ireland", "lat": 53.333, "lon": -6.248}),
        ("45.133.1.20", {"city": "Amsterdam", "country": "Netherlands", "lat": 52.374, "lon": 4.889}),
        ("193.163.125.10", {"city": "Beijing", "country": "China", "lat": 39.904, "lon": 116.407})
    ]
    for ip, geo in mock_ips:
        log_attack(ip, "http-admin-panel", payload="GET /admin", geo_data=geo)
        log_attack(ip, "ssh-brute-force", username="admin", password="password123", geo_data=geo)

def run_honeypot_services():
    print("[*] Launching ShadowHawk Decoy Services...")
    # Run the HTTP Honeypot on port 8080 (non-root)
    http_thread = threading.Thread(target=lambda: http_app.run(host='0.0.0.0', port=8090, debug=False, use_reloader=False))
    http_thread.daemon = True
    http_thread.start()
    
    # Run the SSH Honeypot on port 2222 (non-root)
    ssh_thread = threading.Thread(target=run_ssh_honeypot, args=(2222,), daemon=True)
    ssh_thread.start()
    
    # Mock data generator (one-shot)
    mock_thread = threading.Thread(target=generate_mock_data, daemon=True)
    mock_thread.start()
    
    print("[+] HTTP Honeypot active on port 8090")
    print("[+] SSH Honeypot active on port 2222")

if __name__ == "__main__":
    # 1. Initialize DB
    initialize_db()
    
    # 2. Launch Honeypots in Background
    run_honeypot_services()
    
    # 3. Start Dashboard on port 5050
    print("[*] Launching ShadowHawk Operations Center on port 5050...")
    app.run(host='0.0.0.0', port=5050, debug=False)
