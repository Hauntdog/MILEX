from flask import Flask, request, render_template_string
import os
import sys

# Ensure parent directory is in path for core imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import log_attack
from core.geoip import resolve_ip

app = Flask(__name__)

# Decoy Login Template (Modern & Professional looking)
DECOY_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Internal Maintenance - Secure Login</title>
    <style>
        body { font-family: 'Inter', system-ui; background: #0F172A; color: #E2E8F0; height: 100vh; display: flex; align-items: center; justify-content: center; margin: 0; }
        .card { background: #1E293B; padding: 2.5rem; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); width: 360px; border: 1px solid #334155; }
        h1 { font-size: 1.5rem; margin-bottom: 1.5rem; font-weight: 600; text-align: center; color: #38BDF8; }
        .form-group { margin-bottom: 1.25rem; }
        label { display: block; font-size: 0.875rem; color: #94A3B8; margin-bottom: 0.5rem; }
        input { width: 100%; padding: 0.75rem; border-radius: 6px; border: 1px solid #475569; background: #0F172A; color: white; box-sizing: border-box; }
        button { width: 100%; padding: 0.75rem; border: none; border-radius: 6px; background: #0EA5E9; color: white; font-weight: 600; cursor: pointer; transition: 0.2s; }
        button:hover { background: #38BDF8; }
        .footer { text-align: center; margin-top: 1.5rem; font-size: 0.75rem; color: #64748B; }
    </style>
</head>
<body>
    <div class="card">
        <h1>Administrative Access</h1>
        <form method="POST">
            <div class="form-group">
                <label>User ID</label>
                <input type="text" name="username" placeholder="Enter ID..." required>
            </div>
            <div class="form-group">
                <label>Security Key</label>
                <input type="password" name="password" placeholder="••••••••" required>
            </div>
            <button type="submit">Authentication Required</button>
        </form>
        <div class="footer">
            System Node: NY-EDGE-04 | v2.1.4<br>
            Unauthorized access is strictly monitored.
        </div>
    </div>
</body>
</html>
"""

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def index(path):
    # Gather request context
    ip = request.remote_addr
    # Behind proxy? (Check X-Forwarded-For if available, but for now simple)
    if 'X-Forwarded-For' in request.headers:
        ip = request.headers['X-Forwarded-For'].split(',')[0].strip()

    geo_data = resolve_ip(ip) if ip != '127.0.0.1' else None
    
    # Path requested (including query parameters)
    full_path_with_query = request.full_path
    
    if request.method == 'POST':
        user = request.form.get('username')
        pw = request.form.get('password')
        raw_body = request.get_data(as_text=True)
        payload = f"Path: {full_path_with_query} | Body: {raw_body} | Headers: {dict(request.headers)}"
        log_attack(ip, "http-web-admin", payload=payload, username=user, password=pw, geo_data=geo_data)
        return "Authentication timeout. Please contact internal IT support.", 403
    else:
        log_attack(ip, "http-web-admin", payload=f"{request.method} {full_path_with_query}", geo_data=geo_data)
        # If it's a login looking path, or root, show the decoy
        if path == "" or any(p in path.lower() for p in ['login', 'admin', 'auth', 'sign-in']):
            return render_template_string(DECOY_HTML)
        else:
            # For other things, maybe simulate a directory or just 404 but LOG it
            return f"Error: /{path} not found on server.", 404


def run_http_honeypot(port=8080):
    print(f"ShadowHawk HTTP Honeypot running on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    run_http_honeypot()
