import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'attacks.db')
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'attacks.log')

def initialize_db():
    if not os.path.exists(os.path.dirname(DB_PATH)):
        os.makedirs(os.path.dirname(DB_PATH))
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ip TEXT,
            service TEXT,
            payload TEXT,
            username TEXT,
            password TEXT,
            city TEXT,
            country TEXT,
            lat REAL,
            lon REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_attack(ip, service, payload=None, username=None, password=None, geo_data=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    city = geo_data.get('city', 'Unknown') if geo_data else 'Unknown'
    country = geo_data.get('country', 'Unknown') if geo_data else 'Unknown'
    lat = geo_data.get('lat', 0.0) if geo_data else 0.0
    lon = geo_data.get('lon', 0.0) if geo_data else 0.0

    cursor.execute('''
        INSERT INTO attacks (ip, service, payload, username, password, city, country, lat, lon)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (ip, service, payload, username, password, city, country, lat, lon))
    
    conn.commit()
    conn.close()

    # --- Flat File Logging (New: Easy to read/save) ---
    with open(LOG_FILE, 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [IP: {ip}] [SVC: {service}] [GEO: {city}, {country}] [AUTH: {username}:{password}] [PAYLOAD: {payload}]\n"
        f.write(log_entry)

def get_recent_attacks(limit=50):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM attacks ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_all_attacks():
    """Returns ALL attacks for CSV export."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM attacks ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total attacks
    cursor.execute('SELECT COUNT(*) FROM attacks')
    total = cursor.fetchone()[0]
    
    # Attacks per service
    cursor.execute('SELECT service, COUNT(*) FROM attacks GROUP BY service')
    services = dict(cursor.fetchall())
    
    # Top IPs
    cursor.execute('SELECT ip, COUNT(*) as count FROM attacks GROUP BY ip ORDER BY count DESC LIMIT 5')
    top_ips = dict(cursor.fetchall())
    
    conn.close()
    return {
        'total': total,
        'services': services,
        'top_ips': top_ips
    }

def get_captured_creds():
    """Returns a list of unique captured username/password combinations."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT username, password, service, ip, timestamp 
        FROM attacks 
        WHERE username IS NOT NULL AND password IS NOT NULL 
        ORDER BY timestamp DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    initialize_db()
    print("Database initialized at:", DB_PATH)
