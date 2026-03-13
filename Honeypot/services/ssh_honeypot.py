import socket
import sys
import os
import threading
import logging
import paramiko

# Ensure parent directory is in path for core imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import log_attack
from core.geoip import resolve_ip

# --- SSH Server Configuration ---
HOST_KEY = paramiko.RSAKey.from_private_key_file(os.path.join(os.path.dirname(__file__), 'server_key'))

class SSHServer(paramiko.ServerInterface):
    def __init__(self, client_ip):
        self.client_ip = client_ip
        self.event = threading.Event()

    def check_auth_password(self, username, password):
        # Resolve geographic data
        geo_data = resolve_ip(self.client_ip) if self.client_ip != '127.0.0.1' else None
        
        # Log the attempt
        print(f"[!] SSH Login Attempt: {self.client_ip} | {username}:{password}")
        log_attack(self.client_ip, "ssh", username=username, password=password, geo_data=geo_data)
        
        # Always fail authentication to keep them trying (or log and block)
        return paramiko.AUTH_FAILED

    def get_allowed_auths(self, username):
        return "password"

def handle_ssh_connection(client_socket, client_addr):
    ip = client_addr[0]
    # Resolve geographic data
    geo_data = resolve_ip(ip) if ip != '127.0.0.1' else None
    
    # Log the connection attempt
    print(f"[!] Incoming SSH connection from {ip}")
    log_attack(ip, "ssh-connection", payload="Connection established (waiting for auth)", geo_data=geo_data)

    try:
        transport = paramiko.Transport(client_socket)
        transport.add_server_key(HOST_KEY)
        
        server = SSHServer(ip)
        transport.start_server(server=server)

        
        # We don't need to do anything here because check_auth_password 
        # is called during the handshake and it handles the logging.
        # We'll just wait a bit and close if they don't auth.
        chan = transport.accept(20)
        if chan:
            chan.close()
    except Exception as e:
        print(f"SSH Handler Error: {str(e)}")
    finally:
        transport.close()

def run_ssh_honeypot(port=2222):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', port))
        sock.listen(100)
        
        print(f"ShadowHawk SSH Honeypot running on port {port}...")
        
        while True:
            client, addr = sock.accept()
            # Handle in a thread so the listener doesn't block
            threading.Thread(target=handle_ssh_connection, args=(client, addr), daemon=True).start()
    except Exception as e:
        print(f"SSH Listener Error: {str(e)}")

if __name__ == "__main__":
    run_ssh_honeypot()
