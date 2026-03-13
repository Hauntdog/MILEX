import requests
import logging

def resolve_ip(ip):
    """
    Resolves an IP address using ip-api.com (free, non-commercial).
    """
    if ip in (None, '127.0.0.1', '::1', 'localhost'):
        return None
    
    try:
        url = f"http://ip-api.com/json/{ip}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'success':
            return {
                'city': data.get('city', 'Unknown'),
                'country': data.get('country', 'Unknown'),
                'lat': data.get('lat', 0.0),
                'lon': data.get('lon', 0.0),
                'isp': data.get('isp', 'Unknown')
            }
        else:
            logging.error(f"GeoIP resolution failed for {ip}: {data.get('message')}")
            return None
    except Exception as e:
        logging.error(f"GeoIP API error for {ip}: {str(e)}")
        return None

if __name__ == "__main__":
    test_ip = "8.8.8.8"
    print(f"Resolving {test_ip}...")
    print(resolve_ip(test_ip))
