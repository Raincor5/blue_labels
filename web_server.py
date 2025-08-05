# filename: web_server.py
import http.server
import socketserver
import os
import webbrowser
import threading
import time
from pathlib import Path

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers for development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_GET(self):
        # Serve the web client as index.html
        if self.path == '/' or self.path == '/index.html':
            self.path = '/web_client.html'
        super().do_GET()

    def log_message(self, format, *args):
        # Custom logging
        print(f"[{time.strftime('%H:%M:%S')}] {format % args}")

def start_web_server(port=8080, directory=None):
    """Start a simple HTTP server to serve the web client"""
    
    if directory is None:
        directory = Path(__file__).parent
    
    # Change to the directory containing the web client
    os.chdir(directory)
    
    # Create the server
    with socketserver.TCPServer(("", port), CustomHTTPRequestHandler) as httpd:
        print(f"🌐 Starting web server on port {port}")
        print(f"📁 Serving files from: {directory}")
        print(f"🔗 Web Client URL: http://localhost:{port}")
        print(f"🔗 Network URL: http://{get_local_ip()}:{port}")
        print("\n" + "="*50)
        print("📱 Access from any device on your network!")
        print("🖥️  Desktop: http://localhost:8080")
        print(f"📱 Mobile/Tablet: http://{get_local_ip()}:8080")
        print("="*50 + "\n")
        
        # Auto-open browser after a short delay
        def open_browser():
            time.sleep(1)
            try:
                webbrowser.open(f'http://localhost:{port}')
                print("🚀 Browser opened automatically")
            except Exception as e:
                print(f"⚠️  Could not open browser automatically: {e}")
        
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down web server...")
            httpd.shutdown()

def get_local_ip():
    """Get the local IP address"""
    import socket
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "localhost"

def create_web_client_file():
    """Create the web client HTML file if it doesn't exist"""
    web_client_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO+OCR Pipeline Web Client</title>
    <style>
        /* Include the CSS from the web client artifact here */
        /* This is a placeholder - copy the actual CSS from the artifact */
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .error { color: red; padding: 20px; border: 1px solid red; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>YOLO+OCR Web Client</h1>
        <div class="error">
            <h3>Setup Required</h3>
            <p>Please save the complete web client HTML from the artifact as "web_client.html" in the same directory as this server script.</p>
            <p>The web client artifact contains the full HTML, CSS, and JavaScript needed for the interface.</p>
        </div>
    </div>
</body>
</html>'''
    
    if not os.path.exists('web_client.html'):
        with open('web_client.html', 'w', encoding='utf-8') as f:
            f.write(web_client_content)
        print("📝 Created placeholder web_client.html - please replace with the actual client from the artifact")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Web Server for YOLO+OCR Client")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    parser.add_argument("--directory", help="Directory to serve files from (default: current directory)")
    parser.add_argument("--create-client", action="store_true", help="Create placeholder web client file")
    
    args = parser.parse_args()
    
    if args.create_client:
        create_web_client_file()
        return
    
    try:
        start_web_server(args.port, args.directory)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {args.port} is already in use. Try a different port:")
            print(f"   python web_server.py --port 8081")
        else:
            print(f"❌ Error starting server: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()