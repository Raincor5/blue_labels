# filename: web_server.py
import http.server
import socketserver
import os
import webbrowser
import threading
import time
from pathlib import Path
import ssl
import shutil
import subprocess
import socket

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

def _mkcert_available() -> bool:
    return bool(shutil.which("mkcert") or shutil.which("mkcert.exe"))

def _mkcert_generate(cert_file: Path, key_file: Path, names: list[str]) -> tuple[str, str]:
    # Try install CA (idempotent)
    try:
        subprocess.run(["mkcert", "-install"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        pass
    cmd = ["mkcert", "-cert-file", str(cert_file), "-key-file", str(key_file)] + names
    res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(res.stdout.decode(errors="ignore").strip())
    return str(cert_file), str(key_file)

def start_web_server(port=8080, directory=None, use_https=False, host="127.0.0.1", cert_file: str | None = None, key_file: str | None = None):
    """Start a simple HTTP server to serve the web client"""
    
    if directory is None:
        directory = Path(__file__).parent
    
    # Change to the directory containing the web client
    os.chdir(directory)

    handler = CustomHTTPRequestHandler
    
    # Create the server
    with socketserver.TCPServer((host, port), handler) as httpd:
        if use_https:
            # Resolve certificate files
            if not cert_file or not key_file:
                default_dir = Path("ssl_certs")
                default_dir.mkdir(exist_ok=True)
                default_cert = default_dir / "server.crt"
                default_key = default_dir / "server.key"
                if default_cert.exists() and default_key.exists():
                    cert_file = str(default_cert)
                    key_file = str(default_key)
                else:
                    print("No certificate provided. Attempting mkcert...")
                    if _mkcert_available():
                        names = [host, "localhost"]
                        try:
                            cert_file, key_file = _mkcert_generate(default_cert, default_key, names)
                            print("✅ Generated mkcert certificates.")
                        except Exception as e:
                            print(f"⚠️ mkcert failed: {e}. Falling back to HTTP.")
                            use_https = False
                    else:
                        print("⚠️ mkcert not available and no certs found. Falling back to HTTP.")
                        use_https = False
            if use_https and cert_file and key_file:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                context.load_cert_chain(certfile=cert_file, keyfile=key_file)
                httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
                print(f"🔒 HTTPS enabled on port {port}")
        else:
            print(f"🌐 HTTP running on port {port}")

        print(f"Serving from {directory}")
        if host == "127.0.0.1":
            print(f"🔒 Localhost only: {'https' if use_https else 'http'}://localhost:{port}")
        else:
            print(f"🌐 Network accessible: {'https' if use_https else 'http'}://{host}:{port}")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down web server...")
            httpd.shutdown()

def get_local_ip():
    """Get the local IP address"""
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
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1 for localhost only)")
    parser.add_argument("--directory", help="Directory to serve files from (default: current directory)")
    parser.add_argument("--create-client", action="store_true", help="Create placeholder web client file")
    parser.add_argument("--https", action="store_true", help="Enable HTTPS")
    parser.add_argument("--use-mkcert", action="store_true", help="Use mkcert to auto-generate local trusted certs")
    parser.add_argument("--cert-file", help="Path to SSL certificate file")
    parser.add_argument("--key-file", help="Path to SSL private key file")
    parser.add_argument("--cert-dir", default="ssl_certs", help="Directory to store or look for certs")
    
    args = parser.parse_args()
    
    if args.create_client:
        create_web_client_file()
        return
    
    try:
        cert_file = args.cert_file
        key_file = args.key_file
        if args.https and not (cert_file and key_file):
            # Look in cert-dir or try mkcert if requested
            cert_dir = Path(args.cert_dir)
            default_cert = cert_dir / "server.crt"
            default_key = cert_dir / "server.key"
            if default_cert.exists() and default_key.exists():
                cert_file, key_file = str(default_cert), str(default_key)
            elif args.use_mkcert:
                try:
                    cert_dir.mkdir(exist_ok=True)
                    cert_file, key_file = _mkcert_generate(default_cert, default_key, [args.host, "localhost", get_local_ip(), "127.0.0.1"])
                    print("✅ Generated mkcert certificates in", cert_dir)
                except Exception as e:
                    print(f"⚠️ mkcert failed: {e}. Serving HTTP.")
                    args.https = False
        start_web_server(args.port, args.directory, use_https=args.https, host=args.host, cert_file=cert_file, key_file=key_file)
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