# filename: ssl_server.py
import ssl
import uvicorn
import logging
import os
from pathlib import Path
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import datetime
import ipaddress

# Import the enhanced pipeline
try:
    from enhanced_yolo_ocr_pipeline import app, get_pipeline, EnhancedYOLOOCRPipeline
except ImportError:
    print("Error: enhanced_yolo_ocr_pipeline.py not found!")
    print("Please ensure the enhanced pipeline file is in the same directory.")
    exit(1)

logger = logging.getLogger(__name__)

class SSLCertificateManager:
    """Manages SSL certificate generation and validation"""
    
    def __init__(self, cert_dir: str = "ssl_certs"):
        self.cert_dir = Path(cert_dir)
        self.cert_dir.mkdir(exist_ok=True)
        self.cert_file = self.cert_dir / "server.crt"
        self.key_file = self.cert_dir / "server.key"
    
    def generate_self_signed_cert(self, hostname: str = "localhost", 
                                ip_addresses: list = None, 
                                days_valid: int = 365):
        """
        Generate a self-signed SSL certificate
        """
        if ip_addresses is None:
            ip_addresses = ["127.0.0.1", "::1"]
        
        logger.info(f"Generating self-signed SSL certificate for {hostname}")
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Local"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Development"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "YOLO-OCR Pipeline"),
            x509.NameAttribute(NameOID.COMMON_NAME, hostname),
        ])
        
        # Build certificate
        cert_builder = x509.CertificateBuilder()
        cert_builder = cert_builder.subject_name(subject)
        cert_builder = cert_builder.issuer_name(issuer)
        cert_builder = cert_builder.public_key(private_key.public_key())
        cert_builder = cert_builder.serial_number(x509.random_serial_number())
        cert_builder = cert_builder.not_valid_before(datetime.datetime.utcnow())
        cert_builder = cert_builder.not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=days_valid)
        )
        
        # Add extensions
        san_list = [x509.DNSName(hostname)]
        
        # Add IP addresses
        for ip_str in ip_addresses:
            try:
                ip_obj = ipaddress.ip_address(ip_str)
                san_list.append(x509.IPAddress(ip_obj))
            except ValueError:
                logger.warning(f"Invalid IP address: {ip_str}")
        
        # Add localhost variations
        san_list.extend([
            x509.DNSName("localhost"),
            x509.DNSName("127.0.0.1"),
        ])
        
        cert_builder = cert_builder.add_extension(
            x509.SubjectAlternativeName(san_list),
            critical=False,
        )
        
        # Add basic constraints
        cert_builder = cert_builder.add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        
        # Add key usage
        cert_builder = cert_builder.add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        
        # Add extended key usage
        cert_builder = cert_builder.add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
            ]),
            critical=True,
        )
        
        # Sign certificate
        certificate = cert_builder.sign(private_key, hashes.SHA256())
        
        # Write private key
        with open(self.key_file, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        # Write certificate
        with open(self.cert_file, "wb") as f:
            f.write(certificate.public_bytes(serialization.Encoding.PEM))
        
        logger.info(f"SSL certificate generated:")
        logger.info(f"  Certificate: {self.cert_file}")
        logger.info(f"  Private Key: {self.key_file}")
        logger.info(f"  Valid for: {days_valid} days")
        logger.info(f"  Hostname: {hostname}")
        logger.info(f"  IP Addresses: {ip_addresses}")
        
        return self.cert_file, self.key_file
    
    def cert_exists(self) -> bool:
        """Check if certificate files exist"""
        return self.cert_file.exists() and self.key_file.exists()
    
    def cert_is_valid(self, days_ahead: int = 30) -> bool:
        """Check if certificate is valid for at least the specified days"""
        if not self.cert_exists():
            return False
        
        try:
            with open(self.cert_file, "rb") as f:
                cert_data = f.read()
            
            certificate = x509.load_pem_x509_certificate(cert_data)
            expiry = certificate.not_valid_after
            now = datetime.datetime.utcnow()
            
            return (expiry - now).days > days_ahead
        except Exception as e:
            logger.warning(f"Error checking certificate validity: {e}")
            return False
    
    def get_cert_info(self) -> dict:
        """Get certificate information"""
        if not self.cert_exists():
            return {}
        
        try:
            with open(self.cert_file, "rb") as f:
                cert_data = f.read()
            
            certificate = x509.load_pem_x509_certificate(cert_data)
            
            # Extract SAN
            san_extension = None
            for ext in certificate.extensions:
                if isinstance(ext.value, x509.SubjectAlternativeName):
                    san_extension = ext.value
                    break
            
            san_names = []
            if san_extension:
                for name in san_extension:
                    if isinstance(name, x509.DNSName):
                        san_names.append(f"DNS:{name.value}")
                    elif isinstance(name, x509.IPAddress):
                        san_names.append(f"IP:{name.ip}")
            
            return {
                "subject": certificate.subject.rfc4514_string(),
                "issuer": certificate.issuer.rfc4514_string(),
                "serial_number": str(certificate.serial_number),
                "not_valid_before": certificate.not_valid_before.isoformat(),
                "not_valid_after": certificate.not_valid_after.isoformat(),
                "san": san_names,
                "cert_file": str(self.cert_file),
                "key_file": str(self.key_file)
            }
        except Exception as e:
            logger.error(f"Error reading certificate info: {e}")
            return {"error": str(e)}

def get_local_ip():
    """Get the local IP address"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

def create_ssl_context(cert_file: str, key_file: str) -> ssl.SSLContext:
    """Create SSL context for the server"""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_file, key_file)
    
    # Security settings
    context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
    context.options |= ssl.OP_NO_SSLv2
    context.options |= ssl.OP_NO_SSLv3
    context.options |= ssl.OP_NO_TLSv1
    context.options |= ssl.OP_NO_TLSv1_1
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    
    return context

def main():
    """Run the SSL-enabled server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SSL-Enabled YOLO+OCR Pipeline Server")
    parser.add_argument("--yolo-model", default="runs/obb/train4/weights/best.pt",
                       help="Path to YOLO model file")
    parser.add_argument("--ocr-languages", nargs='+', default=['en'],
                       help="OCR languages (e.g., en zh)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8443, help="HTTPS port to bind to")
    parser.add_argument("--http-port", type=int, default=8080, help="HTTP redirect port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-preprocessing", action="store_true", help="Disable image preprocessing")
    parser.add_argument("--no-rotation-correction", action="store_true", help="Disable rotation correction")
    
    # SSL options
    parser.add_argument("--cert-file", help="Path to SSL certificate file")
    parser.add_argument("--key-file", help="Path to SSL private key file")
    parser.add_argument("--cert-dir", default="ssl_certs", help="Directory for auto-generated certificates")
    parser.add_argument("--hostname", help="Hostname for certificate (default: auto-detect)")
    parser.add_argument("--regenerate-cert", action="store_true", help="Force regenerate SSL certificate")
    parser.add_argument("--cert-days", type=int, default=365, help="Certificate validity in days")
    parser.add_argument("--no-ssl", action="store_true", help="Disable SSL (HTTP only)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    global pipeline
    pipeline = EnhancedYOLOOCRPipeline(args.yolo_model, args.ocr_languages)
    
    # Apply command line settings
    pipeline.debug_mode = args.debug
    pipeline.enable_preprocessing = not args.no_preprocessing
    pipeline.enable_rotation_correction = not args.no_rotation_correction
    
    logger.info(f"Initializing Enhanced YOLO+OCR Pipeline Server")
    logger.info(f"YOLO Model: {args.yolo_model}")
    logger.info(f"OCR Languages: {args.ocr_languages}")
    logger.info(f"Device: {pipeline.device}")
    logger.info(f"Tesseract Available: {pipeline.ocr_engine.tesseract_available}")
    logger.info(f"Preprocessing Enabled: {pipeline.enable_preprocessing}")
    logger.info(f"Rotation Correction: {pipeline.enable_rotation_correction}")
    logger.info(f"Debug Mode: {pipeline.debug_mode}")
    
    if args.no_ssl:
        # Run HTTP only
        logger.info(f"Starting HTTP server on {args.host}:{args.http_port}")
        logger.info(f"🔗 Access at: http://{get_local_ip()}:{args.http_port}")
        uvicorn.run(app, host=args.host, port=args.http_port, log_level="info")
        return
    
    # SSL Setup
    cert_manager = SSLCertificateManager(args.cert_dir)
    
    # Determine hostname
    hostname = args.hostname or get_local_ip()
    
    # Check if we need to generate certificates
    if args.cert_file and args.key_file:
        # Use provided certificates
        cert_file = args.cert_file
        key_file = args.key_file
        logger.info(f"Using provided SSL certificates: {cert_file}, {key_file}")
    else:
        # Use auto-generated certificates
        if args.regenerate_cert or not cert_manager.cert_exists() or not cert_manager.cert_is_valid():
            logger.info("Generating new SSL certificate...")
            cert_file, key_file = cert_manager.generate_self_signed_cert(
                hostname=hostname,
                ip_addresses=[get_local_ip(), "127.0.0.1", "::1"],
                days_valid=args.cert_days
            )
        else:
            cert_file = str(cert_manager.cert_file)
            key_file = str(cert_manager.key_file)
            logger.info("Using existing SSL certificate")
    
    # Display certificate info
    cert_info = cert_manager.get_cert_info()
    if cert_info and "error" not in cert_info:
        logger.info("SSL Certificate Information:")
        logger.info(f"  Subject: {cert_info.get('subject', 'Unknown')}")
        logger.info(f"  Valid Until: {cert_info.get('not_valid_after', 'Unknown')}")
        logger.info(f"  SAN: {', '.join(cert_info.get('san', []))}")
    
    # Create SSL context
    try:
        ssl_context = create_ssl_context(cert_file, key_file)
        logger.info("SSL context created successfully")
    except Exception as e:
        logger.error(f"Failed to create SSL context: {e}")
        logger.info("Falling back to HTTP mode...")
        uvicorn.run(app, host=args.host, port=args.http_port, log_level="info")
        return
    
    # Display connection information
    local_ip = get_local_ip()
    logger.info("="*60)
    logger.info("🔒 SSL-Enabled YOLO+OCR Pipeline Server Starting")
    logger.info("="*60)
    logger.info(f"🌐 HTTPS Server: https://{args.host}:{args.port}")
    logger.info(f"🖥️  Local Access: https://localhost:{args.port}")
    logger.info(f"📱 Network Access: https://{local_ip}:{args.port}")
    logger.info(f"🔌 WebSocket URL: wss://localhost:{args.port}/ws")
    logger.info(f"📋 WebSocket Network: wss://{local_ip}:{args.port}/ws")
    logger.info("="*60)
    logger.info("⚠️  Browser Security Notes:")
    logger.info("  - Accept the self-signed certificate warning")
    logger.info("  - Safari: Allow camera permissions in Settings")
    logger.info("  - Chrome: Click 'Advanced' → 'Proceed to localhost'")
    logger.info("  - Firefox: Click 'Advanced' → 'Accept Risk'")
    logger.info("="*60)
    
    # Start HTTPS server
    try:
        uvicorn.run(
            app, 
            host=args.host, 
            port=args.port,
            ssl_keyfile=key_file,
            ssl_certfile=cert_file,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start HTTPS server: {e}")
        logger.info("Trying HTTP fallback...")
        uvicorn.run(app, host=args.host, port=args.http_port, log_level="info")

if __name__ == "__main__":
    main()