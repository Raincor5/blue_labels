import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import asyncio
import websockets
import json
import base64
import time
from concurrent.futures import ThreadPoolExecutor
import platform

class YOLOOCRClient:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLO+OCR Pipeline Client")
        self.root.geometry("1600x1000")
        
        # Initialize variables
        self.websocket = None
        self.cap = None
        self.is_running = False
        self.is_connected = False
        self.current_frame = None
        self.loop = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # WebSocket settings
        self.ws_url = "ws://localhost:8000/ws"
        
        # Webcam settings
        self.camera_index = 0
        self.camera_backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
        
        # OCR settings
        self.show_extracted_regions = tk.BooleanVar(value=False)
        self.ocr_enabled = tk.BooleanVar(value=True)
        
        self.setup_ui()
        self.start_event_loop()
        
    def setup_ui(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main tab
        main_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text="Main Interface")
        
        # OCR Results tab
        ocr_tab = ttk.Frame(notebook)
        notebook.add(ocr_tab, text="OCR Results")
        
        # Settings tab
        settings_tab = ttk.Frame(notebook)
        notebook.add(settings_tab, text="Settings")
        
        self.setup_main_tab(main_tab)
        self.setup_ocr_tab(ocr_tab)
        self.setup_settings_tab(settings_tab)
        
    def setup_main_tab(self, parent):
        """Setup main interface tab"""
        # Connection section
        conn_frame = ttk.LabelFrame(parent, text="WebSocket Connection", padding="5")
        conn_frame.pack(fill=tk.X, pady=(0, 10))
        
        conn_inner = ttk.Frame(conn_frame)
        conn_inner.pack(fill=tk.X)
        
        ttk.Label(conn_inner, text="Server URL:").pack(side=tk.LEFT)
        self.url_var = tk.StringVar(value=self.ws_url)
        ttk.Entry(conn_inner, textvariable=self.url_var, width=40).pack(side=tk.LEFT, padx=(5, 0))
        
        self.connect_btn = ttk.Button(conn_inner, text="Connect", command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        self.connection_status = ttk.Label(conn_inner, text="Disconnected", foreground="red")
        self.connection_status.pack(side=tk.LEFT, padx=(10, 0))
        
        # Camera controls
        camera_frame = ttk.LabelFrame(parent, text="Camera & Processing Controls", padding="5")
        camera_frame.pack(fill=tk.X, pady=(0, 10))
        
        camera_inner = ttk.Frame(camera_frame)
        camera_inner.pack(fill=tk.X)
        
        ttk.Label(camera_inner, text="Camera:").pack(side=tk.LEFT)
        self.camera_var = tk.IntVar(value=0)
        ttk.Spinbox(camera_inner, from_=0, to=10, textvariable=self.camera_var, width=5).pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Button(camera_inner, text="Test Camera", command=self.test_camera).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(camera_inner, text="Upload Image", command=self.upload_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(camera_inner, text="Start Webcam", command=self.start_webcam).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(camera_inner, text="Stop Webcam", command=self.stop_webcam).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(camera_inner, text="Save Frame", command=self.save_frame).pack(side=tk.LEFT, padx=(0, 5))
        
        # Display area
        display_frame = ttk.LabelFrame(parent, text="Detection & OCR Display", padding="5")
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for image display
        canvas_frame = ttk.Frame(display_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg="black", width=800, height=600)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbars for canvas
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(display_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # Performance info
        perf_frame = ttk.Frame(parent)
        perf_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.fps_label = ttk.Label(perf_frame, text="FPS: 0.0")
        self.fps_label.pack(side=tk.LEFT)
        
        self.yolo_latency_label = ttk.Label(perf_frame, text="YOLO: 0ms")
        self.yolo_latency_label.pack(side=tk.LEFT, padx=(20, 0))
        
        self.ocr_latency_label = ttk.Label(perf_frame, text="OCR: 0ms")
        self.ocr_latency_label.pack(side=tk.LEFT, padx=(20, 0))
        
        self.total_latency_label = ttk.Label(perf_frame, text="Total: 0ms")
        self.total_latency_label.pack(side=tk.LEFT, padx=(20, 0))
        
    def setup_ocr_tab(self, parent):
        """Setup OCR results tab"""
        # OCR Results display
        results_frame = ttk.LabelFrame(parent, text="OCR Text Results", padding="5")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create a frame for the text widget and scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.ocr_text = scrolledtext.ScrolledText(text_frame, width=80, height=20, wrap=tk.WORD)
        self.ocr_text.pack(fill=tk.BOTH, expand=True)
        
        # Extracted regions display
        regions_frame = ttk.LabelFrame(parent, text="Extracted Text Regions", padding="5")
        regions_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame for region thumbnails
        self.regions_canvas = tk.Canvas(regions_frame, height=200, bg="white")
        self.regions_canvas.pack(fill=tk.X)
        
        regions_scroll = ttk.Scrollbar(regions_frame, orient=tk.HORIZONTAL, command=self.regions_canvas.xview)
        regions_scroll.pack(fill=tk.X)
        self.regions_canvas.configure(xscrollcommand=regions_scroll.set)
        
        # Control buttons for OCR tab
        ocr_controls = ttk.Frame(parent)
        ocr_controls.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(ocr_controls, text="Clear OCR Results", command=self.clear_ocr_results).pack(side=tk.LEFT)
        ttk.Button(ocr_controls, text="Export OCR Text", command=self.export_ocr_text).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(ocr_controls, text="Copy All Text", command=self.copy_all_text).pack(side=tk.LEFT, padx=(10, 0))
        
    def setup_settings_tab(self, parent):
        """Setup settings tab"""
        # OCR Settings
        ocr_frame = ttk.LabelFrame(parent, text="OCR Settings", padding="10")
        ocr_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(ocr_frame, text="Enable OCR Processing", variable=self.ocr_enabled).pack(anchor=tk.W)
        ttk.Checkbutton(ocr_frame, text="Show Extracted Regions", variable=self.show_extracted_regions).pack(anchor=tk.W, pady=(5, 0))
        
        # Confidence threshold
        conf_frame = ttk.Frame(ocr_frame)
        conf_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(conf_frame, text="YOLO Confidence Threshold:").pack(side=tk.LEFT)
        self.conf_var = tk.DoubleVar(value=0.5)
        conf_scale = ttk.Scale(conf_frame, from_=0.1, to=1.0, variable=self.conf_var, orient=tk.HORIZONTAL, length=200)
        conf_scale.pack(side=tk.LEFT, padx=(10, 0))
        self.conf_label = ttk.Label(conf_frame, text="0.50")
        self.conf_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # OCR confidence threshold
        ocr_conf_frame = ttk.Frame(ocr_frame)
        ocr_conf_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(ocr_conf_frame, text="OCR Confidence Threshold:").pack(side=tk.LEFT)
        self.ocr_conf_var = tk.DoubleVar(value=0.3)
        ocr_conf_scale = ttk.Scale(ocr_conf_frame, from_=0.1, to=1.0, variable=self.ocr_conf_var, orient=tk.HORIZONTAL, length=200)
        ocr_conf_scale.pack(side=tk.LEFT, padx=(10, 0))
        self.ocr_conf_label = ttk.Label(ocr_conf_frame, text="0.30")
        self.ocr_conf_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Update labels
        def update_conf_labels(*args):
            self.conf_label.config(text=f"{self.conf_var.get():.2f}")
            self.ocr_conf_label.config(text=f"{self.ocr_conf_var.get():.2f}")
        
        self.conf_var.trace('w', update_conf_labels)
        self.ocr_conf_var.trace('w', update_conf_labels)
        
        # Log display
        log_frame = ttk.LabelFrame(parent, text="System Logs", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, width=80, height=15, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Clear logs button
        ttk.Button(parent, text="Clear Logs", command=self.clear_logs).pack(pady=(5, 0))
        
    def start_event_loop(self):
        """Start asyncio event loop in a separate thread"""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()
        
        # Wait for loop to be ready
        while self.loop is None:
            time.sleep(0.01)
            
    def toggle_connection(self):
        if self.is_connected:
            self.disconnect()
        else:
            self.connect()
            
    def connect(self):
        """Connect to WebSocket server"""
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._connect(), self.loop)
            
    async def _connect(self):
        try:
            self.ws_url = self.url_var.get()
            self.websocket = await websockets.connect(self.ws_url)
            self.is_connected = True
            
            # Update UI on main thread
            self.root.after(0, self._update_connection_ui, True)
            self.root.after(0, self.log_message, f"Connected to {self.ws_url}")
            
        except Exception as e:
            self.root.after(0, self._update_connection_ui, False)
            self.root.after(0, self.log_message, f"Connection failed: {str(e)}")
            
    def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._disconnect(), self.loop)
            
    async def _disconnect(self):
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self.is_connected = False
        self.root.after(0, self._update_connection_ui, False)
        self.root.after(0, self.log_message, "Disconnected")
        
    def _update_connection_ui(self, connected):
        if connected:
            self.connection_status.config(text="Connected", foreground="green")
            self.connect_btn.config(text="Disconnect")
        else:
            self.connection_status.config(text="Disconnected", foreground="red")
            self.connect_btn.config(text="Connect")
            
    def test_camera(self):
        """Test camera connection"""
        camera_index = self.camera_var.get()
        self.log_message(f"Testing camera {camera_index}...")
        
        try:
            cap = cv2.VideoCapture(camera_index, self.camera_backend)
            
            if not cap.isOpened():
                self.log_message(f"Camera {camera_index} failed to open")
                return False
            
            ret, frame = cap.read()
            if ret and frame is not None:
                self.log_message(f"Camera {camera_index} working")
                self.display_image(frame)
                cap.release()
                return True
            else:
                self.log_message(f"Camera {camera_index} opened but cannot read frames")
                cap.release()
                return False
                
        except Exception as e:
            self.log_message(f"Camera test error: {str(e)}")
            return False
            
    def upload_image(self):
        if not self.is_connected:
            messagebox.showerror("Error", "Please connect to server first")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.process_image_file(file_path)
            
    def process_image_file(self, image_path):
        """Process a single image file"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                messagebox.showerror("Error", "Failed to load image")
                return
                
            # Encode image as JPEG
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send to WebSocket server
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    self._send_image(img_base64, img, f"File: {image_path}"), 
                    self.loop
                )
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            
    async def _send_image(self, img_base64, original_img, source_info):
        """Send image to WebSocket server and handle response"""
        if not self.websocket:
            self.root.after(0, self.log_message, "WebSocket not connected")
            return
            
        try:
            start_time = time.time()
            
            # Send image
            await self.websocket.send(img_base64)
            
            # Receive response with timeout
            try:
                response = await asyncio.wait_for(self.websocket.recv(), timeout=30.0)
            except asyncio.TimeoutError:
                self.root.after(0, self.log_message, "Server response timeout")
                return
                
            end_time = time.time()
            total_latency = int((end_time - start_time) * 1000)
            
            # Parse response
            try:
                data = json.loads(response)
            except json.JSONDecodeError as e:
                self.root.after(0, self.log_message, f"Invalid JSON response: {str(e)}")
                return
            
            # Handle different response types
            response_type = data.get("type", "unknown")
            
            if response_type == "pipeline_result":
                # Extract data
                yolo_detections = data.get("yolo_detections", [])
                ocr_results = data.get("ocr_results", {})
                timing = data.get("timing", {})
                annotated_image_b64 = data.get("annotated_image")
                
                # Display annotated image if available
                if annotated_image_b64:
                    try:
                        img_data = base64.b64decode(annotated_image_b64)
                        nparr = np.frombuffer(img_data, np.uint8)
                        annotated_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        self.root.after(0, self.display_image, annotated_img)
                    except Exception as e:
                        self.log_message(f"Failed to decode annotated image: {e}")
                        self.root.after(0, self.display_image, original_img)
                else:
                    self.root.after(0, self.display_image, original_img)
                
                # Update results
                self.root.after(0, self.update_pipeline_results, yolo_detections, ocr_results, source_info, timing)
                self.root.after(0, self.update_performance, 0, timing.get("yolo_ms", 0), timing.get("ocr_ms", 0), timing.get("total_ms", 0))
                
            elif response_type == "error":
                error_msg = data.get("message", "Unknown error")
                self.root.after(0, self.log_message, f"Server error: {error_msg}")
                
            elif response_type == "info":
                info_msg = data.get("message", "Server info")
                self.root.after(0, self.log_message, f"Server: {info_msg}")
                
            else:
                self.root.after(0, self.log_message, f"Unknown response type: {response_type}")
            
        except Exception as e:
            self.root.after(0, self.log_message, f"WebSocket error: {str(e)}")
            
    def start_webcam(self):
        if not self.is_connected:
            messagebox.showerror("Error", "Please connect to server first")
            return
            
        if self.is_running:
            return
            
        try:
            camera_index = self.camera_var.get()
            self.cap = cv2.VideoCapture(camera_index, self.camera_backend)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Failed to open camera {camera_index}")
                return
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                self.cap.release()
                messagebox.showerror("Error", "Camera opened but cannot read frames")
                return
                
            self.is_running = True
            self.webcam_thread = threading.Thread(target=self.webcam_loop, daemon=True)
            self.webcam_thread.start()
            
            self.log_message(f"Webcam started (Camera {camera_index})")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start webcam: {str(e)}")
            
    def stop_webcam(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.log_message("Webcam stopped")
        
    def webcam_loop(self):
        """Webcam capture loop"""
        fps_counter = 0
        fps_start_time = time.time()
        frame_skip = 0
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.log_message("Failed to read frame from camera")
                    break
                    
                self.current_frame = frame.copy()
                
                # Skip frames to reduce processing load
                frame_skip += 1
                if frame_skip % 2 != 0:  # Process every 2nd frame
                    continue
                
                # Encode frame
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send to WebSocket server
                if self.loop and self.websocket:
                    asyncio.run_coroutine_threadsafe(
                        self._send_image(img_base64, frame, "Webcam"), 
                        self.loop
                    )
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter >= 10:
                    fps = fps_counter / (time.time() - fps_start_time)
                    self.root.after(0, self.update_performance, fps, 0, 0, 0)
                    fps_counter = 0
                    fps_start_time = time.time()
                    
                # Limit frame rate
                time.sleep(0.05)  # ~20 FPS
                
            except Exception as e:
                self.log_message(f"Webcam loop error: {str(e)}")
                break
                
        self.is_running = False
        if self.cap:
            self.cap.release()
            
    def display_image(self, img):
        """Display image on canvas"""
        if img is None:
            return
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Resize if too large
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img_width, img_height = pil_img.size
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            scale = min(scale_x, scale_y, 1.0)
            
            if scale < 1:
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_img)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Keep reference
        self.canvas.image = photo
        
    def update_pipeline_results(self, yolo_detections, ocr_results, source_info, timing):
        """Update results with both YOLO and OCR data"""
        current_time = time.strftime("%H:%M:%S")
        
        # Update OCR text display
        self.ocr_text.insert(tk.END, f"\n=== {current_time} - {source_info} ===\n")
        
        if yolo_detections:
            self.ocr_text.insert(tk.END, f"Found {len(yolo_detections)} YOLO detections\n")
            
            total_ocr_results = 0
            for detection_idx, ocr_data in ocr_results.items():
                detection = yolo_detections[int(detection_idx)]
                self.ocr_text.insert(tk.END, f"\nDetection {int(detection_idx) + 1} ({detection['class_name']}):\n")
                
                if ocr_data:
                    for i, ocr_result in enumerate(ocr_data):
                        text = ocr_result['text']
                        confidence = ocr_result['confidence']
                        self.ocr_text.insert(tk.END, f"  Text {i+1}: \"{text}\" (confidence: {confidence:.3f})\n")
                        total_ocr_results += 1
                else:
                    self.ocr_text.insert(tk.END, "  No text detected\n")
            
            self.ocr_text.insert(tk.END, f"\nTotal OCR results: {total_ocr_results}\n")
            
            # Display timing
            if timing:
                self.ocr_text.insert(tk.END, f"YOLO: {timing.get('yolo_ms', 0):.1f}ms, ")
                self.ocr_text.insert(tk.END, f"OCR: {timing.get('ocr_ms', 0):.1f}ms, ")
                self.ocr_text.insert(tk.END, f"Total: {timing.get('total_ms', 0):.1f}ms\n")
        else:
            self.ocr_text.insert(tk.END, "No detections found\n")
            
        self.ocr_text.see(tk.END)
        
    def update_performance(self, fps, yolo_ms, ocr_ms, total_ms):
        """Update performance labels"""
        if fps > 0:
            self.fps_label.config(text=f"FPS: {fps:.1f}")
        if yolo_ms > 0:
            self.yolo_latency_label.config(text=f"YOLO: {yolo_ms:.0f}ms")
        if ocr_ms > 0:
            self.ocr_latency_label.config(text=f"OCR: {ocr_ms:.0f}ms")
        if total_ms > 0:
            self.total_latency_label.config(text=f"Total: {total_ms:.0f}ms")
            
    def save_frame(self):
        """Save current frame"""
        if self.current_frame is None:
            messagebox.showerror("Error", "No frame to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Frame",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            cv2.imwrite(file_path, self.current_frame)
            self.log_message(f"Frame saved: {file_path}")
            
    def clear_ocr_results(self):
        """Clear OCR results"""
        self.ocr_text.delete(1.0, tk.END)
        
    def clear_logs(self):
        """Clear log text"""
        self.log_text.delete(1.0, tk.END)
        
    def export_ocr_text(self):
        """Export OCR text to file"""
        file_path = filedialog.asksaveasfilename(
            title="Export OCR Text",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.ocr_text.get(1.0, tk.END))
                self.log_message(f"OCR text exported: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export text: {str(e)}")
                
    def copy_all_text(self):
        """Copy all OCR text to clipboard"""
        text = self.ocr_text.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.log_message("OCR text copied to clipboard")
        
    def log_message(self, message):
        """Log a message"""
        current_time = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{current_time}] {message}\n")
        self.log_text.see(tk.END)
        
    def run(self):
        """Start the application"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()
            
    def on_closing(self):
        """Handle application closing"""
        self.stop_webcam()
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._disconnect(), self.loop)
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.executor.shutdown(wait=False)
        self.root.destroy()

if __name__ == "__main__":
    app = YOLOOCRClient()
    app.run()