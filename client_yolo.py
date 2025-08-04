import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import asyncio
import websockets
import json
import base64
import time
from concurrent.futures import ThreadPoolExecutor
import platform

class WebSocketYOLOClient:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("WebSocket YOLO Client Interface")
        self.root.geometry("1400x900")
        
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
        
        self.setup_ui()
        self.start_event_loop()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Connection section
        conn_frame = ttk.LabelFrame(main_frame, text="WebSocket Connection", padding="5")
        conn_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(conn_frame, text="Server URL:").grid(row=0, column=0, sticky=tk.W)
        self.url_var = tk.StringVar(value=self.ws_url)
        ttk.Entry(conn_frame, textvariable=self.url_var, width=50).grid(row=0, column=1, padx=(5, 0))
        
        self.connect_btn = ttk.Button(conn_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=2, padx=(5, 0))
        
        self.connection_status = ttk.Label(conn_frame, text="Disconnected", foreground="red")
        self.connection_status.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Camera settings section
        camera_frame = ttk.LabelFrame(main_frame, text="Camera Settings", padding="5")
        camera_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(camera_frame, text="Camera Index:").grid(row=0, column=0, sticky=tk.W)
        self.camera_var = tk.IntVar(value=0)
        camera_spin = ttk.Spinbox(camera_frame, from_=0, to=10, textvariable=self.camera_var, width=10)
        camera_spin.grid(row=0, column=1, padx=(5, 0))
        
        ttk.Button(camera_frame, text="Test Camera", command=self.test_camera).grid(row=0, column=2, padx=(10, 0))
        ttk.Button(camera_frame, text="List Cameras", command=self.list_cameras).grid(row=0, column=3, padx=(5, 0))
        
        self.camera_status = ttk.Label(camera_frame, text="Camera not tested")
        self.camera_status.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(5, 0))
        
        # Detection settings section
        settings_frame = ttk.LabelFrame(main_frame, text="Detection Settings", padding="5")
        settings_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(settings_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky=tk.W)
        self.conf_var = tk.DoubleVar(value=0.5)
        conf_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, variable=self.conf_var, orient=tk.HORIZONTAL, length=200)
        conf_scale.grid(row=0, column=1, padx=(5, 0))
        self.conf_label = ttk.Label(settings_frame, text="0.50")
        self.conf_label.grid(row=0, column=2, padx=(5, 0))
        
        # Update confidence label
        def update_conf_label(*args):
            self.conf_label.config(text=f"{self.conf_var.get():.2f}")
        self.conf_var.trace('w', update_conf_label)
        
        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=3, pady=(0, 10))
        
        ttk.Button(control_frame, text="Upload Image", command=self.upload_image).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Start Webcam", command=self.start_webcam).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Stop Webcam", command=self.stop_webcam).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Save Frame", command=self.save_frame).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Clear Results", command=self.clear_results).pack(side=tk.LEFT, padx=(0, 10))
        
        # Display and results frame
        display_results_frame = ttk.Frame(main_frame)
        display_results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_results_frame.columnconfigure(0, weight=2)
        display_results_frame.columnconfigure(1, weight=1)
        display_results_frame.rowconfigure(0, weight=1)
        
        # Display frame
        display_frame = ttk.LabelFrame(display_results_frame, text="Detection Display", padding="5")
        display_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Canvas for image display
        self.canvas = tk.Canvas(display_frame, bg="black", width=800, height=600)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars for canvas
        v_scrollbar = ttk.Scrollbar(display_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.canvas.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(display_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # Results frame
        results_frame = ttk.LabelFrame(display_results_frame, text="Detection Results & Logs", padding="5")
        results_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Results text area with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.results_text = tk.Text(text_frame, width=40, height=30, wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        results_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        # Performance info
        perf_frame = ttk.Frame(results_frame)
        perf_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.fps_label = ttk.Label(perf_frame, text="FPS: 0.0")
        self.fps_label.pack(side=tk.LEFT)
        
        self.latency_label = ttk.Label(perf_frame, text="Latency: 0ms")
        self.latency_label.pack(side=tk.LEFT, padx=(20, 0))
        
    def list_cameras(self):
        """List available cameras"""
        self.log_message("Scanning for available cameras...")
        available_cameras = []
        
        # Test multiple camera indices and backends
        backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2, cv2.CAP_ANY]
        
        for i in range(10):  # Test indices 0-9
            for backend in backends_to_try:
                try:
                    cap = cv2.VideoCapture(i, backend)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            backend_name = self.get_backend_name(backend)
                            available_cameras.append(f"Camera {i} ({backend_name}): {width}x{height} @ {fps:.1f}fps")
                            cap.release()
                            break
                    cap.release()
                except:
                    continue
        
        if available_cameras:
            camera_list = "\n".join(available_cameras)
            self.log_message(f"Available cameras:\n{camera_list}")
        else:
            self.log_message("No cameras found!")
            
    def get_backend_name(self, backend):
        """Get backend name from constant"""
        backend_names = {
            cv2.CAP_DSHOW: "DirectShow",
            cv2.CAP_MSMF: "Media Foundation",
            cv2.CAP_V4L2: "Video4Linux2",
            cv2.CAP_ANY: "Any"
        }
        return backend_names.get(backend, f"Backend_{backend}")
        
    def test_camera(self):
        """Test camera connection"""
        camera_index = self.camera_var.get()
        self.log_message(f"Testing camera {camera_index}...")
        
        # Try different backends in order of preference for Windows
        backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        for backend in backends_to_try:
            try:
                self.log_message(f"Trying {self.get_backend_name(backend)} backend...")
                cap = cv2.VideoCapture(camera_index, backend)
                
                if not cap.isOpened():
                    cap.release()
                    continue
                
                # Set camera properties for better performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                ret, frame = cap.read()
                if ret and frame is not None:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    self.camera_status.config(text=f"Camera {camera_index} OK: {width}x{height} @ {fps:.1f}fps ({self.get_backend_name(backend)})", foreground="green")
                    self.camera_backend = backend
                    self.log_message(f"Camera {camera_index} working with {self.get_backend_name(backend)} backend")
                    
                    # Display test frame
                    self.display_image(frame)
                    cap.release()
                    return True
                    
                cap.release()
                
            except Exception as e:
                self.log_message(f"Error with {self.get_backend_name(backend)}: {str(e)}")
                continue
        
        self.camera_status.config(text=f"Camera {camera_index} failed", foreground="red")
        self.log_message(f"Camera {camera_index} is not available")
        return False
        
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
            
    def upload_image(self):
        if not self.is_connected:
            messagebox.showerror("Error", "Please connect to WebSocket server first")
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
            # Load image
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
            return
            
        try:
            start_time = time.time()
            
            # Send image
            await self.websocket.send(img_base64)
            
            # Receive response
            response = await self.websocket.recv()
            end_time = time.time()
            
            latency = int((end_time - start_time) * 1000)
            
            # Parse response
            data = json.loads(response)
            
            # Handle different response types
            if data.get("type") == "detection_result":
                obb_data = data.get("obb", [])
                
                # Draw detections and update UI
                annotated_img = self.draw_obb_detections(original_img.copy(), obb_data)
                
                # Update UI on main thread
                self.root.after(0, self.display_image, annotated_img)
                self.root.after(0, self.update_results, obb_data, source_info, data)
                self.root.after(0, self.update_performance, 0, latency)
            else:
                # Handle other message types (info, error, etc.)
                self.root.after(0, self.log_message, f"Server: {data.get('message', 'Unknown response')}")
            
        except Exception as e:
            self.root.after(0, self.log_message, f"WebSocket error: {str(e)}")
            
    def start_webcam(self):
        if not self.is_connected:
            messagebox.showerror("Error", "Please connect to WebSocket server first")
            return
            
        if self.is_running:
            return
            
        try:
            camera_index = self.camera_var.get()
            
            # Use the tested backend
            self.cap = cv2.VideoCapture(camera_index, self.camera_backend)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Failed to open camera {camera_index}")
                return
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize delay
            
            # Test if we can read a frame
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                self.cap.release()
                messagebox.showerror("Error", "Camera opened but cannot read frames")
                return
                
            self.is_running = True
            self.webcam_thread = threading.Thread(target=self.webcam_loop, daemon=True)
            self.webcam_thread.start()
            
            self.log_message(f"Webcam started (Camera {camera_index} with {self.get_backend_name(self.camera_backend)})")
            
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
                if frame_skip % 3 != 0:  # Process every 3rd frame
                    continue
                
                # Encode frame as JPEG
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
                    self.root.after(0, self.update_performance, fps, 0)
                    fps_counter = 0
                    fps_start_time = time.time()
                    
                # Limit frame rate
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                self.log_message(f"Webcam loop error: {str(e)}")
                break
                
        self.is_running = False
        if self.cap:
            self.cap.release()
            
    def draw_obb_detections(self, img, obb_data):
        """Draw oriented bounding boxes on image"""
        conf_threshold = self.conf_var.get()
        
        for i, detection in enumerate(obb_data):
            if isinstance(detection, dict):
                # New format from improved server
                conf = detection.get('confidence', 0.8)
                if conf > conf_threshold:
                    center_x = detection['center_x']
                    center_y = detection['center_y']
                    width = detection['width']
                    height = detection['height']
                    rotation = detection['rotation']
                    class_name = detection.get('class_name', f'Detection_{i}')
                    
                    # Calculate rotated rectangle points
                    cos_r = np.cos(rotation)
                    sin_r = np.sin(rotation)
                    
                    # Half dimensions
                    hw, hh = width / 2, height / 2
                    
                    # Corner points relative to center
                    corners = np.array([
                        [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
                    ])
                    
                    # Rotate and translate
                    rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
                    rotated_corners = corners @ rotation_matrix.T
                    final_corners = rotated_corners + [center_x, center_y]
                    
                    # Convert to integer points
                    points = final_corners.astype(int)
                    
                    # Draw rotated rectangle
                    cv2.polylines(img, [points], True, (0, 255, 0), 2)
                    
                    # Draw label
                    label = f'{class_name}: {conf:.2f}'
                    label_pos = (int(center_x - width/4), int(center_y - height/2 - 10))
                    
                    # Draw label background
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, 
                                (label_pos[0], label_pos[1] - label_height - 5),
                                (label_pos[0] + label_width, label_pos[1] + 5),
                                (0, 255, 0), -1)
                    
                    # Draw label text
                    cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            elif isinstance(detection, list) and len(detection) >= 5:
                # Old format compatibility
                center_x, center_y, width, height, rotation = detection[:5]
                conf = 0.8  # Default confidence
                
                if conf > conf_threshold:
                    # Same drawing code as above...
                    cos_r = np.cos(rotation)
                    sin_r = np.sin(rotation)
                    hw, hh = width / 2, height / 2
                    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
                    rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
                    rotated_corners = corners @ rotation_matrix.T
                    final_corners = rotated_corners + [center_x, center_y]
                    points = final_corners.astype(int)
                    cv2.polylines(img, [points], True, (0, 255, 0), 2)
                    
                    label = f'Label_{i}: {conf:.2f}'
                    label_pos = (int(center_x - width/4), int(center_y - height/2 - 10))
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (label_pos[0], label_pos[1] - label_height - 5),
                                (label_pos[0] + label_width, label_pos[1] + 5), (0, 255, 0), -1)
                    cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
        return img
        
    def display_image(self, img):
        """Display image on canvas"""
        if img is None:
            return
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img_rgb)
        
        # Resize if too large
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img_width, img_height = pil_img.size
            
            # Calculate scaling factor
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
        
    def update_results(self, obb_data, source_info, full_response=None):
        """Update results text area"""
        current_time = time.strftime("%H:%M:%S")
        
        self.results_text.insert(tk.END, f"\n--- {current_time} ---\n")
        self.results_text.insert(tk.END, f"Source: {source_info}\n")
        
        if obb_data:
            self.results_text.insert(tk.END, f"Detections found: {len(obb_data)}\n")
            
            # Show timing info if available
            if full_response and 'timing' in full_response:
                timing = full_response['timing']
                self.results_text.insert(tk.END, f"Processing time: {timing.get('total_ms', 0):.1f}ms\n")
                self.results_text.insert(tk.END, f"Inference time: {timing.get('inference_ms', 0):.1f}ms\n")
            
            self.results_text.insert(tk.END, "\n")
            
            for i, detection in enumerate(obb_data):
                if isinstance(detection, dict):
                    # New format from improved server
                    self.results_text.insert(tk.END, f"Detection {i+1}:\n")
                    self.results_text.insert(tk.END, f"  Class: {detection.get('class_name', 'Unknown')}\n")
                    self.results_text.insert(tk.END, f"  Confidence: {detection.get('confidence', 0):.3f}\n")
                    self.results_text.insert(tk.END, f"  Center: ({detection.get('center_x', 0):.1f}, {detection.get('center_y', 0):.1f})\n")
                    self.results_text.insert(tk.END, f"  Size: {detection.get('width', 0):.1f} x {detection.get('height', 0):.1f}\n")
                    self.results_text.insert(tk.END, f"  Rotation: {np.degrees(detection.get('rotation', 0)):.1f}°\n\n")
                elif isinstance(detection, list) and len(detection) >= 5:
                    # Old format compatibility
                    center_x, center_y, width, height, rotation = detection[:5]
                    self.results_text.insert(tk.END, f"Detection {i+1}:\n")
                    self.results_text.insert(tk.END, f"  Center: ({center_x:.1f}, {center_y:.1f})\n")
                    self.results_text.insert(tk.END, f"  Size: {width:.1f} x {height:.1f}\n")
                    self.results_text.insert(tk.END, f"  Rotation: {np.degrees(rotation):.1f}°\n\n")
        else:
            self.results_text.insert(tk.END, "No detections found\n")
            
        self.results_text.see(tk.END)
        
    def update_performance(self, fps, latency):
        """Update performance labels"""
        if fps > 0:
            self.fps_label.config(text=f"FPS: {fps:.1f}")
        if latency > 0:
            self.latency_label.config(text=f"Latency: {latency}ms")
            
    def log_message(self, message):
        """Log a message to results area"""
        current_time = time.strftime("%H:%M:%S")
        self.results_text.insert(tk.END, f"[{current_time}] {message}\n")
        self.results_text.see(tk.END)
        
    def clear_results(self):
        """Clear results text area"""
        self.results_text.delete(1.0, tk.END)
        
    def save_frame(self):
        """Save current frame with detections"""
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
    app = WebSocketYOLOClient()
    app.run()