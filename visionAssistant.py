import cv2
import numpy as np
from collections import deque
import time

class VisionAssistant:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mode = 'detection'  # detection, tracking, measurement, drawing, pose
        
        # Face/eye detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Object tracking
        self.tracking = False
        self.tracker = None
        
        # Motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Drawing mode
        self.drawing_points = deque(maxlen=512)
        self.drawing = False
        
        # Measurement points
        self.measure_points = []
        
        # Performance metrics
        self.fps = 0
        self.frame_times = deque(maxlen=30)
        
    def detect_faces_eyes(self, frame):
        """Detect faces and eyes with accuracy indicators"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Draw face rectangle with confidence
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Detect eyes within face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                
        return frame, len(faces)
    
    def detect_motion(self, frame):
        """Detect and highlight motion in frame"""
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows and noise
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, 'Motion', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame, motion_detected
    
    def track_object(self, frame):
        """Track selected object"""
        if self.tracking and self.tracker is not None:
            success, box = self.tracker.update(frame)
            
            if success:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                cv2.putText(frame, 'Tracking', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            else:
                cv2.putText(frame, 'Tracking Lost', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def detect_edges_corners(self, frame):
        """Detect edges and corners for scene understanding"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Shi-Tomasi corner detection
        corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
        
        if corners is not None:
            corners = np.int8(corners)
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        
        # Overlay edges
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        frame = cv2.addWeighted(frame, 0.8, edges_colored, 0.2, 0)
        
        return frame
    
    def measure_distance(self, frame):
        """Measure pixel distance between two points"""
        if len(self.measure_points) == 2:
            p1, p2 = self.measure_points
            cv2.line(frame, p1, p2, (255, 255, 0), 2)
            cv2.circle(frame, p1, 5, (0, 255, 0), -1)
            cv2.circle(frame, p2, 5, (0, 255, 0), -1)
            
            distance = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            mid_point = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
            cv2.putText(frame, f'{distance:.1f}px', mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
    
    def air_drawing(self, frame):
        """Draw in air using color tracking (green object)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Track green color (adjust ranges as needed)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    self.drawing_points.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        
        # Draw the trail
        for i in range(1, len(self.drawing_points)):
            if self.drawing_points[i-1] is None or self.drawing_points[i] is None:
                continue
            cv2.line(frame, self.drawing_points[i-1], self.drawing_points[i], (255, 0, 255), 2)
        
        return frame
    
    def add_info_overlay(self, frame):
        """Add informative overlay to frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Mode info
        mode_text = f'Mode: {self.mode.upper()}'
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS
        fps_text = f'FPS: {self.fps:.1f}'
        cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Instructions
        instructions = {
            'detection': 'Face & Motion Detection',
            'tracking': 'Click to track object | R: Reset',
            'measurement': 'Click 2 points to measure | C: Clear',
            'drawing': 'Use green object to draw | C: Clear',
            'edges': 'Edge & Corner Detection'
        }
        cv2.putText(frame, instructions.get(self.mode, ''), (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if self.mode == 'tracking':
            if event == cv2.EVENT_LBUTTONDOWN:
                # Initialize tracker
                self.tracker = cv2.TrackerCSRT_create()
                bbox = cv2.selectROI('Vision Assistant', param, False)
                if bbox != (0, 0, 0, 0):
                    self.tracker.init(param, bbox)
                    self.tracking = True
        
        elif self.mode == 'measurement':
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.measure_points) < 2:
                    self.measure_points.append((x, y))
                else:
                    self.measure_points = [(x, y)]
    
    def run(self):
        """Main loop"""
        cv2.namedWindow('Vision Assistant')
        
        print("=== OpenCV Vision Assistant ===")
        print("Keys:")
        print("  1: Face & Motion Detection")
        print("  2: Object Tracking")
        print("  3: Measurement Mode")
        print("  4: Air Drawing (use green object)")
        print("  5: Edge & Corner Detection")
        print("  C: Clear/Reset")
        print("  S: Save screenshot")
        print("  Q: Quit")
        
        while True:
            start_time = time.time()
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            original = frame.copy()
            
            # Process based on mode
            if self.mode == 'detection':
                frame, face_count = self.detect_faces_eyes(frame)
                frame, motion = self.detect_motion(frame)
            
            elif self.mode == 'tracking':
                frame = self.track_object(frame)
                cv2.setMouseCallback('Vision Assistant', self.mouse_callback, original)
            
            elif self.mode == 'measurement':
                frame = self.measure_distance(frame)
                cv2.setMouseCallback('Vision Assistant', self.mouse_callback, frame)
            
            elif self.mode == 'drawing':
                frame = self.air_drawing(frame)
            
            elif self.mode == 'edges':
                frame = self.detect_edges_corners(frame)
            
            # Add overlay
            frame = self.add_info_overlay(frame)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            self.frame_times.append(elapsed)
            self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
            
            cv2.imshow('Vision Assistant', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.mode = 'detection'
            elif key == ord('2'):
                self.mode = 'tracking'
                self.tracking = False
            elif key == ord('3'):
                self.mode = 'measurement'
            elif key == ord('4'):
                self.mode = 'drawing'
            elif key == ord('5'):
                self.mode = 'edges'
            elif key == ord('c'):
                self.drawing_points.clear()
                self.measure_points = []
                self.tracking = False
            elif key == ord('s'):
                filename = f'screenshot_{int(time.time())}.jpg'
                cv2.imwrite(filename, frame)
                print(f'Saved: {filename}')
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    assistant = VisionAssistant()
    assistant.run()