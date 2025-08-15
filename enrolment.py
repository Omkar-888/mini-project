"""
Direct TensorFlow Lite MediaPipe Implementation - Using .tflite models with TFLite interpreter
This implementation directly loads and uses MediaPipe .tflite models with TensorFlow Lite interpreter and uses only TFLite + OpenCV fallback
"""

import os
import sys
import gc
import atexit
import cv2
import numpy as np
import pickle
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import warnings

# Add utils to path
sys.path.append(os.path.dirname(__file__))

# Try to import TensorFlow Lite
HAS_TFLITE = False
try:
    import tensorflow as tf
    HAS_TFLITE = True
    print("✅ TensorFlow Lite available for direct model loading")
except ImportError:
    print("❌ TensorFlow Lite not found - will use OpenCV only")


class DirectTFLiteFaceProcessor:
    """Direct TensorFlow Lite face processor using MediaPipe .tflite models"""
    
    def __init__(self):
        self.tflite_available = False
        self.opencv_available = False
        self.current_mode = "unknown"
        
        # Model paths
        self.models_dir = "mediapipe_models"
        self.face_detection_model = None
        self.face_landmarks_model = None
        
        # Try to initialize TFLite models first
        self._init_tflite_models()
        
        # Always initialize OpenCV as fallback
        self._init_opencv()
        
        # Set processing mode
        if self.tflite_available:
            self.current_mode = "tflite"
            print("🎯 Using Direct TensorFlow Lite for face processing")
        elif self.opencv_available:
            self.current_mode = "opencv"
            print("🔄 Falling back to OpenCV for face processing")
        else:
            raise RuntimeError("Neither TensorFlow Lite nor OpenCV could be initialized")
    
    def _init_tflite_models(self):
        """Initialize TensorFlow Lite models directly"""
        if not HAS_TFLITE:
            return
            
        try:
            # Load face detection model
            detection_model_path = os.path.join(self.models_dir, "mediapipe_face_detection_short_range.tflite")
            if os.path.exists(detection_model_path):
                self.face_detection_model = self._load_tflite_model(detection_model_path, "Face Detection")
            
            # Load face landmarks model  
            landmarks_model_path = os.path.join(self.models_dir, "mediapipe_face_landmark.tflite")
            if os.path.exists(landmarks_model_path):
                self.face_landmarks_model = self._load_tflite_model(landmarks_model_path, "Face Landmarks")
            
            # Check if we have at least face detection
            if self.face_detection_model is not None:
                self.tflite_available = True
                print("✅ TensorFlow Lite models loaded successfully")
            else:
                print("❌ No TensorFlow Lite models could be loaded")
                
        except Exception as e:
            print(f"❌ TensorFlow Lite initialization failed: {e}")
            self.tflite_available = False
    
    def _load_tflite_model(self, model_path: str, model_name: str) -> Optional[tf.lite.Interpreter]:
        """Load a TFLite model with proper delegates"""
        try:
            # Create interpreter with XNNPACK delegate for better performance
            interpreter = tf.lite.Interpreter(
                model_path=model_path,
                experimental_delegates=[
                    tf.lite.experimental.load_delegate('libxnnpack_delegate.so')
                ] if os.name != 'nt' else None  # XNNPACK delegate for non-Windows
            )
            
            # If XNNPACK fails, try without delegates
            if interpreter is None:
                interpreter = tf.lite.Interpreter(model_path=model_path)
            
            # Allocate tensors
            interpreter.allocate_tensors()
            
            # Get input/output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"✅ {model_name} loaded:")
            print(f"   Input shape: {input_details[0]['shape']}")
            print(f"   Output shape: {output_details[0]['shape']}")
            
            return interpreter
            
        except Exception as e:
            print(f"⚠️ Failed to load {model_name} model: {e}")
            # Try loading without delegates as fallback
            try:
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                print(f"✅ {model_name} loaded without delegates")
                return interpreter
            except Exception as e2:
                print(f"❌ {model_name} completely failed to load: {e2}")
                return None
    
    def _init_opencv(self):
        """Initialize OpenCV face detector"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise RuntimeError("Could not load OpenCV face cascade")
            
            self.opencv_available = True
            print("✅ OpenCV face detector initialized")
            
        except Exception as e:
            print(f"❌ OpenCV initialization failed: {e}")
            self.opencv_available = False
    
    def detect_faces_in_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using TFLite first, OpenCV fallback"""
        
        # Try TensorFlow Lite first
        if self.tflite_available and self.face_detection_model is not None:
            try:
                faces = self._detect_faces_tflite(image)
                if faces:  # If TFLite found faces, return them
                    return faces
                else:
                    print("🔄 TFLite found no faces, trying OpenCV fallback...")
            except Exception as e:
                print(f"⚠️ TFLite detection failed, falling back to OpenCV: {e}")
                if self.current_mode == "tflite":
                    print("🔄 Switching to OpenCV fallback mode")
                    self.current_mode = "opencv_fallback"
        
        # Fallback to OpenCV
        if self.opencv_available:
            return self._detect_faces_opencv(image)
        
        raise RuntimeError("No face detection method available")
    
    def _detect_faces_tflite(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using TensorFlow Lite MediaPipe model"""
        
        # Get input details
        input_details = self.face_detection_model.get_input_details()
        output_details = self.face_detection_model.get_output_details()
        
        # Expected input shape for MediaPipe face detection: [1, 320, 320, 3]
        input_shape = input_details[0]['shape']
        input_height, input_width = input_shape[1], input_shape[2]
        
        # Preprocess image
        original_height, original_width = image.shape[:2]
        
        # Resize and normalize image for model input
        resized_image = cv2.resize(image, (input_width, input_height))
        if len(resized_image.shape) == 3:
            # Convert BGR to RGB
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and add batch dimension
        input_data = np.expand_dims(resized_image.astype(np.float32) / 255.0, axis=0)
        
        # Set input tensor
        self.face_detection_model.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        self.face_detection_model.invoke()
        
        # Get outputs
        # MediaPipe face detection output format analysis
        faces = []
        try:
            # Debug: Print output shapes to understand the model
            for i, detail in enumerate(output_details):
                output_tensor = self.face_detection_model.get_tensor(detail['index'])
                print(f"   Output {i}: shape {output_tensor.shape}, dtype {output_tensor.dtype}")
            
            # Get the main output tensor (usually the first one)
            main_output = self.face_detection_model.get_tensor(output_details[0]['index'])
            
            # MediaPipe face detection output is typically [1, 896, 16] where:
            # - 896 is the number of anchor boxes
            # - 16 contains [bbox coordinates, confidence, keypoints]
            
            if len(main_output.shape) == 3 and main_output.shape[2] >= 16:
                # Format: [batch, num_anchors, 16]
                # Elements 0-3: bbox coordinates (ymin, xmin, ymax, xmax)
                # Element 4: confidence score
                # Elements 5-15: keypoints or other data
                
                batch_size, num_anchors, features = main_output.shape
                
                for i in range(num_anchors):
                    detection = main_output[0, i]  # Get detection for batch 0
                    
                    # Extract confidence (usually at index 4 or similar position)
                    # Try different positions for confidence score
                    confidence_candidates = [detection[4], detection[5], detection[15]]
                    confidence = max(confidence_candidates)  # Take the highest as confidence
                    
                    # Skip low confidence detections
                    if confidence < 0.5:  # Lower threshold for testing
                        continue
                    
                    # Extract bounding box coordinates
                    # MediaPipe format is usually [ymin, xmin, ymax, xmax] normalized
                    ymin, xmin, ymax, xmax = detection[0], detection[1], detection[2], detection[3]
                    
                    # Convert to pixel coordinates
                    x = int(xmin * original_width)
                    y = int(ymin * original_height)
                    width = int((xmax - xmin) * original_width)
                    height = int((ymax - ymin) * original_height)
                    
                    # Ensure coordinates are valid
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, original_width - x)
                    height = min(height, original_height - y)
                    
                    if width < 50 or height < 50:  # Skip very small faces
                        continue
                    
                    # Extract face region
                    face_region = image[y:y+height, x:x+width]
                    
                    # Try to get landmarks if landmarks model is available
                    landmarks = None
                    if self.face_landmarks_model is not None:
                        try:
                            landmarks = self._extract_landmarks_tflite(face_region)
                        except:
                            pass  # Continue without landmarks
                    
                    face_info = {
                        'bbox': [x, y, width, height],
                        'confidence': float(confidence),
                        'face_region': face_region,
                        'landmarks': landmarks,
                        'method': 'tflite'
                    }
                    faces.append(face_info)
            
            else:
                # Fallback: try to parse single output differently
                print(f"   Trying alternative parsing for shape {main_output.shape}")
                # Could add more parsing logic here for different model formats
                
        except Exception as e:
            print(f"Error parsing TFLite output: {e}")
            print(f"Output shapes: {[self.face_detection_model.get_tensor(d['index']).shape for d in output_details]}")
            # Don't return empty - let it continue to try OpenCV
            print("🔄 TFLite parsing failed, will use OpenCV fallback")
            
        # If no faces found with TFLite, return empty to trigger OpenCV fallback
        return faces
    
    def _extract_landmarks_tflite(self, face_region: np.ndarray) -> Optional[List[List[float]]]:
        """Extract landmarks using TensorFlow Lite face landmarks model"""
        if self.face_landmarks_model is None:
            return None
        
        try:
            input_details = self.face_landmarks_model.get_input_details()
            output_details = self.face_landmarks_model.get_output_details()
            
            # Expected input shape for MediaPipe face landmarks: [1, 192, 192, 3]
            input_shape = input_details[0]['shape']
            input_height, input_width = input_shape[1], input_shape[2]
            
            # Preprocess face region
            resized_face = cv2.resize(face_region, (input_width, input_height))
            if len(resized_face.shape) == 3:
                resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
            
            # Normalize and add batch dimension
            input_data = np.expand_dims(resized_face.astype(np.float32) / 255.0, axis=0)
            
            # Set input and run inference
            self.face_landmarks_model.set_tensor(input_details[0]['index'], input_data)
            self.face_landmarks_model.invoke()
            
            # Get landmarks output
            landmarks_output = self.face_landmarks_model.get_tensor(output_details[0]['index'])
            
            # Parse landmarks - MediaPipe format is typically flattened [1, 1, 1, 1404]
            # where 1404 = 468 landmarks * 3 coordinates (x, y, z)
            if landmarks_output.size >= 1404:  # 468 landmarks * 3
                # Flatten and reshape
                flat_landmarks = landmarks_output.flatten()
                landmarks = []
                
                # Each landmark has 3 coordinates (x, y, z), we only need x, y
                for i in range(0, len(flat_landmarks), 3):
                    if i + 1 < len(flat_landmarks):  # Ensure we have at least x, y
                        # Convert normalized coordinates to face region coordinates
                        x = flat_landmarks[i] * face_region.shape[1]
                        y = flat_landmarks[i + 1] * face_region.shape[0]
                        landmarks.append([x, y])
                
                return landmarks
            
        except Exception as e:
            print(f"Landmark extraction failed: {e}")
            return None
        
        return None
    
    def _detect_faces_opencv(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using OpenCV (fallback)"""
        # Convert to grayscale for detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        face_rects = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces = []
        for (x, y, w, h) in face_rects:
            # Extract face region
            face_region = image[y:y+h, x:x+w]
            
            face_info = {
                'bbox': [x, y, w, h],
                'confidence': 0.8,  # OpenCV doesn't provide confidence
                'face_region': face_region,
                'landmarks': None,  # OpenCV doesn't provide landmarks
                'method': 'opencv'
            }
            faces.append(face_info)
        
        return faces
    
    def extract_features(self, face_info: Dict[str, Any]) -> np.ndarray:
        """Extract features using hybrid approach"""
        
        face_region = face_info['face_region']
        landmarks = face_info.get('landmarks')
        method = face_info.get('method', 'unknown')
        
        # If we have TFLite landmarks, use them for enhanced features
        if method == 'tflite' and landmarks is not None:
            try:
                return self._extract_tflite_enhanced_features(face_region, landmarks)
            except Exception as e:
                print(f"⚠️ TFLite feature extraction failed, using traditional CV: {e}")
        
        # Fallback to traditional CV features
        return self._extract_traditional_features(face_region)
    
    def _extract_tflite_enhanced_features(self, face_region: np.ndarray, landmarks: List[List[float]]) -> np.ndarray:
        """Extract enhanced features using TFLite landmarks + traditional CV"""
        
        # Convert landmarks to numpy array
        landmarks_array = np.array(landmarks)
        
        # Extract landmark-based features
        landmark_features = []
        
        if len(landmarks_array) >= 68:  # At least key facial landmarks
            # Key facial point indices (adjusted for different landmark models)
            try:
                # Geometric features from landmarks
                if len(landmarks_array) >= 468:  # Full MediaPipe face mesh
                    # Use MediaPipe face mesh landmark indices
                    nose_tip = landmarks_array[1] if len(landmarks_array) > 1 else landmarks_array[0]
                    left_eye = landmarks_array[33] if len(landmarks_array) > 33 else landmarks_array[min(33, len(landmarks_array)-1)]
                    right_eye = landmarks_array[362] if len(landmarks_array) > 362 else landmarks_array[min(362, len(landmarks_array)-1)]
                    mouth_left = landmarks_array[61] if len(landmarks_array) > 61 else landmarks_array[min(61, len(landmarks_array)-1)]
                    mouth_right = landmarks_array[291] if len(landmarks_array) > 291 else landmarks_array[min(291, len(landmarks_array)-1)]
                else:
                    # Use first available landmarks
                    nose_tip = landmarks_array[0]
                    left_eye = landmarks_array[min(1, len(landmarks_array)-1)]
                    right_eye = landmarks_array[min(2, len(landmarks_array)-1)]
                    mouth_left = landmarks_array[min(3, len(landmarks_array)-1)]
                    mouth_right = landmarks_array[min(4, len(landmarks_array)-1)]
                
                # Calculate geometric features
                eye_distance = np.linalg.norm(right_eye - left_eye)
                nose_to_mouth = np.linalg.norm(nose_tip - (mouth_left + mouth_right) / 2)
                face_width = np.max(landmarks_array[:, 0]) - np.min(landmarks_array[:, 0])
                face_height = np.max(landmarks_array[:, 1]) - np.min(landmarks_array[:, 1])
                
                # Ratios (pose-invariant features)
                eye_mouth_ratio = eye_distance / nose_to_mouth if nose_to_mouth > 0 else 0
                aspect_ratio = face_width / face_height if face_height > 0 else 0
                
                landmark_features = [eye_distance, nose_to_mouth, face_width, face_height, 
                                   eye_mouth_ratio, aspect_ratio]
                
                # Add landmark density features
                x_coords = landmarks_array[:, 0]
                y_coords = landmarks_array[:, 1]
                landmark_features.extend([
                    np.mean(x_coords), np.std(x_coords),
                    np.mean(y_coords), np.std(y_coords),
                    np.min(x_coords), np.max(x_coords),
                    np.min(y_coords), np.max(y_coords)
                ])
                
            except Exception as e:
                print(f"Error computing landmark features: {e}")
                landmark_features = []
        
        # Pad or truncate landmark features to 64 dimensions
        landmark_features = landmark_features[:64]
        while len(landmark_features) < 64:
            landmark_features.append(0.0)
        
        # Combine with traditional CV features
        traditional_features = self._extract_traditional_features(face_region)
        traditional_features = traditional_features[:448]  # Take first 448 features
        
        # Combine for 512-dimensional vector
        combined_features = np.concatenate([
            np.array(landmark_features),  # 64 landmark features
            traditional_features         # 448 traditional features
        ])
        
        return combined_features[:512]  # Ensure exactly 512 dimensions
    
    def _extract_traditional_features(self, face_region: np.ndarray) -> np.ndarray:
        """Extract traditional computer vision features"""
        try:
            # Resize to standard size
            face_resized = cv2.resize(face_region, (128, 128))
            
            # Convert to grayscale if needed
            if len(face_resized.shape) == 3:
                gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_resized
            
            features = []
            
            # 1. Local Binary Pattern (LBP) features
            lbp_features = self._compute_lbp_features(gray_face)
            features.extend(lbp_features[:256])  # 256 features
            
            # 2. Histogram of Oriented Gradients (HOG) features
            hog_features = self._compute_hog_features(gray_face)
            features.extend(hog_features[:128])  # 128 features
            
            # 3. Statistical features
            stat_features = self._compute_statistical_features(gray_face)
            features.extend(stat_features[:128])  # 128 features
            
            # Ensure exactly 512 features
            while len(features) < 512:
                features.append(0.0)
            
            features_array = np.array(features[:512], dtype=np.float32)
            
            # Normalize
            norm = np.linalg.norm(features_array)
            if norm > 0:
                features_array = features_array / norm
            
            return features_array
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            # Return zero vector as fallback
            return np.zeros(512, dtype=np.float32)
    
    def _compute_lbp_features(self, gray_image):
        """Compute Local Binary Pattern features"""
        features = []
        
        # Simple LBP computation
        rows, cols = gray_image.shape
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = gray_image[i, j]
                binary_string = ''
                
                # Check 8 neighbors
                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                    gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                    gray_image[i+1, j-1], gray_image[i, j-1]
                ]
                
                for neighbor in neighbors:
                    binary_string += '1' if neighbor >= center else '0'
                
                lbp_value = int(binary_string, 2)
                if len(features) < 256:
                    features.append(lbp_value / 255.0)  # Normalize
        
        # Pad if needed
        while len(features) < 256:
            features.append(0.0)
        
        return features[:256]
    
    def _compute_hog_features(self, gray_image):
        """Compute Histogram of Oriented Gradients features"""
        # Compute gradients
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute magnitude and angle
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)
        
        # Convert angle to degrees
        angle = np.degrees(angle) % 180
        
        # Create histogram of gradients
        hist_features = []
        
        # Divide image into cells
        cell_size = 16
        for i in range(0, gray_image.shape[0], cell_size):
            for j in range(0, gray_image.shape[1], cell_size):
                cell_mag = magnitude[i:i+cell_size, j:j+cell_size]
                cell_angle = angle[i:i+cell_size, j:j+cell_size]
                
                # Create histogram for this cell
                hist, _ = np.histogram(cell_angle.ravel(), bins=9, range=(0, 180), 
                                     weights=cell_mag.ravel())
                hist_features.extend(hist / (np.sum(hist) + 1e-6))  # Normalize
                
                if len(hist_features) >= 128:
                    break
            if len(hist_features) >= 128:
                break
        
        # Pad if needed
        while len(hist_features) < 128:
            hist_features.append(0.0)
        
        return hist_features[:128]
    
    def _compute_statistical_features(self, gray_image):
        """Compute statistical features"""
        features = []
        
        # Global statistics
        features.extend([
            np.mean(gray_image), np.std(gray_image),
            np.min(gray_image), np.max(gray_image),
            np.median(gray_image), np.percentile(gray_image, 25),
            np.percentile(gray_image, 75)
        ])
        
        # Regional statistics (divide image into 4x4 grid)
        h, w = gray_image.shape
        cell_h, cell_w = h // 4, w // 4
        
        for i in range(4):
            for j in range(4):
                region = gray_image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                features.extend([
                    np.mean(region), np.std(region),
                    np.min(region), np.max(region)
                ])
                
                if len(features) >= 128:
                    return features[:128]
        
        # Pad if needed
        while len(features) < 128:
            features.append(0.0)
        
        return features[:128]
    
    def get_processing_info(self) -> Dict[str, Any]:
        """Get information about current processing mode"""
        return {
            'current_mode': self.current_mode,
            'tflite_available': self.tflite_available,
            'opencv_available': self.opencv_available,
            'has_tflite_lib': HAS_TFLITE,
            'face_detection_model_loaded': self.face_detection_model is not None,
            'face_landmarks_model_loaded': self.face_landmarks_model is not None
        }


class DirectTFLiteDatabase:
    """Database manager for direct TFLite system"""
    
    def __init__(self, db_path: str = "database/enrolment.pkl"):
        self.db_path = db_path
        self.data = {
            'metadata': {
                'created_date': datetime.now().isoformat(),
                'last_modified': datetime.now().isoformat(),
                'total_persons': 0,
                'embedding_dimension': 512,
                'processing_mode': 'direct_tflite'
            },
            'persons': {}
        }
        self.load_database()
    
    def load_database(self):
        """Load existing database"""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'rb') as f:
                    self.data = pickle.load(f)
                print(f"✅ Database loaded: {len(self.data['persons'])} persons")
            else:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
                print("📝 New database will be created")
        except Exception as e:
            print(f"❌ Database load error: {e}")
            print("📝 Starting with empty database")
    
    def add_person(self, person_id: str, embeddings: List[np.ndarray], 
                   folder_path: str, processing_info: Dict[str, Any]):
        """Add person to database"""
        # Calculate aggregated embedding
        aggregated_embedding = np.mean(embeddings, axis=0)
        
        self.data['persons'][person_id] = {
            'folder_name': os.path.basename(folder_path),
            'embeddings': [emb.tolist() for emb in embeddings],
            'aggregated_embedding': aggregated_embedding.tolist(),
            'image_count': len(embeddings),
            'enrolment_date': datetime.now().isoformat(),
            'processing_info': processing_info
        }
        
        # Update metadata
        self.data['metadata']['total_persons'] = len(self.data['persons'])
        self.data['metadata']['last_modified'] = datetime.now().isoformat()
        
        print(f"✅ Added {person_id} with {len(embeddings)} embeddings")
    
    def save_database(self):
        """Save database to file"""
        try:
            # Create backup
            if os.path.exists(self.db_path):
                backup_path = self.db_path.replace('.pkl', '_backup.pkl')
                os.rename(self.db_path, backup_path)
            
            # Save new database
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.data, f, protocol=4)
            
            print(f"✅ Database saved: {self.data['metadata']['total_persons']} persons")
            return True
        except Exception as e:
            print(f"❌ Database save error: {e}")
            return False
    
    def get_all_persons(self) -> Dict[str, Any]:
        """Get all persons data"""
        return self.data['persons']
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            'total_persons': len(self.data['persons']),
            'total_embeddings': sum(person['image_count'] for person in self.data['persons'].values()),
            'created_date': self.data['metadata']['created_date'],
            'last_modified': self.data['metadata']['last_modified'],
            'processing_mode': self.data['metadata'].get('processing_mode', 'unknown')
        }


def process_images_from_folder(folder_path: str, processor: DirectTFLiteFaceProcessor) -> List[np.ndarray]:
    """Process all images in a folder"""
    embeddings = []
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(supported_formats)]
    
    if not image_files:
        print("❌ No supported image files found")
        return []
    
    print(f"📸 Processing {len(image_files)} images...")
    
    for filename in image_files:
        try:
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"⚠️ Could not load {filename}")
                continue
            
            # Detect faces
            faces = processor.detect_faces_in_image(image)
            
            if not faces:
                print(f"⚠️ No faces found in {filename}")
                continue
            
            # Use the first (or best) face
            face_info = faces[0]
            features = processor.extract_features(face_info)
            embeddings.append(features)
            
            print(f"✅ Processed {filename} ({face_info['method']}) - confidence: {face_info['confidence']:.2f}")
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue
    
    return embeddings


def main():
    """Main CLI interface"""
    try:
        # Initialize direct TFLite processor
        print("🚀 Initializing Direct TensorFlow Lite Face Enrolment System...")
        processor = DirectTFLiteFaceProcessor()
        database = DirectTFLiteDatabase()
        
        # Show processing mode
        proc_info = processor.get_processing_info()
        print(f"\n📊 System Status:")
        print(f"   Current Mode: {proc_info['current_mode']}")
        print(f"   TFLite Available: {proc_info['tflite_available']}")
        print(f"   Face Detection Model: {'✅' if proc_info['face_detection_model_loaded'] else '❌'}")
        print(f"   Face Landmarks Model: {'✅' if proc_info['face_landmarks_model_loaded'] else '❌'}")
        print(f"   OpenCV Available: {proc_info['opencv_available']}")
        
        while True:
            print("\n" + "="*50)
            print("🎯 Direct TensorFlow Lite Face Enrolment System")
            print("="*50)
            print("1. 👤 Add New Registration")
            print("2. 💾 Save Database & Exit")
            print("3. 📊 View Current Registrations")
            print("4. ❌ Exit without saving")
            print("="*50)
            
            try:
                choice = input("Enter your choice (1-4): ").strip()
                
                if choice == '1':
                    # Add new registration
                    folder_path = input("📁 Enter folder path containing images: ").strip().strip('"\'')
                    
                    if not os.path.exists(folder_path):
                        print("❌ Folder not found!")
                        continue
                    
                    person_id = input("👤 Enter person ID: ").strip()
                    
                    if not person_id:
                        print("❌ Person ID cannot be empty!")
                        continue
                    
                    if person_id in database.get_all_persons():
                        print("⚠️ Person already exists! Use a different ID.")
                        continue
                    
                    print(f"🔄 Processing images for: {person_id}")
                    embeddings = process_images_from_folder(folder_path, processor)
                    
                    if embeddings:
                        proc_info = processor.get_processing_info()
                        database.add_person(person_id, embeddings, folder_path, proc_info)
                        print(f"✅ Successfully enrolled {person_id} with {len(embeddings)} face embeddings")
                    else:
                        print("❌ No faces could be processed!")
                
                elif choice == '2':
                    # Save and exit
                    print("💾 Saving database...")
                    if database.save_database():
                        stats = database.get_statistics()
                        print(f"✅ Database saved successfully!")
                        print(f"   Total persons: {stats['total_persons']}")
                        print(f"   Total embeddings: {stats['total_embeddings']}")
                        print(f"   Processing mode: {stats['processing_mode']}")
                    break
                
                elif choice == '3':
                    # View registrations
                    persons = database.get_all_persons()
                    if not persons:
                        print("📝 No registrations found")
                    else:
                        print(f"\n📊 Current Registrations ({len(persons)} persons):")
                        print("-" * 80)
                        for person_id, data in persons.items():
                            proc_mode = data.get('processing_info', {}).get('current_mode', 'unknown')
                            tflite_used = '🎯' if 'tflite' in proc_mode else '🔄'
                            print(f"{tflite_used} {person_id:20} | Images: {data['image_count']:2} | Mode: {proc_mode}")
                
                elif choice == '4':
                    # Exit without saving
                    print("❌ Exiting without saving...")
                    break
                
                else:
                    print("❌ Invalid choice! Please enter 1-4.")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️ Operation cancelled by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"💥 System error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
