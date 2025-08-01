import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceEncodingBuilder:
    def __init__(self, model_path="mediapipe_models/"):
        """
        Initialize the Face Encoding Builder with MediaPipe face detection
        
        Args:
            model_path (str): Path to MediaPipe models directory
        """
        self.model_path = model_path
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face detection and face mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Storage for face encodings and IDs
        self.face_encodings = []
        self.face_ids = []
        
    def extract_face_landmarks(self, image):
        """
        Extract face landmarks using MediaPipe Face Mesh
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            numpy array: Flattened face landmarks or None if no face found
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            # Convert landmarks to numpy array
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks)
        return None
    
    def detect_face(self, image):
        """
        Detect face in image using MediaPipe Face Detection
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            bool: True if face detected, False otherwise
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        return results.detections is not None and len(results.detections) > 0
    
    def process_user_folder(self, folder_path):
        """
        Process all images in a user folder to extract face encodings
        
        Args:
            folder_path (str): Path to user's image folder
            
        Returns:
            tuple: (user_id, list of face encodings)
        """
        folder_path = Path(folder_path)
        user_id = folder_path.name
        
        logger.info(f"Processing user: {user_id}")
        
        # Supported image formats
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = [f for f in folder_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            logger.warning(f"No image files found in {folder_path}")
            return user_id, []
        
        user_encodings = []
        processed_count = 0
        
        for image_file in image_files:
            try:
                # Read image
                image = cv2.imread(str(image_file))
                if image is None:
                    logger.warning(f"Could not read image: {image_file}")
                    continue
                
                # Check if face is detected
                if not self.detect_face(image):
                    logger.warning(f"No face detected in: {image_file}")
                    continue
                
                # Extract face landmarks/encoding
                face_encoding = self.extract_face_landmarks(image)
                if face_encoding is not None:
                    user_encodings.append(face_encoding)
                    processed_count += 1
                    logger.info(f"Processed: {image_file.name}")
                else:
                    logger.warning(f"Could not extract landmarks from: {image_file}")
                    
            except Exception as e:
                logger.error(f"Error processing {image_file}: {str(e)}")
                continue
        
        logger.info(f"Successfully processed {processed_count} images for user: {user_id}")
        return user_id, user_encodings
    
    def add_user_registration(self, folder_path):
        """
        Add a new user registration from folder path
        
        Args:
            folder_path (str): Path to user's image folder
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            if not os.path.exists(folder_path):
                logger.error(f"Folder path does not exist: {folder_path}")
                return False
            
            user_id, encodings = self.process_user_folder(folder_path)
            
            if not encodings:
                logger.error(f"No valid face encodings found for user: {user_id}")
                return False
            
            # Add encodings to storage
            for encoding in encodings:
                self.face_encodings.append(encoding)
                self.face_ids.append(user_id)
            
            logger.info(f"Added {len(encodings)} face encodings for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error in user registration: {str(e)}")
            return False
    
    def save_model(self, output_file="face_encodings.pkl"):
        """
        Save face encodings and IDs to pickle file
        
        Args:
            output_file (str): Output pickle file name
        """
        try:
            if not self.face_encodings:
                logger.warning("No face encodings to save!")
                return False
            
            # Prepare data to save
            data = {
                'encodings': np.array(self.face_encodings),
                'ids': self.face_ids,
                'total_users': len(set(self.face_ids)),
                'total_encodings': len(self.face_encodings)
            }
            
            # Save to pickle file
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Model saved successfully to: {output_file}")
            logger.info(f"Total users: {data['total_users']}")
            logger.info(f"Total face encodings: {data['total_encodings']}")
            
            # Print summary
            unique_ids = set(self.face_ids)
            for user_id in unique_ids:
                count = self.face_ids.count(user_id)
                logger.info(f"User '{user_id}': {count} face encodings")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def get_registration_summary(self):
        """
        Get summary of current registrations
        
        Returns:
            dict: Summary information
        """
        unique_ids = set(self.face_ids)
        summary = {
            'total_users': len(unique_ids),
            'total_encodings': len(self.face_encodings),
            'users': {}
        }
        
        for user_id in unique_ids:
            summary['users'][user_id] = self.face_ids.count(user_id)
        
        return summary

def main():
    """
    Main function to run the face encoding builder
    """
    print("=" * 60)
    print("🎯 FACE RECOGNITION ATTENDANCE SYSTEM - MODEL BUILDER")
    print("=" * 60)
    print()
    
    # Initialize the builder
    builder = FaceEncodingBuilder()
    
    while True:
        print("\n" + "─" * 50)
        print("📋 REGISTRATION OPTIONS:")
        print("─" * 50)
        print("1. 👤 Add New Registration")
        print("2. 💾 Don't want to register further (Save Model)")
        print("3. 📊 View Current Registrations")
        print("4. ❌ Exit without saving")
        print()
        
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                print("\n" + "─" * 30)
                print("📁 ADD NEW USER REGISTRATION")
                print("─" * 30)
                
                folder_path = input("Enter the path to user's folder: ").strip()
                
                if not folder_path:
                    print("❌ Error: Folder path cannot be empty!")
                    continue
                
                print(f"\n🔄 Processing folder: {folder_path}")
                print("Please wait...")
                
                success = builder.add_user_registration(folder_path)
                
                if success:
                    print("✅ User registration added successfully!")
                    
                    # Show current summary
                    summary = builder.get_registration_summary()
                    print(f"\n📊 Current Status:")
                    print(f"   Total Users: {summary['total_users']}")
                    print(f"   Total Face Encodings: {summary['total_encodings']}")
                else:
                    print("❌ Failed to add user registration!")
                    print("   Please check the folder path and ensure it contains valid face images.")
            
            elif choice == '2':
                print("\n" + "─" * 30)
                print("💾 SAVING MODEL")
                print("─" * 30)
                
                summary = builder.get_registration_summary()
                
                if summary['total_users'] == 0:
                    print("❌ Error: No users registered yet!")
                    print("   Please add at least one user registration before saving.")
                    continue
                
                # Show final summary
                print("📊 Final Registration Summary:")
                print(f"   Total Users: {summary['total_users']}")
                print(f"   Total Face Encodings: {summary['total_encodings']}")
                print("\n👥 User Details:")
                for user_id, count in summary['users'].items():
                    print(f"   • {user_id}: {count} face encodings")
                
                print(f"\n🔄 Saving model...")
                
                # Get output filename
                output_file = input("\nEnter output filename (or press Enter for 'face_encodings.pkl'): ").strip()
                if not output_file:
                    output_file = "face_encodings.pkl"
                
                if not output_file.endswith('.pkl'):
                    output_file += '.pkl'
                
                success = builder.save_model(output_file)
                
                if success:
                    print(f"\n🎉 SUCCESS!")
                    print(f"   Model saved as: {output_file}")
                    print(f"   Ready for face recognition attendance system!")
                    break
                else:
                    print("❌ Failed to save model!")
            
            elif choice == '3':
                print("\n" + "─" * 30)
                print("📊 CURRENT REGISTRATIONS")
                print("─" * 30)
                
                summary = builder.get_registration_summary()
                
                if summary['total_users'] == 0:
                    print("📝 No users registered yet.")
                else:
                    print(f"Total Users: {summary['total_users']}")
                    print(f"Total Face Encodings: {summary['total_encodings']}")
                    print("\n👥 Registered Users:")
                    for user_id, count in summary['users'].items():
                        print(f"   • {user_id}: {count} face encodings")
            
            elif choice == '4':
                print("\n👋 Exiting without saving...")
                break
            
            else:
                print("❌ Invalid choice! Please enter 1, 2, 3, or 4.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Program interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"❌ An error occurred: {str(e)}")
            logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
