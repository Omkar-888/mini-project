"""
Face Recognition Attendance System
Uses the enrollment database to identify faces from input images
Supports multiple image formats and provides similarity scoring
"""

import os
import sys
import cv2
import numpy as np
import pickle
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import warnings
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Import the face processor from enrollment system
try:
    from enrolment import DirectTFLiteFaceProcessor
    print("✅ Imported face processor from enrollment system")
except ImportError as e:
    print(f"❌ Could not import face processor: {e}")
    sys.exit(1)


class AttendanceDatabase:
    """Manages attendance records storage and retrieval"""
    
    def __init__(self, attendance_db_path: str = "database/attendance.pkl"):
        self.attendance_db_path = attendance_db_path
        self.attendance_data = {
            'metadata': {
                'created_date': datetime.now().isoformat(),
                'total_records': 0,
                'version': '1.0'
            },
            'records': []
        }
        self.load_attendance_database()
    
    def load_attendance_database(self):
        """Load existing attendance database"""
        try:
            if os.path.exists(self.attendance_db_path):
                with open(self.attendance_db_path, 'rb') as f:
                    self.attendance_data = pickle.load(f)
                print(f"✅ Attendance database loaded: {len(self.attendance_data['records'])} records")
            else:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.attendance_db_path), exist_ok=True)
                print("📝 New attendance database will be created")
        except Exception as e:
            print(f"❌ Attendance database load error: {e}")
            print("📝 Starting with empty attendance database")
    
    def add_attendance_record(self, person_id: str, confidence: float, 
                            image_path: str = "", detection_method: str = "manual"):
        """Add a new attendance record"""
        record = {
            'person_id': person_id,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime("%Y-%m-%d"),
            'time': datetime.now().strftime("%H:%M:%S"),
            'image_path': image_path,
            'detection_method': detection_method,
            'record_id': len(self.attendance_data['records']) + 1
        }
        
        self.attendance_data['records'].append(record)
        self.attendance_data['metadata']['total_records'] = len(self.attendance_data['records'])
        self.attendance_data['metadata']['last_updated'] = datetime.now().isoformat()
        
        print(f"✅ Attendance recorded: {person_id} at {record['time']}")
        
    def save_attendance_database(self):
        """Save attendance database to file"""
        try:
            # Create backup if exists
            if os.path.exists(self.attendance_db_path):
                backup_path = self.attendance_db_path.replace('.pkl', '_backup.pkl')
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(self.attendance_db_path, backup_path)
            
            # Save new database
            with open(self.attendance_db_path, 'wb') as f:
                pickle.dump(self.attendance_data, f, protocol=4)
            
            print(f"✅ Attendance database saved: {self.attendance_data['metadata']['total_records']} records")
            return True
            
        except Exception as e:
            print(f"❌ Error saving attendance database: {e}")
            return False
    
    def get_attendance_by_date(self, date: str = None) -> List[Dict[str, Any]]:
        """Get attendance records for a specific date (YYYY-MM-DD format)"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        return [record for record in self.attendance_data['records'] if record['date'] == date]
    
    def get_attendance_summary(self) -> Dict[str, Any]:
        """Get attendance summary statistics"""
        records = self.attendance_data['records']
        today = datetime.now().strftime("%Y-%m-%d")
        
        today_records = self.get_attendance_by_date(today)
        unique_persons_today = len(set(record['person_id'] for record in today_records))
        
        return {
            'total_records': len(records),
            'today_records': len(today_records),
            'unique_persons_today': unique_persons_today,
            'last_record': records[-1] if records else None,
            'database_path': self.attendance_db_path
        }


class AttendanceSystem:
    """Face recognition attendance system"""
    
    def __init__(self, database_path: str = "database/enrolment.pkl", 
                 similarity_threshold: float = 0.7):
        """
        Initialize attendance system
        
        Args:
            database_path: Path to enrollment database
            similarity_threshold: Minimum similarity score for identification (0-1)
        """
        self.database_path = database_path
        self.similarity_threshold = similarity_threshold
        self.enrollment_data = {}
        self.face_processor = None
        
        # Initialize attendance database
        self.attendance_db = AttendanceDatabase()
        
        # Initialize face processor
        print("🚀 Initializing Face Recognition Attendance System...")
        try:
            self.face_processor = DirectTFLiteFaceProcessor()
            print("✅ Face processor initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize face processor: {e}")
            raise
        
        # Load enrollment database
        self.load_database()
    
    def load_database(self):
        """Load the enrollment database"""
        try:
            if not os.path.exists(self.database_path):
                print(f"❌ Database not found at {self.database_path}")
                print("💡 Please run the enrollment system first to create the database")
                return
            
            with open(self.database_path, 'rb') as f:
                data = pickle.load(f)
            
            self.enrollment_data = data.get('persons', {})
            metadata = data.get('metadata', {})
            
            total_persons = len(self.enrollment_data)
            print(f"✅ Database loaded successfully")
            print(f"📊 Total enrolled persons: {total_persons}")
            print(f"🔧 Processing mode: {metadata.get('processing_mode', 'unknown')}")
            print(f"📅 Last modified: {metadata.get('last_modified', 'unknown')}")
            
            if total_persons == 0:
                print("⚠️ No persons found in database. Please enroll some people first.")
                
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            self.enrollment_data = {}
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1 (1 = identical, 0 = completely different)
        """
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            normalized1 = embedding1 / norm1
            normalized2 = embedding2 / norm2
            
            # Calculate cosine similarity
            similarity = np.dot(normalized1, normalized2)
            
            # Convert to 0-1 range (cosine similarity is -1 to 1)
            similarity = (similarity + 1) / 2
            
            return float(np.clip(similarity, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def identify_person(self, query_embedding: np.ndarray) -> Tuple[str, float]:
        """
        Identify a person by comparing against all enrolled embeddings
        
        Args:
            query_embedding: Face embedding to identify
            
        Returns:
            Tuple of (person_id, confidence_score) or ("unknown", 0.0)
        """
        if not self.enrollment_data:
            return "unknown", 0.0
        
        best_match = "unknown"
        best_score = 0.0
        
        for person_id, person_data in self.enrollment_data.items():
            # Get all embeddings for this person
            embeddings = person_data.get('embeddings', [])
            
            if not embeddings:
                continue
            
            # Compare against all embeddings for this person
            scores = []
            for stored_embedding in embeddings:
                stored_emb_array = np.array(stored_embedding, dtype=np.float32)
                similarity = self.cosine_similarity(query_embedding, stored_emb_array)
                scores.append(similarity)
            
            # Use the maximum similarity score for this person
            person_best_score = max(scores) if scores else 0.0
            
            # Also compare with aggregated embedding if available
            if 'aggregated_embedding' in person_data:
                aggregated_emb = np.array(person_data['aggregated_embedding'], dtype=np.float32)
                agg_similarity = self.cosine_similarity(query_embedding, aggregated_emb)
                person_best_score = max(person_best_score, agg_similarity)
            
            # Update best match
            if person_best_score > best_score:
                best_score = person_best_score
                best_match = person_id
        
        # Check if score meets threshold
        if best_score < self.similarity_threshold:
            return "unknown", best_score
        
        return best_match, best_score
    
    def process_single_image(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Process a single image and identify all faces
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of identification results for each face found
        """
        results = []
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                return results
            
            # Detect faces
            faces = self.face_processor.detect_faces_in_image(image)
            
            if not faces:
                print(f"⚠️ No faces found in {os.path.basename(image_path)}")
                return results
            
            print(f"👤 Found {len(faces)} face(s) in {os.path.basename(image_path)}")
            
            # Process each face
            for i, face_info in enumerate(faces):
                try:
                    # Extract features
                    embedding = self.face_processor.extract_features(face_info)
                    
                    # Identify person
                    person_id, confidence = self.identify_person(embedding)
                    
                    # Store result
                    result = {
                        'image_path': image_path,
                        'face_index': i,
                        'person_id': person_id,
                        'confidence': confidence,
                        'bbox': face_info['bbox'],
                        'detection_confidence': face_info['confidence'],
                        'detection_method': face_info['method'],
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    results.append(result)
                    
                    # Print result
                    if person_id != "unknown":
                        print(f"✅ Face {i+1}: {person_id} (confidence: {confidence:.3f})")
                    else:
                        print(f"❓ Face {i+1}: Unknown person (best score: {confidence:.3f})")
                        
                except Exception as e:
                    print(f"❌ Error processing face {i+1}: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error processing image {image_path}: {e}")
        
        return results
    
    def process_images_from_location(self, location_path: str, single_person_attempt: bool = False) -> List[Dict[str, Any]]:
        """
        Process all images from a specified location
        
        Args:
            location_path: Path to directory containing images
            single_person_attempt: If True, treats all images as one person attendance attempt
            
        Returns:
            List of identification results for all faces found
        """
        if not os.path.exists(location_path):
            print(f"❌ Location does not exist: {location_path}")
            return []
        
        # Supported image formats
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        all_results = []
        
        if os.path.isfile(location_path):
            # Single file
            if Path(location_path).suffix.lower() in supported_formats:
                results = self.process_single_image(location_path)
                all_results.extend(results)
            else:
                print(f"⚠️ Unsupported file format: {location_path}")
        else:
            # Directory
            image_files = []
            for file_path in Path(location_path).rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                    image_files.append(str(file_path))
            
            if not image_files:
                print(f"⚠️ No supported image files found in: {location_path}")
                return []
            
            print(f"📸 Found {len(image_files)} image file(s)")
            print(f"🔍 Processing images from: {location_path}")
            
            if single_person_attempt:
                print(f"👤 Treating all images as one person attendance attempt")
            
            print("-" * 60)
            
            # Process each image
            for image_path in sorted(image_files):
                results = self.process_single_image(image_path)
                all_results.extend(results)
            
            # If single person attempt, consolidate results
            if single_person_attempt and all_results:
                all_results = self.consolidate_single_person_results(all_results, location_path)
        
        return all_results
    
    def consolidate_single_person_results(self, results: List[Dict[str, Any]], folder_path: str) -> List[Dict[str, Any]]:
        """
        Consolidate multiple image results into a single person attendance attempt
        Uses majority vote and highest confidence for final identification
        """
        if not results:
            return []
        
        # Count identifications and find best confidence
        person_votes = {}
        confidences = {}
        
        for result in results:
            person_id = result['person_id']
            confidence = result['confidence']
            
            if person_id not in person_votes:
                person_votes[person_id] = 0
                confidences[person_id] = []
            
            person_votes[person_id] += 1
            confidences[person_id].append(confidence)
        
        # Find the person with most votes
        best_person = max(person_votes, key=person_votes.get)
        
        # Calculate average confidence for best person
        avg_confidence = sum(confidences[best_person]) / len(confidences[best_person])
        max_confidence = max(confidences[best_person])
        
        # Create consolidated result
        consolidated_result = {
            'image_path': folder_path,
            'face_index': 0,
            'person_id': best_person,
            'confidence': max_confidence,  # Use highest confidence
            'average_confidence': avg_confidence,
            'vote_count': person_votes[best_person],
            'total_images': len(results),
            'bbox': results[0]['bbox'],  # Use first result's bbox
            'detection_confidence': max(r['detection_confidence'] for r in results),
            'detection_method': 'multipart_consolidated',
            'timestamp': datetime.now().isoformat(),
            'voting_details': {
                'all_votes': person_votes,
                'all_confidences': {k: sum(v)/len(v) for k, v in confidences.items()}
            }
        }
        
        print(f"🗳️  Voting Results:")
        for person, votes in sorted(person_votes.items(), key=lambda x: x[1], reverse=True):
            avg_conf = sum(confidences[person]) / len(confidences[person])
            print(f"   • {person}: {votes} votes, avg confidence: {avg_conf:.3f}")
        
        print(f"🎯 Final Decision: {best_person} (confidence: {max_confidence:.3f})")
        
        return [consolidated_result]
    
    def generate_attendance_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate attendance report from identification results
        
        Args:
            results: List of identification results
            
        Returns:
            Attendance report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_faces_detected': len(results),
            'identified_persons': {},
            'unknown_faces': 0,
            'summary': {}
        }
        
        # Process results
        for result in results:
            person_id = result['person_id']
            
            if person_id == "unknown":
                report['unknown_faces'] += 1
            else:
                if person_id not in report['identified_persons']:
                    report['identified_persons'][person_id] = {
                        'detections': [],
                        'best_confidence': 0.0,
                        'detection_count': 0
                    }
                
                person_data = report['identified_persons'][person_id]
                person_data['detections'].append(result)
                person_data['detection_count'] += 1
                person_data['best_confidence'] = max(
                    person_data['best_confidence'], 
                    result['confidence']
                )
        
        # Generate summary
        report['summary'] = {
            'unique_persons_identified': len(report['identified_persons']),
            'total_unknown_faces': report['unknown_faces'],
            'identification_rate': len(report['identified_persons']) / len(results) if results else 0.0
        }
        
        return report
    
    def print_attendance_report(self, report: Dict[str, Any]):
        """Print formatted attendance report"""
        print("\n" + "=" * 60)
        print("📊 ATTENDANCE REPORT")
        print("=" * 60)
        print(f"📅 Generated: {report['timestamp']}")
        print(f"👥 Total faces detected: {report['total_faces_detected']}")
        print(f"✅ Unique persons identified: {report['summary']['unique_persons_identified']}")
        print(f"❓ Unknown faces: {report['summary']['total_unknown_faces']}")
        print(f"📈 Identification rate: {report['summary']['identification_rate']:.1%}")
        print("-" * 60)
        
        # List identified persons
        if report['identified_persons']:
            print("👤 IDENTIFIED PERSONS:")
            for person_id, person_data in sorted(report['identified_persons'].items()):
                print(f"   • {person_id}: {person_data['detection_count']} detection(s), "
                      f"best confidence: {person_data['best_confidence']:.3f}")
        else:
            print("👤 No persons identified")
        
        if report['summary']['total_unknown_faces'] > 0:
            print(f"❓ {report['summary']['total_unknown_faces']} unknown face(s) detected")
        
        print("=" * 60)
    
    def record_attendance_from_results(self, results: List[Dict[str, Any]]) -> int:
        """Record attendance for identified persons from results"""
        recorded_count = 0
        
        for result in results:
            person_id = result['person_id']
            
            if person_id != "unknown":
                self.attendance_db.add_attendance_record(
                    person_id=person_id,
                    confidence=result['confidence'],
                    image_path=result.get('image_path', ''),
                    detection_method=result.get('detection_method', 'auto')
                )
                recorded_count += 1
        
        if recorded_count > 0:
            self.attendance_db.save_attendance_database()
            
        return recorded_count
    
    def manual_attendance_entry(self):
        """Manual attendance entry for enrolled persons"""
        if not self.enrollment_data:
            print("❌ No enrolled persons found")
            return
        
        print("\n👥 ENROLLED PERSONS:")
        enrolled_persons = list(self.enrollment_data.keys())
        for i, person_id in enumerate(enrolled_persons, 1):
            print(f"{i}. {person_id}")
        print("0. 🔙 Back to main menu")
        
        try:
            choice = input(f"\nSelect person (0-{len(enrolled_persons)}) or enter name: ").strip()
            
            # Check if user wants to go back
            if choice == "0":
                print("🔙 Returning to main menu...")
                return
            
            # Check if it's a number
            if choice.isdigit():
                person_index = int(choice) - 1
                if 0 <= person_index < len(enrolled_persons):
                    person_id = enrolled_persons[person_index]
                else:
                    print("❌ Invalid selection")
                    return
            else:
                # Check if name exists
                if choice in self.enrollment_data:
                    person_id = choice
                else:
                    print("❌ Person not found in enrollment database")
                    return
            
            # Record attendance
            self.attendance_db.add_attendance_record(
                person_id=person_id,
                confidence=1.0,  # Manual entry gets perfect confidence
                detection_method="manual"
            )
            
            if self.attendance_db.save_attendance_database():
                print(f"✅ Manual attendance recorded for {person_id}")
            else:
                print("❌ Failed to save attendance record")
                
        except Exception as e:
            print(f"❌ Error in manual attendance entry: {e}")
    
    def show_attendance_summary(self):
        """Display attendance summary"""
        summary = self.attendance_db.get_attendance_summary()
        today_records = self.attendance_db.get_attendance_by_date()
        
        print("\n" + "=" * 50)
        print("📊 ATTENDANCE SUMMARY")
        print("=" * 50)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"📈 Total records (all time): {summary['total_records']}")
        print(f"👥 Today's records: {summary['today_records']}")
        print(f"🎯 Unique persons today: {summary['unique_persons_today']}")
        
        if summary['last_record']:
            last_record = summary['last_record']
            print(f"🕒 Last record: {last_record['person_id']} at {last_record['time']}")
        
        if today_records:
            print("\n📋 TODAY'S ATTENDANCE:")
            for record in today_records:
                method_icon = "🤖" if record['detection_method'] == "auto" else "✋"
                print(f"   {method_icon} {record['person_id']} - {record['time']} "
                      f"(confidence: {record['confidence']:.3f})")
        else:
            print("\n📋 No attendance records for today")
        
        print("=" * 50)
    
    def export_attendance_report(self, date: str = None):
        """Export attendance report in JSON format for a specific date"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        records = self.attendance_db.get_attendance_by_date(date)
        
        if not records:
            print(f"⚠️ No attendance records found for {date}")
            return
        
        # Ask for save location
        print(f"\n📋 Exporting attendance report for {date}")
        print("💾 Choose save location:")
        print("1. Current directory")
        print("2. database/ directory")
        print("3. Custom path")
        print("0. 🔙 Back to main menu")
        
        try:
            location_choice = input("Select location (0-3): ").strip()
            
            if location_choice == "0":
                print("🔙 Returning to main menu...")
                return
            
            # Determine save path
            if location_choice == "1":
                save_dir = "."
            elif location_choice == "2":
                save_dir = "database"
                os.makedirs(save_dir, exist_ok=True)
            elif location_choice == "3":
                save_dir = input("Enter custom directory path: ").strip()
                if not save_dir:
                    save_dir = "."
                # Create directory if it doesn't exist
                try:
                    os.makedirs(save_dir, exist_ok=True)
                except Exception as e:
                    print(f"❌ Could not create directory: {e}")
                    save_dir = "."
            else:
                print("❌ Invalid selection, using current directory")
                save_dir = "."
            
            # Create JSON report
            report_data = {
                "metadata": {
                    "report_date": date,
                    "generated_at": datetime.now().isoformat(),
                    "total_records": len(records),
                    "unique_persons": len(set(r['person_id'] for r in records)),
                    "report_type": "daily_attendance"
                },
                "summary": {
                    "total_attendance": len(records),
                    "unique_attendees": list(set(r['person_id'] for r in records)),
                    "detection_methods": {
                        method: len([r for r in records if r['detection_method'] == method])
                        for method in set(r['detection_method'] for r in records)
                    }
                },
                "attendance_records": records,
                "statistics": {
                    "average_confidence": sum(r['confidence'] for r in records) / len(records),
                    "highest_confidence": max(r['confidence'] for r in records),
                    "lowest_confidence": min(r['confidence'] for r in records),
                    "first_attendance": min(r['timestamp'] for r in records),
                    "last_attendance": max(r['timestamp'] for r in records)
                }
            }
            
            # Generate filename
            report_filename = f"attendance_report_{date.replace('-', '')}.json"
            report_path = os.path.join(save_dir, report_filename)
            
            # Save JSON file
            import json
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ JSON attendance report exported to: {report_path}")
            print(f"📊 Report contains {len(records)} records for {len(set(r['person_id'] for r in records))} unique persons")
            
        except Exception as e:
            print(f"❌ Error exporting report: {e}")


def main():
    """Main CLI interface for attendance system"""
    print("🎯 Face Recognition Attendance System")
    print("=" * 50)
    
    try:
        # Initialize attendance system
        attendance_system = AttendanceSystem()
        
        if not attendance_system.enrollment_data:
            print("\n❌ No enrollment data found. Please run the enrollment system first.")
            return
        
        while True:
            print(f"\n📋 ATTENDANCE SYSTEM MENU")
            print("-" * 35)
            print("1. 📸 Process images & record attendance")
            print("2. ✋ Manual attendance entry")
            print("3. 📊 Show attendance summary")
            print("4. 📋 Export attendance report")
            print("5. ⚙️ Change similarity threshold")
            print("6. � Show enrolled persons")
            print("7. ❌ Exit")
            
            try:
                choice = input("\nSelect option (1-7): ").strip()
                
                if choice == '1':
                    # Process images and record attendance
                    location = input("\n📂 Enter path to images (file or directory): ").strip()
                    if not location:
                        print("⚠️ Please provide a valid path")
                        continue
                    
                    # Ask if this is a single person attempt
                    single_person = input("\n👤 Treat all images as one person attendance attempt? (y/n): ").strip().lower()
                    single_person_attempt = single_person == 'y'
                    
                    print(f"\n🔍 Processing images from: {location}")
                    if single_person_attempt:
                        print("👤 Using multipart consolidation mode")
                    
                    results = attendance_system.process_images_from_location(location, single_person_attempt)
                    
                    if results:
                        # Generate and display report
                        report = attendance_system.generate_attendance_report(results)
                        attendance_system.print_attendance_report(report)
                        
                        # Ask if user wants to record attendance
                        record_choice = input("\n📝 Record attendance for identified persons? (y/n): ").strip().lower()
                        if record_choice == 'y':
                            recorded_count = attendance_system.record_attendance_from_results(results)
                            print(f"✅ Recorded attendance for {recorded_count} person(s)")
                        
                        # Option to save detailed results
                        save_choice = input("\n💾 Save detailed results to file? (y/n): ").strip().lower()
                        if save_choice == 'y':
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            results_file = f"recognition_results_{timestamp}.pkl"
                            
                            try:
                                with open(results_file, 'wb') as f:
                                    pickle.dump({
                                        'results': results,
                                        'report': report,
                                        'settings': {
                                            'similarity_threshold': attendance_system.similarity_threshold,
                                            'database_path': attendance_system.database_path,
                                            'single_person_attempt': single_person_attempt
                                        }
                                    }, f)
                                print(f"✅ Results saved to: {results_file}")
                            except Exception as e:
                                print(f"❌ Error saving results: {e}")
                    else:
                        print("\n⚠️ No faces detected or processed")
                
                elif choice == '2':
                    # Manual attendance entry
                    attendance_system.manual_attendance_entry()
                
                elif choice == '3':
                    # Show attendance summary
                    attendance_system.show_attendance_summary()
                
                elif choice == '4':
                    # Export attendance report
                    date_input = input("\n📅 Enter date (YYYY-MM-DD) or press Enter for today: ").strip()
                    if not date_input:
                        date_input = None
                    attendance_system.export_attendance_report(date_input)
                
                elif choice == '5':
                    # Change threshold
                    current = attendance_system.similarity_threshold
                    print(f"\n🎯 Current similarity threshold: {current:.2f}")
                    print("💡 Lower values = more lenient identification")
                    print("💡 Higher values = more strict identification")
                    
                    try:
                        new_threshold = float(input("Enter new threshold (0.0-1.0): ").strip())
                        if 0.0 <= new_threshold <= 1.0:
                            attendance_system.similarity_threshold = new_threshold
                            print(f"✅ Threshold updated to: {new_threshold:.2f}")
                        else:
                            print("❌ Threshold must be between 0.0 and 1.0")
                    except ValueError:
                        print("❌ Invalid threshold value")
                
                elif choice == '6':
                    # Show enrolled persons
                    print(f"\n👥 ENROLLED PERSONS ({len(attendance_system.enrollment_data)})")
                    print("-" * 40)
                    
                    if attendance_system.enrollment_data:
                        for person_id, person_data in sorted(attendance_system.enrollment_data.items()):
                            image_count = person_data.get('image_count', 0)
                            enroll_date = person_data.get('enrolment_date', 'Unknown')
                            method = person_data.get('processing_info', {}).get('current_mode', 'Unknown')
                            
                            print(f"• {person_id}")
                            print(f"  - Images: {image_count}")
                            print(f"  - Enrolled: {enroll_date[:19] if enroll_date != 'Unknown' else 'Unknown'}")
                            print(f"  - Method: {method}")
                    else:
                        print("No persons enrolled")
                
                elif choice == '7':
                    print("\n👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid choice. Please select 1-7.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
    
    except Exception as e:
        print(f"❌ System initialization failed: {e}")


if __name__ == "__main__":
    main()