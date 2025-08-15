"""
FastAPI Microservice for Face Recognition Attendance System
Provides REST API endpoints for face recognition and attendance tracking
Uses the AttendanceSystem class for face identification with cosine similarity
Compatible with main.py structure but with full functionality
"""

import os
import sys
import tempfile
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Import the attendance system
try:
    from attendance_system import AttendanceSystem
    print("✅ Imported AttendanceSystem from attendance_system")
except ImportError as e:
    print(f"❌ Could not import AttendanceSystem: {e}")
    print("❌ Please install required packages: pip install fastapi uvicorn python-multipart")
    sys.exit(1)

# Try to import FastAPI components
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, status
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    print("❌ FastAPI not available. Please install: pip install fastapi uvicorn python-multipart")
    FASTAPI_AVAILABLE = False

if not FASTAPI_AVAILABLE:
    # Create dummy classes for development without FastAPI
    class BaseModel:
        pass
    
    class FastAPI:
        def __init__(self, **kwargs):
            pass
    
    def File(*args, **kwargs):
        pass
    
    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition Attendance API",
    description="REST API for face recognition attendance system using MediaPipe TFLite",
    version="1.0.0"
) if FASTAPI_AVAILABLE else None

# Global attendance system instance
attendance_system: Optional[AttendanceSystem] = None

# Response models - Compatible with main.py structure
class PredictionResponse(BaseModel):
    """Prediction response schema - matches main.py"""
    user_id: str  # Changed from int to str to match person names
    confidence_score: float

class BatchPredictionResponse(BaseModel):
    """Batch processing results"""
    total_images: int
    total_faces: int
    results: List[PredictionResponse]
    processing_time_ms: float
    timestamp: str

# Global variables for storing latest results - matches main.py pattern
res: PredictionResponse = PredictionResponse(user_id="unknown", confidence_score=0.0)
latest_batch_results: Optional[BatchPredictionResponse] = None

def initialize_system():
    """Initialize the attendance system"""
    global attendance_system
    
    try:
        print("🚀 Initializing Face Recognition Attendance System...")
        
        # Initialize attendance system
        attendance_system = AttendanceSystem(
            database_path="database/enrolment.pkl",
            similarity_threshold=0.7
        )
        
        if not attendance_system.enrollment_data:
            print("⚠️ Warning: No enrollment data found in database")
            return False
        else:
            print(f"✅ System ready with {len(attendance_system.enrollment_data)} enrolled persons")
            return True
            
    except Exception as e:
        print(f"❌ Failed to initialize attendance system: {e}")
        attendance_system = None
        return False

if FASTAPI_AVAILABLE:
    @app.on_event("startup")
    async def startup_event():
        """Initialize the attendance system on startup"""
        initialize_system()

def save_upload_file(upload_file) -> str:
    """Save uploaded file to temporary location"""
    try:
        # Create a temporary file
        suffix = os.path.splitext(upload_file.filename)[1] if upload_file.filename else '.jpg'
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        # Copy file contents
        shutil.copyfileobj(upload_file.file, temp_file)
        temp_file.close()
        
        return temp_file.name
        
    except Exception as e:
        if FASTAPI_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error saving uploaded file: {str(e)}"
            )
        else:
            raise Exception(f"Error saving uploaded file: {str(e)}")

def cleanup_temp_file(file_path: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception:
        pass  # Ignore cleanup errors

def user_id_prediction(frames: List, single_person_attempt: bool = True) -> None:
    """
    Mutate server response with model prediction.
    This function implements the core logic from main.py
    
    Args:
        frames: List of uploaded files
        single_person_attempt: If True, consolidates all images as one person attempt
    """
    global res, latest_batch_results
    
    if not attendance_system:
        print("❌ Attendance system not initialized")
        res = PredictionResponse(user_id="error", confidence_score=0.0)
        return
    
    if not frames:
        res = PredictionResponse(user_id="no_input", confidence_score=0.0)
        return
    
    start_time = datetime.now()
    all_results = []
    processed_images = 0
    
    try:
        # Process each uploaded image
        for upload_file in frames:
            try:
                # Save uploaded file temporarily
                temp_path = save_upload_file(upload_file)
                
                try:
                    # Process the image using attendance system
                    results = attendance_system.process_single_image(temp_path)
                    all_results.extend(results)
                    processed_images += 1
                    
                finally:
                    # Clean up temporary file
                    cleanup_temp_file(temp_path)
                    
            except Exception as e:
                print(f"Error processing uploaded file: {e}")
                continue
        
        # Consolidate results if single person attempt
        if single_person_attempt and len(all_results) > 1:
            # Use the consolidation logic from attendance system
            consolidated = attendance_system.consolidate_single_person_results(all_results, "multipart_upload")
            all_results = consolidated
            print(f"🔄 Consolidated {processed_images} images into single person attempt")
        
        # Update global response with best result
        if all_results:
            # Find the result with highest confidence
            best_result = max(all_results, key=lambda x: x.get('confidence', 0.0))
            
            res = PredictionResponse(
                user_id=best_result.get('person_id', 'unknown'),
                confidence_score=best_result.get('confidence', 0.0)
            )
            
            # Create batch response
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            batch_results = []
            for result in all_results:
                batch_results.append(PredictionResponse(
                    user_id=result.get('person_id', 'unknown'),
                    confidence_score=result.get('confidence', 0.0)
                ))
            
            latest_batch_results = BatchPredictionResponse(
                total_images=processed_images,
                total_faces=len(all_results),
                results=batch_results,
                processing_time_ms=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
            print(f"✅ Processed {processed_images} images, found {len(all_results)} faces")
            print(f"🎯 Best match: {res.user_id} (confidence: {res.confidence_score:.3f})")
            
            # Additional info for consolidated results
            if single_person_attempt and len(batch_results) == 1 and all_results:
                result = all_results[0]
                if 'voting_details' in result:
                    print(f"🗳️  Voting details: {result['voting_details']['all_votes']}")
        else:
            res = PredictionResponse(user_id="no_faces", confidence_score=0.0)
            print("⚠️ No faces detected in uploaded images")
            
    except Exception as e:
        print(f"❌ Error in user_id_prediction: {e}")
        res = PredictionResponse(user_id="error", confidence_score=0.0)

# FastAPI endpoints (only if FastAPI is available)
if FASTAPI_AVAILABLE:
    @app.get("/", response_model=dict)
    async def root():
        """Root endpoint - system status"""
        if not attendance_system:
            return {
                "status": "error",
                "message": "Attendance system not initialized",
                "system_ready": False
            }
        
        return {
            "status": "success",
            "message": "Face Recognition Attendance API is running",
            "system_ready": True,
            "enrolled_persons": len(attendance_system.enrollment_data),
            "similarity_threshold": attendance_system.similarity_threshold
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system_ready": attendance_system is not None
        }

    @app.get("/attendance/recognise", response_model=PredictionResponse)
    async def get_attendance_recognise():
        """Return attendance recognition data - matches main.py structure"""
        return res

    @app.post("/attendance/recognise")
    async def post_attendance_model(frames: List[UploadFile] = File(...), multipart_mode: bool = True):
        """
        Get the frames for recognition - matches main.py structure
        
        Args:
            frames: List of uploaded image files
            multipart_mode: If True, treats all images as one person attempt (default: True)
        """
        if not attendance_system:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Attendance system not initialized"
            )
        
        print(f"📸 Received {len(frames)} images, multipart_mode: {multipart_mode}")
        
        # Call the prediction function (matches main.py)
        user_id_prediction(frames, single_person_attempt=multipart_mode)
        
        return {"status": "success", "images_processed": len(frames), "multipart_mode": multipart_mode}
    
    @app.post("/attendance/recognise-single")
    async def post_attendance_single_mode(frames: List[UploadFile] = File(...)):
        """
        Process each image individually (legacy mode)
        """
        if not attendance_system:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Attendance system not initialized"
            )
        
        # Call prediction with single person mode disabled
        user_id_prediction(frames, single_person_attempt=False)
        
        return {"status": "success", "images_processed": len(frames), "multipart_mode": False}

    @app.get("/attendance/batch-results", response_model=BatchPredictionResponse)
    async def get_batch_results():
        """Get detailed batch processing results"""
        if latest_batch_results is None:
            return BatchPredictionResponse(
                total_images=0,
                total_faces=0,
                results=[],
                processing_time_ms=0.0,
                timestamp=datetime.now().isoformat()
            )
        
        return latest_batch_results

    @app.get("/attendance/voting-details")
    async def get_voting_details():
        """Get detailed voting information from multipart processing"""
        if latest_batch_results is None:
            return {"message": "No recent batch results available"}
        
        # Try to get voting details if available
        voting_info = {
            "total_images": latest_batch_results.total_images,
            "total_faces": latest_batch_results.total_faces,
            "processing_time_ms": latest_batch_results.processing_time_ms,
            "final_result": {
                "user_id": res.user_id,
                "confidence_score": res.confidence_score
            },
            "timestamp": latest_batch_results.timestamp
        }
        
        return voting_info

    @app.get("/attendance/enrolled-persons")
    async def get_enrolled_persons():
        """Get list of all enrolled persons"""
        if not attendance_system:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Attendance system not initialized"
            )
        
        return {
            "total_persons": len(attendance_system.enrollment_data),
            "persons": list(attendance_system.enrollment_data.keys()),
            "database_path": attendance_system.database_path,
            "similarity_threshold": attendance_system.similarity_threshold
        }

    @app.post("/attendance/config")
    async def update_config(config: dict):
        """Update system configuration"""
        if not attendance_system:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Attendance system not initialized"
            )
        
        updated_fields = {}
        
        if "similarity_threshold" in config:
            threshold = config["similarity_threshold"]
            if 0.0 <= threshold <= 1.0:
                attendance_system.similarity_threshold = threshold
                updated_fields["similarity_threshold"] = threshold
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Similarity threshold must be between 0.0 and 1.0"
                )
        
        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "updated_fields": updated_fields
        }

    # Error handlers
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        """Handle general exceptions"""
        print(f"Unhandled error: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "Internal server error",
                "detail": str(exc),
                "timestamp": datetime.now().isoformat()
            }
        )

# Main execution
def main():
    """Main function for running the microservice"""
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI not available. Please install dependencies:")
        print("   pip install fastapi uvicorn python-multipart")
        return
    
    # Initialize system
    if not initialize_system():
        print("❌ System initialization failed. Exiting.")
        return
    
    print("🚀 Starting Face Recognition Attendance Microservice...")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    print("🔍 Interactive API at: http://localhost:8000/redoc")
    print("\n🎯 API Endpoints (compatible with main.py):")
    print("   GET  /attendance/recognise  - Get latest recognition result")
    print("   POST /attendance/recognise  - Upload images for recognition")
    print("   GET  /attendance/batch-results - Get detailed batch results")
    print("   GET  /attendance/enrolled-persons - List enrolled persons")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

if __name__ == "__main__":
    main()