from typing import Optional, Any, Dict
from utils.logging_config import logger

class EducationalAssistantError(Exception):
    """Base exception class for Educational Assistant errors."""
    def __init__(self, message: str, error_code: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class DocumentProcessingError(EducationalAssistantError):
    """Exception raised for errors during document processing."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DOC_PROC_ERR", details)

class PipelineError(EducationalAssistantError):
    """Exception raised for errors in the DSPy pipeline."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "PIPELINE_ERR", details)

class LessonPlanGenerationError(EducationalAssistantError):
    """Exception raised for errors during lesson plan generation."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "LESSON_GEN_ERR", details)

def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Centralized error handler that logs and formats errors."""
    error_info = {
        "type": type(error).__name__,
        "message": str(error),
        "context": context or {}
    }
    
    if isinstance(error, EducationalAssistantError):
        error_info["error_code"] = error.error_code
        error_info["details"] = error.details
        logger.error(f"{error.error_code}: {error.message}", extra=error.details)
    else:
        error_info["error_code"] = "UNKNOWN_ERR"
        logger.error(f"Unexpected error: {str(error)}", extra=context or {})
    
    return error_info 