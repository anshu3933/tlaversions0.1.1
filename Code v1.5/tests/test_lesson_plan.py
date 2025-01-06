import unittest
from pathlib import Path
import logging
from dotenv import load_dotenv
from langchain.schema import Document
from dspy_pipeline import LessonPlanPipeline, ProcessingConfig
from loaders import load_documents

load_dotenv()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestLessonPlanGeneration(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(__file__).parent / "test_files"
        self.test_file = self.test_dir / "Balanujan.pdf"
        self.test_dir.mkdir(exist_ok=True)
        
        # Test data
        self.test_data = {
            "iep_content": "Student requires visual aids and extended time",
            "subject": "Mathematics",
            "grade_level": "5th Grade",
            "duration": "45 minutes",
            "specific_goals": ["Master basic algebra concepts"],
            "materials": ["Textbook", "Worksheets"],
            "additional_accommodations": ["Visual aids"],
            "timeframe": "daily",
            "days": ["Monday", "Wednesday", "Friday"]
        }
        
    def test_lesson_plan_initialization(self):
        """Test lesson plan pipeline initialization."""
        try:
            pipeline = LessonPlanPipeline()
            self.assertIsNotNone(pipeline)
            self.assertIsNotNone(pipeline.lm)
            logger.debug("LessonPlanPipeline initialized successfully")
        except Exception as e:
            self.fail(f"Pipeline initialization failed: {e}")
            
    def test_lesson_plan_generation(self):
        """Test generating a lesson plan."""
        pipeline = LessonPlanPipeline()
        plan = pipeline.generate_lesson_plan(
            data=self.test_data,
            timeframe="daily"
        )
        
        self.assertIsNotNone(plan)
        self.assertIn('schedule', plan)
        self.assertIn('learning_objectives', plan)
        self.assertIn('assessment_criteria', plan)
        self.assertIn('modifications', plan)
        
        # Verify plan structure
        self.assertTrue(len(plan['learning_objectives']) >= 3)
        self.assertTrue(len(plan['assessment_criteria']) >= 3)
        self.assertTrue(len(plan['modifications']) >= 2)
        
    def test_lesson_plan_quality(self):
        """Test the quality evaluation of generated plans."""
        pipeline = LessonPlanPipeline()
        plan = pipeline.generate_lesson_plan(
            data=self.test_data,
            timeframe="daily"
        )
        
        quality_score = pipeline.evaluate_lesson_plan(plan)
        self.assertGreaterEqual(quality_score, 0.0)
        self.assertLessEqual(quality_score, 1.0)
        
    def test_weekly_plan_generation(self):
        """Test generating a weekly lesson plan."""
        pipeline = LessonPlanPipeline()
        self.test_data['timeframe'] = 'weekly'
        
        plan = pipeline.generate_lesson_plan(
            data=self.test_data,
            timeframe="weekly"
        )
        
        self.assertIsNotNone(plan)
        self.assertIn('schedule', plan)
        
        # Verify weekly structure
        schedule = plan['schedule']
        if isinstance(schedule, dict):
            for day in self.test_data['days']:
                self.assertIn(day, schedule)

if __name__ == '__main__':
    unittest.main(verbosity=2) 