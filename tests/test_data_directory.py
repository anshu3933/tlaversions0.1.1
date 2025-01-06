import unittest
import os
import shutil
import tempfile
from langchain.schema import Document
from loaders import load_documents
from main import initialize_system

class TestDataDirectory(unittest.TestCase):
    def setUp(self):
        """Set up test environment with a temporary data directory."""
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, "data")
        os.makedirs(self.data_dir)
        
        # Create test documents
        self.create_test_files()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
        
    def create_test_files(self):
        """Create test files in the data directory."""
        # Create a test PDF
        from reportlab.pdfgen import canvas
        pdf_path = os.path.join(self.data_dir, "test.pdf")
        c = canvas.Canvas(pdf_path)
        c.drawString(72, 720, "Test PDF Content")
        c.save()
        
        # Create a test DOCX
        from docx import Document
        docx_path = os.path.join(self.data_dir, "test.docx")
        doc = Document()
        doc.add_paragraph("Test DOCX Content")
        doc.save(docx_path)
        
    def test_data_directory_loading(self):
        """Test loading documents from data directory."""
        # Get all files from data directory
        data_files = [os.path.join(self.data_dir, f) 
                     for f in os.listdir(self.data_dir) 
                     if f.endswith(('.pdf', '.docx', '.doc', '.txt'))]
        
        # Load documents
        documents = load_documents(data_files)
        
        # Verify documents were loaded
        self.assertTrue(len(documents) > 0)
        self.assertTrue(any(doc.metadata["type"] == "pdf" for doc in documents))
        self.assertTrue(any(doc.metadata["type"] == "docx" for doc in documents))
        
    def test_system_initialization(self):
        """Test system initialization with data directory."""
        # Initialize system
        chain = initialize_system(data_dir=self.data_dir, use_dspy=False)
        
        # Verify chain was created
        self.assertIsNotNone(chain)
        self.assertTrue("chain" in chain)
        
    def test_empty_data_directory(self):
        """Test handling of empty data directory."""
        # Create empty data directory
        empty_dir = os.path.join(self.test_dir, "empty_data")
        os.makedirs(empty_dir)
        
        # Get files (should be empty)
        data_files = [os.path.join(empty_dir, f) 
                     for f in os.listdir(empty_dir) 
                     if f.endswith(('.pdf', '.docx', '.doc', '.txt'))]
        
        # Load documents
        documents = load_documents(data_files)
        
        # Verify no documents were loaded
        self.assertEqual(len(documents), 0)

if __name__ == '__main__':
    unittest.main() 