import dspy
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
import os
from datetime import datetime
import re
import json
import logging
from utils.logging_config import logger

class ProcessingConfig:
    """Configuration for document processing."""
    def __init__(self, use_only_uploaded: bool = False):
        self.use_only_uploaded = use_only_uploaded

class IEPSignature(dspy.Signature):
    """Signature for analyzing assessments and generating IEP components."""
    content = dspy.InputField(desc="Document content (assessment or IEP)")
    
    # Analysis outputs - must be grounded in input content
    academic_analysis = dspy.OutputField(desc="""
        Analysis of academic performance based ONLY on provided document:
        - Current performance levels with specific examples
        - Identified skill gaps with evidence
        - Observed error patterns with examples
        - Learning barriers noted in document
        Include specific quotes or references from the document.
    """)
    
    iep_components = dspy.OutputField(desc="""
        IEP components based ONLY on document evidence:
        - SMART learning objectives tied to identified needs
        - Specific accommodations supported by assessment data
        - Modifications based on documented challenges
        - Progress monitoring aligned with identified gaps
        Reference specific findings from the document.
    """)
    
    evidence = dspy.OutputField(desc="""
        Document evidence supporting analysis:
        - Direct quotes supporting each finding
        - Page or section references
        - Specific test scores or observations cited
        - Context for each recommendation
    """)

class IEPPipeline:
    """Processes educational documents to generate IEP components."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", config: Optional[ProcessingConfig] = None):
        """Initialize the IEP pipeline with configuration."""
        try:
            self.config = config or ProcessingConfig()
            logger.debug("Initializing OpenAI LM...")
            self.lm = dspy.OpenAI(model=model_name, max_tokens=4000)
            logger.debug("Configuring DSPy settings...")
            dspy.configure(lm=self.lm)
            logger.debug("Creating IEP extractor...")
            self.extractor = dspy.Predict(IEPSignature)
            logger.debug("IEP pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize IEP pipeline: {e}")
            logger.debug(f"Exception type: {type(e).__name__}")
            logger.debug(f"Exception details: {str(e)}")
            raise
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents and generate IEP components with evidence."""
        if not documents:
            logger.warning("No documents provided to process")
            return []
        
        logger.debug(f"Processing {len(documents)} documents")
        enhanced_docs = []
        
        for doc in documents:
            try:
                # Check for empty document or content
                if not doc:
                    logger.warning("Empty document object encountered")
                    continue
                    
                if not doc.page_content or not doc.page_content.strip():
                    logger.warning(f"Empty content in document: {doc.metadata.get('source', 'unknown')}")
                    enhanced_docs.append(doc)  # Keep original document even if empty
                    continue
                
                # Process valid document
                result = self.extractor(content=doc.page_content)
                
                # Verify result has required fields
                if (hasattr(result, 'academic_analysis') and 
                    hasattr(result, 'iep_components') and
                    hasattr(result, 'evidence')):
                    logger.debug("Successfully extracted IEP components")
                    enhanced_docs.append(Document(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            "processed_with": "dspy",
                            "has_verified_content": True,
                            "extraction_result": result
                        }
                    ))
                else:
                    logger.warning(f"Incomplete extraction result for document: {doc.metadata.get('source', 'unknown')}")
                    enhanced_docs.append(doc)
                    
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                enhanced_docs.append(doc)  # Keep original document on error
                continue
        
        if not enhanced_docs:
            logger.warning("No documents were successfully processed")
        
        return enhanced_docs
    
    def _verify_evidence(self, result: Any, original_content: str) -> bool:
        """Verify that generated content is supported by document evidence."""
        try:
            # Check if evidence quotes exist in original content
            evidence_quotes = self._extract_quotes(result.evidence)
            return all(
                quote.lower() in original_content.lower()
                for quote in evidence_quotes
                if quote.strip()
            )
        except Exception as e:
            logger.error(f"Error verifying evidence: {e}")
            return False
    
    def _extract_quotes(self, evidence_text: str) -> List[str]:
        """Extract quoted passages from evidence text."""
        quotes = re.findall(r'"([^"]*)"', evidence_text)
        if not quotes:
            # Try single quotes if no double quotes found
            quotes = re.findall(r"'([^']*)'", evidence_text)
        return quotes
    
    def _calculate_evidence_score(self, result: Any) -> float:
        """Calculate evidence score based on verification results."""
        try:
            evidence_parts = result.evidence.split('\n')
            valid_evidence = [part for part in evidence_parts if part.strip()]
            return len(valid_evidence) / max(len(result.academic_analysis.split('\n')), 1)
        except Exception as e:
            logger.error(f"Error calculating evidence score: {e}")
            return 0.0
    
    def _format_verified_content(self, result: Any) -> str:
        """Format content with evidence references."""
        return f"""
Academic Analysis:
{result.academic_analysis}

Supporting Evidence:
{result.evidence}

IEP Components:
{result.iep_components}
"""
    
    def _format_content(self,
                       original_content: str,
                       analysis: str,
                       iep_components: str,
                       recommendations: str) -> str:
        """Format processed content into structured output."""
        return f"""
Original Content:
{original_content}

Academic Analysis:
{analysis}

IEP Components:
{iep_components}

Educational Recommendations:
{recommendations}
"""
    
    def _create_metadata(self, 
                        doc: Document, 
                        result: Any) -> Dict[str, Any]:
        """Create metadata for processed document."""
        return {
            **doc.metadata,
            "processed_with": "dspy",
            "processing_timestamp": datetime.now().isoformat(),
            "has_analysis": bool(result.academic_analysis),
            "has_iep_components": bool(result.iep_components),
            "has_recommendations": bool(result.recommendations)
        }

def build_faiss_index_with_dspy(documents: List[Document], 
                               persist_directory: str,
                               model_name: str = "gpt-4o-mini") -> Optional[FAISS]:
    """Build a FAISS index with DSPy-enhanced documents.
    
    Args:
        documents (List[Document]): Original documents to process
        persist_directory (str): Directory to save the FAISS index
        model_name (str): Name of the OpenAI model to use for DSPy
        
    Returns:
        Optional[FAISS]: Enhanced FAISS vectorstore
    """
    try:
        # Initialize DSPy pipeline
        pipeline = IEPPipeline(model_name=model_name)
        
        # Process documents
        enhanced_docs = pipeline.process_documents(documents)
        
        # Combine original and enhanced documents
        all_docs = documents + enhanced_docs
        
        # Build and return index
        from embeddings import build_faiss_index
        return build_faiss_index(all_docs, persist_directory)
        
    except Exception as e:
        print(f"Error in DSPy-enhanced indexing: {e}")
        # Fall back to regular indexing
        from embeddings import build_faiss_index
        return build_faiss_index(documents, persist_directory)

class LessonPlanRM(dspy.Signature):
    """Signature for lesson plan reasoning module."""
    context = dspy.InputField()
    reasoning = dspy.OutputField()

class LessonPlanSignature(dspy.Signature):
    """Signature for generating lesson plans with reasoning."""
    
    # Input fields with detailed descriptions
    iep_content = dspy.InputField(desc="Full IEP content including student needs and accommodations")
    subject = dspy.InputField(desc="Subject area (e.g., Math, Science)")
    grade_level = dspy.InputField(desc="Student's grade level")
    duration = dspy.InputField(desc="Length of each lesson")
    specific_goals = dspy.InputField(desc="Specific learning objectives to be achieved")
    materials = dspy.InputField(desc="Required teaching materials and resources")
    additional_accommodations = dspy.InputField(desc="Additional accommodations beyond IEP requirements")
    timeframe = dspy.InputField(desc="Daily or weekly planning timeframe")
    days = dspy.InputField(desc="Days of the week for instruction")
    
    # Output fields with detailed structure requirements
    schedule = dspy.OutputField(desc="""
        Detailed daily schedule including:
        - Warm-up activities (5-10 minutes)
        - Main concept introduction with visual aids
        - Guided practice with accommodations
        - Independent work time
        - Assessment and closure
        Minimum length: 200 words
    """)
    
    lesson_plan = dspy.OutputField(desc="""
        Comprehensive lesson plan including:
        1. Detailed teaching strategies
        2. Step-by-step instructions
        3. Differentiation methods
        4. IEP accommodations integration
        5. Real-world connections
        6. Student engagement techniques
        7. Time management details
        Minimum length: 300 words
    """)
    
    learning_objectives = dspy.OutputField(desc="""
        Specific, measurable objectives including:
        - Knowledge acquisition goals
        - Skill development targets
        - Application objectives
        - Assessment criteria
        Minimum 5 detailed objectives
    """)
    
    assessment_criteria = dspy.OutputField(desc="""
        Detailed assessment criteria including:
        - Understanding checks
        - Skill demonstration requirements
        - Progress monitoring methods
        - Success indicators
        Minimum 5 specific criteria
    """)
    
    modifications = dspy.OutputField(desc="""
        Specific IEP-aligned modifications including:
        - Learning accommodations
        - Assessment modifications
        - Environmental adjustments
        - Support strategies
        Minimum 5 detailed modifications
    """)
    
    instructional_strategies = dspy.OutputField(desc="""
        Detailed teaching strategies including:
        - Visual learning techniques
        - Hands-on activities
        - Technology integration
        - Differentiation methods
        - Student engagement approaches
        Minimum 5 specific strategies
    """)

class LessonPlanPipeline:
    """Pipeline for generating adaptive lesson plans from IEPs."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.lm = dspy.OpenAI(
            model=model_name,
            max_tokens=4000
        )
        
        dspy.configure(lm=self.lm)
        
        self.reasoning_module = dspy.ChainOfThought(LessonPlanRM)
        self.generator = dspy.ChainOfThought(LessonPlanSignature)  # Only initialize once
        
        # Enhanced prompt template
        self.prompt_template = """
        As an experienced special education teacher, create a detailed and comprehensive lesson plan following these steps:
        
        1. Analyze the IEP requirements and student needs:
        - Review all accommodations and modifications needed
        - Identify specific learning style preferences
        - Note particular challenges and strengths
        - Consider past performance and progress
        
        2. Design comprehensive grade-level learning objectives:
        - Align with curriculum standards
        - Break down complex concepts into manageable parts
        - Set both short-term and long-term goals
        - Include measurable outcomes
        
        3. Create detailed accommodations and modifications:
        - Incorporate all IEP requirements
        - Design multiple forms of visual and hands-on activities
        - Plan differentiated instruction for various skill levels
        - Include technology and multi-sensory approaches
        
        4. Develop a structured yet flexible lesson plan:
        - Create detailed timeline with buffer time
        - Include varied activities for different learning styles
        - Plan transition strategies
        - Incorporate regular check-ins and assessments
        
        5. Design assessment and progress monitoring:
        - Include multiple forms of assessment
        - Plan for ongoing progress monitoring
        - Create success criteria
        - Design feedback mechanisms
        
        Ensure all components are detailed, specific, and aligned with student needs.
        Minimum length for each section: 200 words
        """
        
        print("Successfully initialized LessonPlanPipeline")
        
        # Add detailed example for better zero-shot learning
        self.example = {
            "iep_content": "Student requires extended time, visual aids, and hands-on learning",
            "subject": "Mathematics",
            "grade_level": "3rd Grade",
            "duration": "45 minutes",
            "specific_goals": "Understanding Pythagoras theorem through practical applications",
            "materials": "Cardboard triangles, measuring tape, grid paper",
            "additional_accommodations": "Connect to real-world examples, provide visual supports",
            "timeframe": "weekly",
            "days": "Monday through Friday",
            "schedule": """
                Daily Schedule Pattern:
                1. Warm-up (5 min): Review previous concepts using visual aids
                2. Introduction (10 min): Present new concept with real-world examples
                3. Guided Practice (15 min): Hands-on activities with manipulatives
                4. Independent Work (10 min): Practice with support as needed
                5. Closure (5 min): Quick assessment and preview next day
            """,
            "lesson_plan": """
                Week-long Progression:
                Monday: Introduction to right triangles using real objects
                Tuesday: Exploring square numbers with grid paper
                Wednesday: Discovering Pythagoras pattern with manipulatives
                Thursday: Applying theorem to real-world problems
                Friday: Review and creative applications
                
                Teaching Methodology:
                - Use visual aids consistently
                - Incorporate hands-on activities
                - Connect to real-world examples
                - Provide frequent checks for understanding
                - Allow extended time as needed
            """,
            "learning_objectives": [
                "Identify right triangles in real-world objects",
                "Calculate missing sides using Pythagoras theorem",
                "Apply theorem to solve practical problems",
                "Demonstrate understanding through multiple methods"
            ],
            "assessment_criteria": [
                "Accurate identification of right triangles",
                "Correct calculation of missing sides",
                "Proper use of theorem in applications",
                "Clear explanation of process"
            ],
            "modifications": [
                "Extended time for calculations",
                "Use of calculator when needed",
                "Visual step-by-step guides",
                "Reduced problem set with increased depth"
            ],
            "instructional_strategies": [
                "Multi-sensory approach to learning",
                "Regular comprehension checks",
                "Peer learning opportunities",
                "Visual and hands-on demonstrations"
            ]
        }
    
    def _format_list_to_string(self, items: List[str]) -> str:
        """Convert a list of items to a numbered string."""
        if not items:
            return ""
        return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
    
    def generate_lesson_plan(self, data: Dict[str, Any], timeframe: str = "daily") -> Optional[Dict[str, Any]]:
        """Generate a lesson plan based on IEP data."""
        try:
            if not self.lm:
                logger.error("LLM not initialized")
                return None
            
            # Convert list fields to strings for DSPy
            specific_goals = "\n".join(data["specific_goals"]) if isinstance(data["specific_goals"], list) else data["specific_goals"]
            materials = "\n".join(data["materials"]) if isinstance(data["materials"], list) else data["materials"]
            accommodations = "\n".join(data["additional_accommodations"]) if isinstance(data["additional_accommodations"], list) else data["additional_accommodations"]
            
            # Generate plan using input_data
            result = self.generator(**{
                "iep_content": data["iep_content"],
                "subject": data["subject"],
                "grade_level": data["grade_level"],
                "duration": data["duration"],
                "specific_goals": specific_goals,
                "materials": materials,
                "additional_accommodations": accommodations,
                "timeframe": timeframe,
                "days": ", ".join(data["days"]) if isinstance(data["days"], list) else data["days"]
            })
            
            return self._format_lesson_plan(result, timeframe)
            
        except Exception as e:
            logger.error(f"Error generating lesson plan: {str(e)}")
            return None
    
    def _prepare_context(self, data: Dict[str, Any], use_only_uploaded: bool = False) -> str:
        """Prepare context for lesson plan generation."""
        if use_only_uploaded:
            # Use only the uploaded document content
            return f"""
            Content: {data.get('content', data.get('iep_content', ''))}
            Source: {data.get('source', '')}
            Assessment Data: {json.dumps(data.get('assessment_data', {}))}
            """
        else:
            # Include additional context from knowledge base
            return self._enrich_context(data)
    
    def _enrich_context(self, data: Dict[str, Any]) -> str:
        """Enrich context with additional knowledge base information."""
        # Your existing context enrichment logic here
        return f"""
        Content: {data['content']}
        Source: {data['source']}
        Assessment Data: {json.dumps(data.get('assessment_data', {}))}
        Additional Context: {self._get_additional_context(data)}
        """
    
    def _format_lesson_plan(self, result: Any, timeframe: str) -> Optional[Dict[str, Any]]:
        """Format the lesson plan result into a structured dictionary."""
        try:
            plan_data = {
                "schedule": self._process_field(result, 'schedule'),
                "lesson_plan": self._process_field(result, 'lesson_plan'),
                "learning_objectives": self._process_field(result, 'learning_objectives'),
                "assessment_criteria": self._process_field(result, 'assessment_criteria'),
                "modifications": self._process_field(result, 'modifications'),
                "instructional_strategies": self._process_field(result, 'instructional_strategies'),
                "timeframe": timeframe
            }
            return plan_data
        except Exception as e:
            logger.error(f"Error formatting lesson plan: {e}")
            return None
    
    def _process_field(self, result: Any, field_name: str) -> List[str]:
        """Process a field from the result into a list of strings."""
        try:
            value = getattr(result, field_name, [])
            if isinstance(value, str):
                return [item.strip() for item in value.split('\n') if item.strip()]
            elif isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
            return []
        except Exception as e:
            logger.error(f"Error processing field {field_name}: {e}")
            return []
    
    def evaluate_lesson_plan(self, plan: Dict[str, Any]) -> float:
        """Evaluate the quality of a generated lesson plan."""
        try:
            score = 0.0
            
            # Check segmentation
            if isinstance(plan.get('schedule'), str) and len(plan.get('schedule', '').split('\n')) > 2:
                score += 0.3
                
            # Check real-world anchoring
            lesson_plan_text = str(plan.get('lesson_plan', ''))
            if 'real-world' in lesson_plan_text.lower() or 'application' in lesson_plan_text.lower():
                score += 0.3
                
            # Check IEP alignment
            if plan.get('modifications') and len(plan.get('modifications', [])) > 0:
                score += 0.4
                
            return score
            
        except Exception as e:
            print(f"Error in evaluate_lesson_plan: {str(e)}")
            return 0.0

    def _get_additional_context(self, data: Dict[str, Any]) -> str:
        """Get additional context from knowledge base."""
        try:
            # For now, return empty string as knowledge base integration 
            # will be implemented later
            return ""
        except Exception as e:
            logger.error(f"Error getting additional context: {e}")
            return ""

def process_iep_to_lesson_plans(documents: List[Document], timeframes: List[str] = ["daily"], config: Optional[ProcessingConfig] = None) -> List[Document]:
    """Process IEP documents into lesson plans.
    
    Args:
        documents: List of Document objects containing IEP content
        timeframes: List of timeframes to generate plans for (e.g. ["daily", "weekly"])
        config: Optional ProcessingConfig object
        
    Returns:
        List of Document objects containing generated lesson plans
    """
    pipeline = LessonPlanPipeline()
    iep_pipeline = IEPPipeline(config=config)
    enhanced_docs = []
    
    if not documents:
        logger.warning("No documents provided")
        return []
        
    logger.debug(f"Starting IEP to lesson plan processing for {len(documents)} documents")
    
    for doc in documents:
        try:
            # Process document and handle empty results
            processed_docs = iep_pipeline.process_documents([doc])
            logger.debug(f"IEP processing complete for {doc.metadata.get('source', 'unknown')}")
            
            # Safely get IEP result or use original
            if processed_docs and len(processed_docs) > 0:
                iep_result = processed_docs[0]
                logger.debug("Using processed IEP result")
            else:
                logger.warning("No IEP processing results, using original document")
                iep_result = doc
            
            # Generate lesson plans for each timeframe
            for timeframe in timeframes:
                try:
                    if not iep_result.page_content.strip():
                        logger.warning(f"Empty content in IEP result for {timeframe}")
                        continue
                        
                    # Prepare data for lesson plan generation
                    data = {
                        "iep_content": iep_result.page_content,
                        "subject": "Mathematics",
                        "grade_level": "5th Grade",
                        "duration": "45 minutes",
                        "specific_goals": ["Master basic algebra concepts"],
                        "materials": ["Textbook", "Worksheets"],
                        "additional_accommodations": ["Visual aids"],
                        "timeframe": timeframe,
                        "days": ["Monday", "Wednesday", "Friday"],
                        "assessment_data": {}
                    }
                    
                    # Generate lesson plan
                    lesson_plan = pipeline.generate_lesson_plan(
                        data=data,
                        timeframe=timeframe
                    )
                    
                    if lesson_plan:
                        enhanced_docs.append(Document(
                            page_content=str(lesson_plan),
                            metadata={
                                **doc.metadata,
                                "type": "lesson_plan",
                                "timeframe": timeframe,
                                "source_iep": doc.metadata.get("source")
                            }
                        ))
                        logger.debug(f"Generated {timeframe} lesson plan for {doc.metadata.get('source', 'unknown')}")
                    
                except Exception as e:
                    logger.error(f"Error generating {timeframe} lesson plan: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing document {doc.metadata.get('source', 'unknown')}: {str(e)}")
            continue
            
    logger.debug(f"Completed processing with {len(enhanced_docs)} lesson plans generated")
    return enhanced_docs

def _extract_assessment_data(doc: Document) -> Dict[str, Any]:
    """Extract assessment data from processed document."""
    empty_result = {
        "academic_analysis": "",
        "skill_gaps": "",
        "learning_objectives": ""
    }
    
    if not doc or not isinstance(doc, Document):
        logger.warning("Invalid document for assessment data extraction")
        return empty_result
        
    try:
        # Safely handle content extraction
        content = doc.page_content if (doc and hasattr(doc, 'page_content')) else ""
        if not content:
            logger.warning("Empty content in document")
            return empty_result
            
        return {
            "academic_analysis": _extract_section(content, "Academic Analysis") or "",
            "skill_gaps": _extract_section(content, "Skill Gaps") or "",
            "learning_objectives": _extract_section(content, "Learning Objectives") or ""
        }
        
    except Exception as e:
        logger.error(f"Error extracting assessment data: {e}")
        return empty_result

def _extract_section(content: str, section_name: str) -> str:
    """Extract a specific section from the enhanced content."""
    pattern = f"{section_name}:\n(.*?)(?=\n\n|$)"
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip() if match else ""

