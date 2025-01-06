import streamlit as st
from chains import build_rag_chain
from embeddings import build_faiss_index, load_faiss_index
from loaders import load_documents
from dspy_pipeline import IEPPipeline, LessonPlanPipeline, ProcessingConfig, process_iep_to_lesson_plans
import os
import tempfile
import shutil
import time
import json
from datetime import datetime
from langchain.schema import Document
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
import zipfile
from utils.logging_config import logger
from typing import List, Optional
from main import initialize_system

# Page Configuration
st.set_page_config(
    page_title="Educational Assistant",
    page_icon=":books:",
    layout="wide"
)

# Initialize directories
DATA_DIR = "data"
INDEX_DIR = "models/faiss_index"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Initialize session state
if "chain" not in st.session_state:
    st.session_state["chain"] = initialize_system(
        data_dir=DATA_DIR,
        use_dspy=False
    )
if "documents_processed" not in st.session_state:
    st.session_state["documents_processed"] = True  # Always true since we're using data directory
if "messages" not in st.session_state:
    st.session_state.messages = []
if "iep_results" not in st.session_state:
    st.session_state["iep_results"] = []
if "documents" not in st.session_state:
    # Load documents from data directory
    data_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(('.pdf', '.docx', '.doc', '.txt'))]
    st.session_state["documents"] = load_documents(data_files)
if "lesson_plans" not in st.session_state:
    st.session_state["lesson_plans"] = []

def process_uploaded_file(uploaded_file):
    """Process an uploaded file with DSPy pipeline."""
    try:
        logger.debug(f"=== Processing uploaded file: {uploaded_file.name} ===")
        logger.debug(f"File size: {uploaded_file.size} bytes")
        
        use_only_uploaded = st.checkbox(
            "Use only uploaded document content",
            help="When checked, processing will only use the content from your uploaded document",
            key=f"upload_checkbox_{uploaded_file.name}"  # Add unique key
        )
        
        logger.debug(f"Use only uploaded content: {use_only_uploaded}")
        config = ProcessingConfig(use_only_uploaded=use_only_uploaded)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            logger.debug(f"Saving to temp path: {file_path}")
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                logger.debug("File written to temp directory")
            
            logger.info(f"Processing file: {uploaded_file.name}")
            documents = load_documents([file_path])
            
            if not documents:
                st.error("No text could be extracted from the uploaded file.")
                return False
            
            documents = process_iep_to_lesson_plans(documents, config=config)
            st.session_state["documents"] = documents
            
            if not use_only_uploaded:
                vectorstore = build_faiss_index(documents, persist_directory=INDEX_DIR)
                if vectorstore:
                    st.session_state["chain"] = build_rag_chain(vectorstore)
            
            return True
            
    except Exception as e:
        logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        st.error(f"Error processing file: {str(e)}")
        return False

def create_lesson_plan_pdf(plan_data):
    """Create a formatted PDF from lesson plan data."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph(f"Lesson Plan - {plan_data['timeframe'].title()}", title_style))
    story.append(Spacer(1, 12))

    # Sections
    for section, content in plan_data.items():
        if section not in ['timeframe', 'timestamp', 'source_iep', 'quality_score']:
            # Section header
            story.append(Paragraph(section.replace('_', ' ').title(), styles['Heading2']))
            story.append(Spacer(1, 6))
            
            # Section content
            if isinstance(content, list):
                for item in content:
                    story.append(Paragraph(f"• {item}", styles['Normal']))
            else:
                story.append(Paragraph(str(content), styles['Normal']))
            story.append(Spacer(1, 12))

    # Build PDF
    doc.build(story)
    return buffer

def serialize_iep_data(iep_data):
    """Convert IEP data to JSON-serializable format."""
    return {
        "source": str(iep_data.get("source", "Unknown")),
        "timestamp": iep_data.get("timestamp", datetime.now().isoformat()),
        "content": str(iep_data.get("content", "")),
        "metadata": {
            k: str(v) for k, v in iep_data.get("metadata", {}).items()
        }
    }

def format_iep_content(content: str, max_length: int = 500) -> dict:
    """Format IEP content into sections and truncate if needed."""
    sections = {
        "Present Levels": "",
        "Goals & Objectives": "",
        "Accommodations": "",
        "Assessment Data": ""
    }
    
    content_parts = content.split("\n")
    current_section = "Present Levels"
    
    for part in content_parts:
        if "GOAL" in part.upper() or "OBJECTIVE" in part.upper():
            current_section = "Goals & Objectives"
        elif "ACCOMMODATION" in part.upper() or "MODIFICATION" in part.upper():
            current_section = "Accommodations"
        elif "ASSESSMENT" in part.upper() or "EVALUATION" in part.upper():
            current_section = "Assessment Data"
        sections[current_section] += part + "\n"
    
    # Truncate long sections
    truncated = False
    for section in sections:
        if len(sections[section]) > max_length:
            sections[section] = sections[section][:max_length] + "..."
            truncated = True
    
    return {"sections": sections, "truncated": truncated}

st.title("Educational Assistant with GPT-4o Mini")

# Sidebar for API Key and File Upload
with st.sidebar:
    st.title("Document Upload")
    # Set hardcoded API key
    os.environ["OPENAI_API_KEY"] = "sk-BgZv_xfB_2WV1aSultKnv1OWyhUT53GmApiOJvDEI-T3BlbkFJZP08EtkzS-qgqdgr9VskIH6wJiN4kDOpc5f_lusYUA"  # Replace with your actual API key

    #use_dspy = st.checkbox("Use DSPy Processing", value=False)

    uploaded_files = st.file_uploader(
        "Upload educational documents",
        type=["txt", "docx", "pdf", "md"],
        accept_multiple_files=True
    )

    if uploaded_files and not st.session_state["documents_processed"]:
        processing_success = True  # Track overall success
        with st.spinner("Processing documents..."):
            st.write("### Processing Files")
            for file in uploaded_files:
                status_container = st.empty()
                status_container.info(f"Processing {file.name}...")
                
                try:
                    # Save uploaded file to temp directory
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                        tmp_file.write(file.getbuffer())
                        file_path = tmp_file.name
                    
                    # Load and process document
                    documents = load_documents([file_path])
                    if not documents:
                        status_container.error(f"Could not extract text from {file.name}")
                        processing_success = False
                        continue
                        
                    # Process with IEP pipeline
                    try:
                        processed_docs = process_iep_to_lesson_plans(documents)
                        if processed_docs and len(processed_docs) > 0:
                            st.session_state["documents"].extend(processed_docs)
                            status_container.success(f"Successfully processed {file.name}")
                        else:
                            status_container.warning(f"No lesson plans generated for {file.name}")
                            processing_success = False
                    except IndexError:
                        status_container.error(f"Error processing content from {file.name}. File may be empty or corrupted.")
                        processing_success = False
                    except Exception as e:
                        status_container.error(f"Error processing {file.name}: {str(e)}")
                        processing_success = False
                        
                    # Clean up temp file
                    os.unlink(file_path)
                    
                except Exception as e:
                    status_container.error(f"Error handling {file.name}: {str(e)}")
                    processing_success = False
            
            if processing_success:
                st.session_state["documents_processed"] = True
                st.success("All documents processed successfully!")
            else:
                st.error("Error processing some documents.")

    if st.session_state["documents_processed"]:
        if st.button("Clear Documents"):
            st.session_state["documents_processed"] = False
            st.session_state["chain"] = None
            st.session_state["documents"] = []  # Clear documents
            st.session_state["iep_results"] = []  # Clear IEP results
            if os.path.exists(INDEX_DIR):
                shutil.rmtree(INDEX_DIR)
            st.rerun()

    #st.title("System Status")
#    if st.button("Check System Health"):
 #       status = check_system_health()
        
  #      st.write("### System Components Status")
   #     for component, is_healthy in status.items():
    #        if is_healthy:
    #            st.success(f"✅ {component}: OK")
    #        else:
    #            st.error(f"❌ {component}: Failed")
                
        # Show detailed status
        #with st.expander("System Details"):
         #   st.write("- Directories:", "Created" if status["directories"] else "Failed")
          #  st.write("- Document Loading:", "Working" if status["document_loading"] else "Failed")
           # st.write("- Vector Store:", "Operational" if status["vectorstore"] else "Failed")
            #st.write("- RAG Chain:", "Functional" if status["chain"] else "Failed")
            st.write("- DSPy Integration:", "Available" if status["dspy"] else "Not Available")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Document Q&A", "Chat", "IEP Generation", "Lesson Plans"])

# Document Q&A Tab
with tab1:
    st.header("Document Q&A")
    st.info("Using documents from the data directory for question answering.")
    
    query = st.text_area(
        "Ask a question about the documents:",
        placeholder="Example: Can you summarize the main points?"
    )

    if query:
        if st.session_state["chain"]:
            with st.spinner("Generating response..."):
                try:
                    start_time = time.time()
                    
                    # Get response
                    chain_response = st.session_state["chain"]["chain"].invoke({"query": query})
                    end_time = time.time()
                    
                    # Extract answer and sources
                    answer = chain_response.get('result', '')
                    sources = chain_response.get('source_documents', [])
                    
                    # Display response
                    st.write("### Response")
                    st.write(answer)
                    
                    # Display sources
                    with st.expander("View Retrieved Context"):
                        if sources:
                            for i, doc in enumerate(sources, 1):
                                st.write(f"Source {i}:")
                                st.write(doc.page_content)
                                st.write("---")
                        else:
                            st.write("No source documents were retrieved.")
                            # Debug info
                            st.write("Debug Info:")
                            st.write(f"Chain type: {type(st.session_state['chain'])}")
                            st.write(f"Response type: {type(chain_response)}")
                            st.write(f"Response keys: {chain_response.keys() if isinstance(chain_response, dict) else 'Not a dict'}")
                    
                    st.write(f"Response time: {end_time - start_time:.2f} seconds")
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        else:
            st.warning("Please upload documents first!")

# Chat Interface Tab
with tab2:
    st.header("Chat Interface")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        if not st.session_state.get("documents_processed"):
            st.error("Please upload documents first!")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Generate response using the chain
                        response = st.session_state["chain"]["chain"].invoke({"query": prompt})
                        if response and "result" in response:
                            st.session_state.messages.append({"role": "assistant", "content": response["result"]})
                            
                            # Show sources in expander
                            with st.expander("View Sources"):
                                sources = response.get('source_documents', [])
                                if sources:
                                    for i, doc in enumerate(sources, 1):
                                        st.write(f"Source {i}:")
                                        st.write(doc.page_content)
                                        st.write("---")
                                else:
                                    st.write("No source documents were retrieved.")
                        else:
                            st.error("Error generating response: No valid result returned")
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# IEP Generation Tab
with tab3:
    st.header("IEP Generation")
    
    if st.session_state["documents_processed"]:
        # Add debug information
        st.write(f"Number of documents loaded: {len(st.session_state.get('documents', []))}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("IEP Generation Controls")
            generate_button = st.button(
                "Generate IEPs",
                help="Click to generate IEPs from uploaded documents"
            )
            
            if generate_button:
                try:
                    with st.spinner("Generating IEPs... This may take a few minutes."):
                        # Get the pipeline
                        pipeline = IEPPipeline()
                        
                        # Process documents and store results
                        iep_results = []
                        for doc in st.session_state.get("documents", []):
                            # Add debug print
                            st.write(f"Processing document: {doc.metadata.get('source', 'Unknown')}")
                            
                            result = pipeline.process_documents([doc])
                            if result:
                                iep_data = {
                                    "source": doc.metadata.get("source", "Unknown"),
                                    "timestamp": datetime.now().isoformat(),
                                    "content": str(result[0].page_content),
                                    "metadata": {k: str(v) for k, v in result[0].metadata.items()}
                                }
                                iep_results.append(iep_data)
                                # Add debug print
                                st.write(f"Successfully processed: {iep_data['source']}")
                        
                        st.session_state["iep_results"] = iep_results
                        st.success(f"Successfully generated {len(iep_results)} IEPs!")
                
                except Exception as e:
                    st.error(f"Error generating IEPs: {str(e)}")
                    st.error("Full error:", exc_info=True)
        
        with col2:
            if st.session_state["iep_results"]:
                st.subheader("Bulk Download")
                # Create a ZIP file containing all IEPs
                import io
                import zipfile
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for idx, iep in enumerate(st.session_state["iep_results"]):
                        json_data = json.dumps(serialize_iep_data(iep), indent=2)
                        zip_file.writestr(f"IEP_{idx + 1}.json", json_data)
                
                st.download_button(
                    label="Download All IEPs (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="all_ieps.zip",
                    mime="application/zip"
                )
        
        # Display IEP results
        if st.session_state["iep_results"]:
            st.subheader("Generated IEPs")
            
            for idx, iep in enumerate(st.session_state["iep_results"]):
                with st.expander(f"IEP {idx + 1} - {iep['source']}", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        content = iep['content']
                        formatted_content = format_iep_content(content)
                        
                        # Create tabs for sections
                        tabs = st.tabs(list(formatted_content["sections"].keys()))
                        for tab, (section, text) in zip(tabs, formatted_content["sections"].items()):
                            with tab:
                                if text.strip():
                                    st.markdown(text)
                                else:
                                    st.info(f"No {section} information found.")
                        
                        if formatted_content["truncated"]:
                            st.markdown("---")
                            if st.button("View Full Content", key=f"view_full_{idx}"):
                                st.markdown(content)
                    
                    with col2:
                        # Display metadata in a cleaner format
                        st.markdown("### Metadata")
                        for key, value in iep['metadata'].items():
                            st.markdown(f"**{key.title()}**: {value}")
                        
                        # Individual download button
                        json_data = json.dumps(serialize_iep_data(iep), indent=2)
                        st.download_button(
                            label=f"Download IEP {idx + 1}",
                            data=json_data,
                            file_name=f"IEP_{idx + 1}.json",
                            mime="application/json",
                            key=f"download_iep_{idx}"
                        )
        
        # Clear results button
        if st.session_state["iep_results"]:
            if st.button("Clear IEP Results"):
                st.session_state["iep_results"] = []
                st.rerun()
                
    else:
        st.warning("Please upload and process documents first!")
        st.info("Once documents are processed, you can generate IEPs here.")

# Lesson Plan Generation Tab
with tab4:
    st.header("Lesson Plan Generation")
    
    # Combined form for all lesson plan generation
    st.subheader("Lesson Plan Details")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("lesson_plan_form"):
            # Required form fields
            st.markdown("### Basic Information")
            subject = st.text_input("Subject *", placeholder="e.g., Mathematics, Reading, Science")
            grade_level = st.text_input("Grade Level *", placeholder="e.g., 3rd Grade, High School")
            
            # Timeframe selection
            timeframe = st.radio(
                "Schedule Type *",
                ["Daily", "Weekly"],
                help="Choose between a daily lesson plan or a weekly schedule"
            )
            
            duration = st.text_input(
                "Daily Duration *", 
                placeholder="e.g., 45 minutes per session"
            )
            
            if timeframe == "Weekly":
                days_per_week = st.multiselect(
                    "Select Days *",
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                )
            
            st.markdown("### Learning Details")
            specific_goals = st.text_area(
                "Specific Learning Goals *",
                placeholder="Enter specific goals for this lesson, one per line"
            )
            
            materials = st.text_area(
                "Materials Needed",
                placeholder="List required materials, one per line"
            )
            
            st.markdown("### Additional Support")
            additional_accommodations = st.text_area(
                "Additional Accommodations",
                placeholder="Enter any specific accommodations beyond those in the IEP"
            )
            
            # IEP Selection
            st.markdown("### IEP Integration")
            if st.session_state.get("iep_results"):
                selected_iep = st.selectbox(
                    "Select IEP to Integrate *",
                    options=[iep["source"] for iep in st.session_state["iep_results"]],
                    format_func=lambda x: f"IEP from {x}"
                )
            else:
                st.error("No IEPs available. Please generate an IEP first.")
                selected_iep = None
            
            st.markdown("*Required fields")
            
            generate_button = st.form_submit_button("Generate Enhanced Lesson Plan")

            if generate_button:
                if not all([subject, grade_level, duration, specific_goals, selected_iep]):
                    st.error("Please fill in all required fields.")
                else:
                    try:
                        # Get selected IEP data
                        iep_data = next(
                            iep for iep in st.session_state["iep_results"] 
                            if iep["source"] == selected_iep
                        )
                        
                        # Prepare data for lesson plan generation
                        combined_data = {
                            "iep_content": iep_data["content"],
                            "subject": subject,
                            "grade_level": grade_level,
                            "duration": duration,
                            "specific_goals": specific_goals.split('\n'),
                            "materials": materials.split('\n') if materials else [],
                            "additional_accommodations": additional_accommodations.split('\n') if additional_accommodations else [],
                            "timeframe": timeframe.lower(),
                            "days": days_per_week if timeframe == "Weekly" else ["Daily"]
                        }
                        
                        # Generate lesson plan
                        pipeline = LessonPlanPipeline()
                        lesson_plan = pipeline.generate_lesson_plan(combined_data)
                        
                        if lesson_plan:
                            # Create complete plan data structure
                            plan_data = {
                                # Input data
                                "subject": subject,
                                "grade_level": grade_level,
                                "duration": duration,
                                "timeframe": timeframe,
                                "days": days_per_week if timeframe == "Weekly" else ["Daily"],
                                "specific_goals": specific_goals.split('\n'),
                                "materials": materials.split('\n') if materials else [],
                                "additional_accommodations": additional_accommodations.split('\n') if additional_accommodations else [],
                                # Generated content
                                "schedule": lesson_plan.get('schedule', []),
                                "lesson_content": lesson_plan.get('lesson_plan', []),
                                "learning_objectives": lesson_plan.get('learning_objectives', []),
                                "assessment_criteria": lesson_plan.get('assessment_criteria', []),
                                "modifications": lesson_plan.get('modifications', []),
                                # Metadata
                                "source_iep": selected_iep,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            if "lesson_plans" not in st.session_state:
                                st.session_state["lesson_plans"] = []
                            
                            st.session_state["lesson_plans"].append(plan_data)
                            st.success("Lesson plan generated successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to generate lesson plan.")
                            
                    except Exception as e:
                        st.error(f"Error generating lesson plan: {str(e)}")

# Move PDF download outside form
if st.session_state.get("current_plan"):
    pdf_buffer = create_lesson_plan_pdf(st.session_state["current_plan"])
    st.download_button(
        label="Download PDF",
        data=pdf_buffer.getvalue(),
        file_name=f"lesson_plan_{timeframe.lower()}_{subject.lower().replace(' ', '_')}.pdf",
        mime="application/pdf"
    )

# Display generated lesson plans
if st.session_state.get("lesson_plans"):
    st.markdown("### Generated Lesson Plans")
    
    for idx, plan in enumerate(st.session_state["lesson_plans"]):
        with st.expander(f"Lesson Plan {idx + 1} - {plan.get('subject', 'Untitled')}", expanded=False):
            # Basic info
            st.markdown(f"**Subject**: {plan.get('subject', 'Not specified')}")
            st.markdown(f"**Grade Level**: {plan.get('grade_level', 'Not specified')}")
            st.markdown(f"**Duration**: {plan.get('duration', 'Not specified')}")
            
            # PDF download with safe access
            pdf_buffer = create_lesson_plan_pdf(plan)
            st.download_button(
                label=f"Download Plan {idx + 1} (PDF)",
                data=pdf_buffer.getvalue(),
                file_name=f"lesson_plan_{idx + 1}_{plan.get('subject', 'untitled').lower().replace(' ', '_')}.pdf",
                mime="application/pdf",
                key=f"download_plan_{idx}"
            )

# Footer
st.markdown("---")
st.markdown("Educational Assistant powered by GPT-4o Mini and LangChain")
