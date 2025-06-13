import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import tempfile

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import ChatOllama

# Document/Image loading imports
import docx
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Multi-Document Analyst Agent",
    page_icon="🤖",
    layout="wide"
)

# --- BACKEND LOGIC (Agent and Document Processing) ---

# Use Streamlit's cache to avoid re-initializing the LLM on every interaction
@st.cache_resource
def get_llm():
    """Initializes and returns the LLM instance."""
    # return ChatOpenAI(
    #     model_name="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    #     openai_api_base="https://api.together.xyz/v1/",
    #     openai_api_key=together_api_key,
    #     temperature=0.0,
    #     max_tokens=2048,
    # )
    return ChatOllama(
        model = "llama3:latest",
        temperature=0
        )

@st.cache_data(show_spinner="Loading and processing document...")
def load_document(uploaded_file):
    """
    Loads a user-uploaded file and returns its content in a standardized format.
    """
    # Use a temporary file to handle the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension in ['.csv', '.xlsx']:
            df = pd.read_csv(file_path) if file_extension == '.csv' else pd.read_excel(file_path)
            df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
            return "dataframe", df
        elif file_extension == '.pdf':
            with fitz.open(file_path) as doc:
                text = "".join(page.get_text() for page in doc)
            return "text", text
        elif file_extension == '.docx':
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return "text", text
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return "text", text
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            try:
                image = Image.open(file_path)
                text = pytesseract.image_to_string(image)
                if not text.strip():
                     return "text", "OCR could not detect any text in the image."
                return "text", f"This is an image file. OCR has extracted the following text:\n---\n{text}\n---"
            except pytesseract.TesseractNotFoundError:
                st.error("Tesseract OCR not found. Please install it on the system running this app.")
                return "error", "Tesseract OCR is not installed. Cannot process image files."
        else:
            return "error", f"Unsupported file type: {file_extension}"
    except Exception as e:
        return "error", f"An error occurred while loading the file: {e}"
    finally:
        os.remove(file_path) # Clean up temp file

# In app.py

def create_dataframe_analyst_agent(llm, df: pd.DataFrame, temp_dir):
    """Creates a LangChain agent for Pandas DataFrame analysis."""
    # The agent will save visualizations to a temp directory
    viz_path = os.path.join(temp_dir, "visualization.png")
    
    agent_prompt = f"""
    You are an expert data analyst. You are given a pandas DataFrame named `df`.
    When asked to create a plot or visualization, you MUST use matplotlib or seaborn and save the plot to a file named '{viz_path}'.
    You MUST NOT show the plot. Only save it. After saving, respond with the confirmation "Plot has been created and saved."
    For any other data analysis query, provide a clear, concise, and accurate answer.
    """
    return create_pandas_dataframe_agent(
        llm,
        df,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        agent_executor_kwargs={"handle_parsing_errors": True},
        prefix=agent_prompt,
        allow_dangerous_code=True  # <-- THIS IS THE REQUIRED FIX
    )
def create_document_qa_chain(llm):
    """Creates a LangChain chain for Q&A over unstructured text."""
    prompt_template = """
You are a helpful assistant. Answer the user's question based ONLY on the provided document context.
If the answer is not in the document, say "The answer is not found in the provided document."

Document Context:
---
{context}
---

User's Question: {question}

Answer:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return LLMChain(llm=llm, prompt=prompt)

# --- UI (Streamlit App) ---

st.title("🤖 Data Analyst Agent")
st.markdown("Upload a document (`.csv`, `.xlsx`, `.pdf`, `.txt`, `.docx`, image) and ask questions about it.")

# --- SIDEBAR FOR SETUP ---
with st.sidebar:
    # st.header("⚙️ Configuration")
    # api_key_input = st.text_input(
    #     "Together.ai API Key", 
    #     type="password",
    #     help="Get your key from https://api.together.xyz/",
    #     placeholder="Enter your key here"
    # )

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload your document",
        type=['csv', 'xlsx', 'pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg']
    )
    st.markdown("---")
    st.info(
        """
        **How to use:**
        1. Enter your Together.ai API Key.
        2. Upload a document.
        3. Ask questions in the chat box!
        """
    )

# --- MAIN CHAT INTERFACE ---

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "data_type" not in st.session_state:
    st.session_state.data_type = None

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"])

# Handle file upload
if uploaded_file is not None and st.session_state.uploaded_file_name != uploaded_file.name:
    st.session_state.uploaded_file_name = uploaded_file.name
    st.session_state.messages = [] # Reset chat
    st.session_state.data_type, st.session_state.processed_data = load_document(uploaded_file)
    with st.chat_message("assistant"):
        st.markdown(f"✅ Successfully loaded and processed `{uploaded_file.name}`. Ready for your questions!")
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"✅ Successfully loaded and processed `{uploaded_file.name}`. Ready for your questions!"
    })

# Main chat input
if prompt := st.chat_input("Ask a question about the document..."):
    # Check for prerequisites
    # if not api_key_input:
    #     st.error("Please enter your Together.ai API Key in the sidebar.")
    if uploaded_file is None:
        st.warning("Please upload a document first.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                # Create a temporary directory for this session to store visualizations
                with tempfile.TemporaryDirectory() as temp_dir:
                    viz_path = os.path.join(temp_dir, "visualization.png")
                    
                    # llm = get_llm(api_key_input)
                    llm = get_llm()
                    answer = None
                    image_to_display = None

                    if st.session_state.data_type == "dataframe":
                        agent = create_dataframe_analyst_agent(llm, st.session_state.processed_data, temp_dir)
                        try:
                            response = agent.invoke(prompt)
                            answer = response.get('output', 'Sorry, I had trouble processing that.')
                            if os.path.exists(viz_path):
                                image_to_display = viz_path
                        except Exception as e:
                            answer = f"An error occurred: {e}"

                    elif st.session_state.data_type == "text":
                        qa_chain = create_document_qa_chain(llm)
                        response = qa_chain.invoke({
                            "context": st.session_state.processed_data,
                            "question": prompt
                        })
                        answer = response.get('text', 'Sorry, I had trouble processing that.')
                    
                    elif st.session_state.data_type == "error":
                        answer = st.session_state.processed_data

                    # Display response
                    st.markdown(answer)
                    if image_to_display:
                        st.image(image_to_display)
                    
                    # Add assistant response to chat history
                    assistant_message = {"role": "assistant", "content": answer}
                    if image_to_display:
                        assistant_message["image"] = image_to_display
                    st.session_state.messages.append(assistant_message)