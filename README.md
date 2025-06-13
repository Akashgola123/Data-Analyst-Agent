# Data-Analyst-Agent

Of course. Here is a concise and professional `README.md` file that covers the essential information for setting up and running your Data Analyst Agent.

---

# Data Analyst Agent

This project is a multi-functional Data Analyst Agent built with Python, LangChain, and Streamlit. It allows users to upload various types of documents, ask questions about the content, perform data analysis, and generate visualizations in an interactive chat interface.

## Features

-   **Multi-Document Support**: Analyze `.csv`, `.xlsx`, `.pdf`, `.docx`, `.txt`, and image files (`.png`, `.jpg`, `.jpeg`).
-   **Intelligent Analysis**: For tabular data, the agent can perform complex operations like filtering, aggregation, and statistical analysis.
-   **Data Visualization**: Generate plots and graphs (e.g., bar charts) from data by simply asking in natural language.
-   **Document Q&A**: Extract information and answer questions from unstructured text in documents and images (using OCR).
-   **Interactive UI**: A simple and intuitive web interface powered by Streamlit.

## Core Technologies

-   **Backend**: Python, LangChain
-   **Frontend**: Streamlit
-   **LLM Integration**:
    -   Together.ai (for cloud-based models)
    -   Ollama (for local models)
-   **Data Handling**: Pandas
-   **File Parsing**: PyMuPDF, python-docx, Pillow, Pytesseract

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.9+** and Pip
2.  **Tesseract OCR**: Required for reading text from images.
    -   **On Debian/Ubuntu**: `sudo apt-get install tesseract-ocr`
    -   **On macOS**: `brew install tesseract`
    -   **On Windows**: Download the installer from the [Tesseract GitHub page](https://github.com/UB-Mannheim/tesseract/wiki).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Running the Application

This application can be configured to run with either a cloud-based LLM from Together.ai or a local LLM via Ollama.

### Option 1: Using Together.ai (Assignment Requirement)

This is the primary method for the assignment submission.

1.  **Get your API Key**:
    -   Sign up for a free account at [Together.ai](https://api.together.xyz/).
    -   Navigate to your account settings to find your API Key.

2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

3.  **Use the App:**
    -   A new tab will open in your web browser.
    -   In the sidebar, enter your **Together.ai API Key**.
    -   Upload a document and start asking questions!

### Option 2: Using a Local LLM with Ollama

This method is for local development and testing.

1.  **Install and run Ollama**:
    -   Download and install [Ollama](https://ollama.com/) for your operating system.
    -   Ensure the Ollama application is running in the background.

2.  **Pull a model**:
    -   Open your terminal and pull a model. For example, to get Llama 3:
      ```bash
      ollama pull llama3
      ```

3.  **Modify `app.py` (if necessary)**:
    -   Ensure the `get_llm()` function in `app.py` is configured to use `ChatOllama` and points to your desired local model (e.g., `model="llama3"`).
    -   Comment out or remove the API key input from the sidebar.

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

5.  **Use the App:**
    -   The application will now use your local LLM. No API key is required.
    -   Upload a document and start your analysis.
