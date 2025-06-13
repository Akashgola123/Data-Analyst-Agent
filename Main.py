#Together API KEY IN MY COMPUTER OR SHELL


# basic libary
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from getpass import getpass
import together

# langchian import
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.prompts import prompt, PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import ChatOllama

# image and pdf and other input
import docx
import pytesseract
from PIL import Image
import fitz


# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)



## checking together API KEY

if "TOGETHER_API_KEY" not in os.environ:
    os.environ["TOGETHER_API_KEY"] = getpass("Please Enter the API KEY OF TOGETHER.AI")


##Initialize the LLM

# llm = ChatOpenAI(
#     model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
#     base_url = "https://api.together.xyz/v1",
#     api_key=os.environ.get("TOGETHER_API_KEY"),
#     temperature=0
# )
llm = ChatOllama(
    model = "llama3:latest",
    temperature=0
)

print("LLM Initialized Successfully!")
print(f"Model: {llm.model}")

#Document And Pre-processing 

def load_Documnet(file_path : str):
    """
    Loads a document based on its file extension and returns its content
    in a standardized format.

    Returns:
        A tuple of (data_type, data), where:
        - data_type is 'dataframe' for tabular data or 'text' for unstructured data.
        - data is a pandas DataFrame or a string.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    print(f"Loading file: {file_path} (Type: {file_extension})")

    try:
        if file_extension in ['.csv', '.xlsx']:
            df = pd.read_csv(file_path) if file_extension == '.csv' else pd.read_excel(file_path)
            # Sanitize column names for better LLM processing
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
                print("\n--- TESSERACT OCR NOT FOUND ---")
                print("Please install Tesseract OCR on your system and ensure it's in your PATH.")
                return "text", "Error: Tesseract OCR is not installed or not in the system's PATH. Cannot process image files."
        else:
            return "error", f"Unsupported file type: {file_extension}"
    except Exception as e:
        return "error", f"An error occurred while loading the file: {e}"

# Agent and Chain 


def create_dataframe_analyst_agent(df: pd.DataFrame):
    """Creates a LangChain agent specifically for Pandas DataFrame analysis."""
    
    agent_prompt = """
    You are an expert data analyst. You are given a pandas DataFrame named `df`.
    When asked to create a plot or visualization, you MUST use matplotlib or seaborn and save the plot to a file named 'visualization.png'.
    After saving the plot, you MUST respond with the confirmation "Plot has been created and saved as visualization.png."
    For any other data analysis query, provide a clear, concise, and accurate answer based on the data.
    Think step-by-step and show your work if necessary.
    """
    return create_pandas_dataframe_agent(
        llm,
        df,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, # Set to True to see the agent's thought process
        agent_executor_kwargs={"handle_parsing_errors": True}, # Gracefully handle LLM output errors
        prefix=agent_prompt,
        allow_dangerous_code=True  # <-- THIS IS THE REQUIRED FIX
    )

def create_document_qa_chain():
    """Creates a LangChain chain for Q&A over unstructured text."""
    prompt_template = """
    You are a helpful assistant. You are given a document's content below.
    Please answer the user's question based ONLY on the provided document context.
    If the answer is not in the document, say "The answer is not found in the provided document."

    Document Context:
    ---
    {context}
    ---

    User's Question: {question}

    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return LLMChain(llm=llm, prompt=prompt)

# The Main Agent Respone
def get_agent_response(file_path: str, query: str):
    """
    Main function to orchestrate the agent's response.
    It loads the document, selects the appropriate tool (agent/chain),
    and returns the answer.
    """
    # Clear previous visualizations
    if os.path.exists("visualization.png"):
        os.remove("visualization.png")

    data_type, data = load_Documnet(file_path)

    if data_type == "error":
        return data, None

    if data_type == "dataframe":
        # Use the powerful DataFrame agent
        agent = create_dataframe_analyst_agent(data)
        try:
            response = agent.invoke(query)
            answer = response.get('output', 'Could not process the request.')
        except Exception as e:
            answer = f"An error occurred while the agent was processing: {e}"

        # Check if a visualization was created
        if os.path.exists("visualization.png"):
            return answer, "visualization.png"
        else:
            return answer, None

    elif data_type == "text":
        # Use the simpler Document Q&A chain
        qa_chain = create_document_qa_chain()
        response = qa_chain.invoke({"context": data, "question": query})
        return response.get('text', 'Could not process the request.'), None

    else:
        return "An unknown error occurred.", None


# @title 6. Demonstration and Testing

# --- Create dummy files for testing ---
print("\n--- Creating Dummy Files for Demonstration ---")

# 1. CSV file
csv_data = {
    'EmployeeID': [1, 2, 3, 4, 5, 6],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'Department': ['HR', 'Engineering', 'Engineering', 'Sales', 'Sales', 'HR'],
    'Salary': [70000, 80000, 95000, 65000, 72000, 55000],
    'YearsAtCompany': [5, 3, 7, 2, 4, 1]
}
pd.DataFrame(csv_data).to_csv('sample_data.csv', index=False)
print("Created sample_data.csv")

# 2. Text file
text_content = """
Project Titan: Q3 2023 Update

Project Titan is on track to meet its key objectives for the third quarter. The engineering team, led by Bob, successfully completed the alpha version of the core module. The final cost for this phase was $120,000, which is 5% under budget.

The sales team, consisting of David and Eve, secured two new major clients, bringing the total projected annual revenue to $1.5 million. The marketing team will launch the new campaign in early November.

Key risks include potential supply chain disruptions and a new competitor entering the market. Mitigation plans are being developed.
"""
with open('sample_doc.txt', 'w') as f:
    f.write(text_content)
print("Created sample_doc.txt")

# --- Run Tests ---

print("\n\n--- TEST 1: CSV DATA ANALYSIS ---")
file_to_test = 'sample_data.csv'
query1 = "What is the average salary for the Engineering department?"
answer1, viz1 = get_agent_response(file_to_test, query1)
print(f"\nQ: {query1}")
print(f"A: {answer1}")

print("\n\n--- TEST 2: CSV DATA VISUALIZATION ---")
query2 = "Plot a bar chart showing the average salary for each department. Save it as visualization.png"
answer2, viz2 = get_agent_response(file_to_test, query2)
print(f"\nQ: {query2}")
print(f"A: {answer2}")
if viz2:
    print(f"Visualization generated at: {viz2}")
    # Display the plot in the notebook
    img = Image.open(viz2)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

print("\n\n--- TEST 3: UNSTRUCTURED TEXT Q&A ---")
file_to_test = 'sample_doc.txt'
query3 = "Who is the lead of the engineering team and what was the final cost of the alpha version?"
answer3, viz3 = get_agent_response(file_to_test, query3)
print(f"\nQ: {query3}")
print(f"A: {answer3}")

query4 = "Summarize the key risks mentioned in the document."
answer4, viz4 = get_agent_response(file_to_test, query4)
print(f"\nQ: {query4}")
print(f"A: {answer4}")