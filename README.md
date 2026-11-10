## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:

### DESIGN STEPS:

#### STEP 1: Import Libraries and Initialize LLM
Import the required Python libraries, load the course API key, and initialize the GPT-4o-mini language model.

#### STEP 2: Load and Process PDF
Upload the PDF using PyPDFLoader, extract its text, and split it into smaller chunks using RecursiveCharacterTextSplitter.

#### STEP 3: Create Vector Database and QA Chain
Generate embeddings using OpenAIEmbeddings, store them in a vector database, and create a conversational question-answering chain using LangChain.

#### STEP 4: Develop and Test Chatbot
Create an interactive chatbot interface using Panel. Upload the PDF, ask questions, display answers and source documents, and maintain conversation history.

### PROGRAM:
```python
import os
import openai
import sys
import panel as pn
import param

from dotenv import load_dotenv, find_dotenv

pn.extension()

# Load course API key
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# Initialize language model
llm_name = "gpt-4o-mini"

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    model_name=llm_name,
    temperature=0
)

# Import document processing libraries
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import ConversationalRetrievalChain

# Function to load and process PDF
def load_db(file):

    loader = PyPDFLoader(file)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    db = DocArrayInMemorySearch.from_documents(
        docs, embeddings
    )

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True
    )

    return qa

# Initialize chatbot
qa = None
chat_history = []

# Create widgets
file_input = pn.widgets.FileInput(accept=".pdf")

button_load = pn.widgets.Button(
    name="Load PDF",
    button_type="primary"
)

button_clearhistory = pn.widgets.Button(
    name="Clear History",
    button_type="warning"
)

question_input = pn.widgets.TextInput(
    placeholder="Enter your question here..."
)

ask_button = pn.widgets.Button(
    name="Ask",
    button_type="success"
)

status = pn.pane.Markdown(
    "**Status:** Please upload a PDF."
)

chat_panel = pn.Column()
source_panel = pn.Column()
history_panel = pn.Column()

# PDF loading function
def call_load_db(event):

    global qa, chat_history

    if file_input.value is None:
        status.object = "Please upload a PDF first."
        return

    try:
        file_path = "uploaded_document.pdf"

        with open(file_path, "wb") as f:
            f.write(file_input.value)

        qa = load_db(file_path)
        chat_history = []

        chat_panel.objects = [
            pn.pane.Markdown(
                "### PDF loaded! Ask your first question."
            )
        ]

        history_panel.objects = []
        source_panel.objects = []

        status.object = "**PDF loaded successfully!**"

    except Exception as e:
        status.object = f"**Error loading PDF:** {str(e)}"


button_load.on_click(call_load_db)

# Question-answering function
def convchain(event):

    global chat_history

    question = question_input.value.strip()

    if not question:
        return

    if qa is None:
        status.object = "Please load a PDF first."
        return

    try:
        result = qa({
            "question": question,
            "chat_history": chat_history
        })

        answer = result["answer"]

        chat_history.append((question, answer))

        chat_panel.append(
            pn.pane.Markdown(f"**You:** {question}")
        )

        chat_panel.append(
            pn.pane.Markdown(f"**ChatBot:** {answer}")
        )

        source_panel.objects = [
            pn.pane.Markdown("### Retrieved Source Documents")
        ]

        for doc in result["source_documents"]:
            page = doc.metadata.get("page", "Unknown")

            source_panel.append(
                pn.pane.Markdown(
                    f"**Page:** {page}\n\n"
                    f"{doc.page_content[:500]}\n\n---"
                )
            )

        history_panel.objects = [
            pn.pane.Markdown(
                f"**Q:** {q}\n\n**A:** {a}\n\n---"
            )
            for q, a in chat_history
        ]

        question_input.value = ""
        status.object = "**Answer generated successfully!**"

    except Exception as e:
        status.object = f"**Error generating answer:** {str(e)}"


ask_button.on_click(convchain)

# Clear history function
def clr_history(event):

    global chat_history

    chat_history = []

    chat_panel.objects = [
        pn.pane.Markdown("### Chat history cleared.")
    ]

    history_panel.objects = []
    source_panel.objects = []

    status.object = "**Conversation history cleared.**"


button_clearhistory.on_click(clr_history)

# Create dashboard tabs
tab1 = pn.Column(
    pn.pane.Markdown("## PDF Question-Answering Chatbot"),
    status,
    pn.Row(question_input, ask_button),
    pn.layout.Divider(),
    pn.Column(chat_panel, height=350, scroll=True)
)

tab2 = pn.Column(
    pn.pane.Markdown("## Retrieved Database Results"),
    pn.Column(source_panel, height=450, scroll=True)
)

tab3 = pn.Column(
    pn.pane.Markdown("## Conversation History"),
    pn.Column(history_panel, height=450, scroll=True)
)

tab4 = pn.Column(
    pn.pane.Markdown("## Configure Chatbot"),
    pn.pane.Markdown("Upload your PDF document"),
    file_input,
    button_load,
    pn.layout.Divider(),
    button_clearhistory
)

# Display dashboard
dashboard = pn.Column(
    pn.pane.Markdown("# ChatWithYourData_Bot"),
    pn.Tabs(
        ("Conversation", tab1),
        ("Database", tab2),
        ("Chat History", tab3),
        ("Configure", tab4)
    )
)

dashboard
```

### OUTPUT:

<img width="1202" height="771" alt="image" src="https://github.com/user-attachments/assets/e61d3909-7644-45d0-8276-540fbae42d23" />



<img width="1167" height="342" alt="image" src="https://github.com/user-attachments/assets/f44d15c6-372f-4e91-9e31-8ae42cb6b905" />



### RESULT:

Thus, a PDF-based Question-Answering Chatbot was successfully developed . The system successfully processes PDF documents, retrieves relevant information, generates context-based answers, maintains conversational history.
