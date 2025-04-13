import streamlit as st
import streamlit
import tempfile
import os
from langchain_community.document_loaders import CSVLoader, WebBaseLoader, PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="DocTalk 🧠",
    page_icon="📄",
    layout="centered"
)

parser = StrOutputParser()

model = OllamaLLM(model = 'llama3.2')

prompt = PromptTemplate(
    template="""
You are a helpful assistant. Answer the following question using ONLY the data provided below.

If the answer is not in the data, say "I couldn’t find the answer in the provided content."

Question:
{question}

Data:
{data}

THE ANSWER SHOULD FOLLOW FORMAT AS: FOR THE GIVEN QUESTION {question}, the answer is <answer>.
""",
    input_variables=['question', 'data']
)

st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="font-size: 48px; color: #2c3e50; margin-bottom: 10px;">DocTalk 🤖</h1>
    <h3 style="font-weight: normal; font-size: 22px; color: #555;">Ask questions, get smart answers from your files.</h3>
    <p style="font-size: 16px; color: #666; margin-top: 20px;">
        📂 <b>Interact With Your Files Like Never Before</b><br>
        Just drop your document and ask — your AI will do the rest! ⚡
    </p>
</div>
""", 
unsafe_allow_html=True)


st.markdown("""
📂 <b>Upload a file</b>, 🔗 <b>enter a website URL</b>, or 📊 <b>drop in a CSV</b> —  Then just ask your question! Your AI assistant will handle the rest. 🧠✨
""", unsafe_allow_html=True)

# Dropdown selector
input_type = st.selectbox("Choose input type:", ["CSV", "PDF", "Website URL"])

# Handle input based on selection
if input_type == "CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        st.success("CSV file uploaded successfully!")
        user_text = st.text_input("Enter your question for the given file")
        # You could add further processing here, e.g., displaying the dataframe
        if user_text:
            loader = CSVLoader(uploaded_file)
            doc = loader.load()
            chain = prompt | model | parser

            with st.spinner('Getting your answer ... 🤔'):
                result = chain.invoke({'question': user_text, 'data':doc})
                st.success("Done!")
                st.write(result)


elif input_type == "PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        st.success("PDF file uploaded successfully!")
        user_question = st.text_input("Enter your question about the PDF")

        if user_question:
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Use the file path with PyPDFLoader
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()

            # Cleanup temp file after loading (optional)
            os.remove(tmp_file_path)

            # Combine page content into one string
            full_text = "\n".join([doc.page_content for doc in docs])
            # short_text = full_text[:3000]  # Optional truncation

            chain = prompt | model | parser
            with st.spinner("Thinking... 🤔"):
                result = chain.invoke({'question': user_question, 'data': full_text})
                st.success("Done!")
                st.write(result)


elif input_type == "Website URL":
    url = st.text_input("Enter the website URL")
    if url:
        st.success(f"URL entered: {url}")
        user_question = st.text_input("Enter your question about the website")

        if user_question:
            loader = WebBaseLoader(url)
            docs = loader.load()
            full_text = "\n".join([doc.page_content for doc in docs])

            chain = prompt | model | parser
            with st.spinner('Getting your answer ... 🤔'):
                result = chain.invoke({'question': user_question, 'data': full_text})
                st.success("Done!")
                st.write(result)
