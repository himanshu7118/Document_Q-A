import app as st
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
import textwrap

# Function to load document and create pipeline
@st.cache(allow_output_mutation=True)
def load_document_pipeline(uploaded_file):
    # Load PDF document
    loader = UnstructuredFileLoader(uploaded_file)
    documents = loader.load()

    # Split document into chunks
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})

    # Create vector store
    vectorstore = FAISS.from_documents(text_chunks, embeddings)

    # Load tokenizer and model for LLM
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=True)

    # Initialize LLM pipeline
    pipe = HuggingFacePipeline(model=model, tokenizer=tokenizer, device="cuda")

    # Initialize RetrievalQA chain
    chain = RetrievalQA.from_chain_type(llm=pipe, chain_type="stuff", return_source_documents=True, retriever=vectorstore.as_retriever())

    return chain

# Streamlit UI
def main():
    st.title("Document QA Chatbot")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload PDF document", type=['pdf'])

    if uploaded_file is not None:
        st.write("PDF document uploaded successfully!")
        chain = load_document_pipeline(uploaded_file)

        # Accept query from user
        query = st.text_input("Enter your question:")

        if st.button("Get Answer"):
            st.write("Processing...")

            # Get answer
            result = chain({"query": query}, return_only_outputs=True)
            wrapped_text = textwrap.fill(result['result'], width=500)

            st.write("Answer:")
            st.write(wrapped_text)

if __name__ == '__main__':
    main()
