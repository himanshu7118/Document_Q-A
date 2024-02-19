import os
import re
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

pc = Pinecone(api_key='API KEY', environment='gcp-starter')

model = SentenceTransformer('all-MiniLM-L6-v2')

index = pc.Index('documentqa')


def create_chunks(text, chunk_length):
    words = iter(text.split())
    current_chunk = next(words, '')

    for word in words:
        if len(current_chunk) + len(word) + 1 <= chunk_length:  # Add 1 for space
            current_chunk += ' ' + word
        else:
            yield current_chunk
            current_chunk = word

    yield current_chunk


dir_path = r"C:/Users/himanshu_moontechnol/Downloads/pinecone/data/"
id = 1

lst = list()

for filename in os.listdir(dir_path):
    if filename.endswith(".pdf"):
        path = os.path.join(dir_path, filename)
        print("Processing:", path)

        # Read text from PDF
        text = ""
        with open(path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()

        # Clean and preprocess text
        text = re.sub("\n\n+", "\n", text)

        chunk_length = 200

        for chunk in create_chunks(text, chunk_length):
            vector = model.encode(chunk).tolist()
            metadata = {'text': chunk}

            vectors = (str(id), vector, metadata)
            lst.append(vectors)
            id += 1

index.upsert(vectors=lst)

print("Upsert completed.")
