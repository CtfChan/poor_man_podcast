from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import time
import os
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

# Load Documents
loader = PyPDFLoader('data/harvesting_alpha.pdf')
documents = loader.load()

print("==========================================")
print("Splitting documents into chunks...")
text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
documents = text_splitter.split_documents(documents)
print("Documents size: {doc_len}".format(doc_len=len(documents)))

# Define summarizer chain
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
chain = load_summarize_chain(llm, chain_type="stuff")

# 3 RPM for free tier.
print("==========================================")
print("Generating summaries of documents...")
summaries = []
for i, doc in enumerate(documents):
    if (i != 0 and i % 3 == 0):
        time.sleep(70)
    summary = chain.run([doc])
    summaries.append(summary)
    print(summary)
    
time.sleep(70)

print("==========================================")
print("Final Summary:")
summary_docs = [Document(page_content=t) for t in summaries]
final_summary = chain.run(summary_docs)
print(final_summary)