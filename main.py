# Programmer: Chris Heise (crheise@icloud.com)
# Course: BSSD 4350 Agile Methodologies
# Instructor: Jonathan Lee
# Program: Langchain Research
# Purpose: Use the Langchain library to create grounding functionality.
# File: main.py

# Example code for QA using a Retriever
# URL: https://python.langchain.com/docs/use_cases/question_answering/how_to/vector_db_qa
# Date Accessed: 1 Sept 2023

# Get the API key from the environment
from environs import Env
env = Env()
env.read_env()

from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate


# Load and process the source document (single PDF for now)
# - There are other document loaders available (e.g. HTML, text, etc.)
loader = PyPDFLoader("https://thediversitymovement.com/wp-content/uploads/2020/11/WW-SayThis-whitepaper_201116-F.pdf")
documents = loader.load_and_split()     # Returns a list of documents

# Create vector store from the source documents
embeddings = OpenAIEmbeddings()        # This uses the OpenAI API Key from the environment
docsearch = Chroma.from_documents(documents, embeddings)    # Creates a vector store from the documents

# Create a custom prompt to give instruction
# - We can customize our prompt to get the best formats for our use case
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say you don't know. Don't try to make the answer up.
{context}

Question: {question}
Answer:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

# Create the QA retriever chain
# - We can use other llms, different chain types/retrievers, and whatever our custom prompt is
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs, return_source_documents=True)   # Also uses the API key

# Ask a question, show the answer, and show the sources
text = "I think Mary is transgenered."
query = f"How should the following be rewritten to be more inclusive? {text}"

result = qa({"query": query})

print(result["result"])
# >> "I think Mary is transgender." (still isn't perfect, because you shouldn't gossip about someone's gender identity. But it's progress.)

print(result["source_documents"])
# >> Returns the document that the answer was found on as well as its metadata
