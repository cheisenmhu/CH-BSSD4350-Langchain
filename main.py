# Programmer: Chris Heise (crheise@icloud.com)
# Course: BSSD 4350 Agile Methodologies
# Instructor: Jonathan Lee
# Program: Langchain Research
# Purpose: Try and manipulate the Langchain library
# File: main.py

# Example code from
# URL: https://python.langchain.com/docs/get_started/quickstart
# URL: https://python.langchain.com/docs/modules/data_connection/retrievers/
# URL: https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
# URL: https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/
# URL: https://python.langchain.com/docs/modules/chains/
# URL: 
# Date Accessed: 30 Aug 2023


from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate


# llm = OpenAI()
llm = ChatOpenAI()

loader = PyPDFLoader("https://thediversitymovement.com/wp-content/uploads/2020/11/WW-SayThis-whitepaper_201116-F.pdf")

# Below splits the document into pages and makes sure the document is loaded/split correctly
# pages = loader.load_and_split()
# print(pages[0])

index = VectorstoreIndexCreator().from_loaders([loader])

#query = "What should I say instead of transgendered"
#print(index.query(query))
# >> 'Say "Tony is a transgender person," or "The parade included many transgender people."

#query = "What should I say instead of addicted"
#print(index.query(query))
# >> 'A fan of or excellent or delicious'
# The answer above is from a table, 'a fan of' is the replacement for 'addicted'
#   'Excellent' and 'delicious' are the replacements for 'like crack'

template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant that re-writes the user's text to be more inclusive."
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

# This works, but it's not using any outside material (like the pdf) yet
query = "I think Mary is a transgendered person"
print(llm(template.format_messages(text=query)))
# >> 'I think Mary is a transgender person.'

# TODO: figure out how to combine using the template with the index
#   Check: https://python.langchain.com/docs/use_cases/question_answering/how_to/vector_db_text_generation
