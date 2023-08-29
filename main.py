# Programmer: Chris Heise (crheise@icloud.com)
# Course: BSSD 4350 Agile Methodologies
# Instructor: Jonathan Lee
# Program: Langchain Research
# Purpose: Try and manipulate the Langchain library
# File: main.py


from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator

loader = PyPDFLoader("https://thediversitymovement.com/wp-content/uploads/2020/11/WW-SayThis-whitepaper_201116-F.pdf")
pages = loader.load_and_split()
print(pages[0])

index = VectorstoreIndexCreator().from_loaders([loader])

query = "What should I say instead of transgendered"
print(index.query(query))
# >> 'Say "Tody is a transgender person," or "The parade included many transgender people."

query = "What should I say instead of addicted"
print(index.query(query))
# >> 'A fan of or excellent or delicious'
# The answer above is from a table, 'a fan of' is the replacement for 'addicted'
#   'Excellent' and 'delicious' are the replacements for 'like crack'
