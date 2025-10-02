#%%
from typing import Any

from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

dir = "./llava"

#reference : https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_multi_modal_RAG_LLaMA2.ipynb

#use unstructured to partition pdf into images, text and table elements
raw_pdf_elements = partition_pdf(
    filename=dir+"/llava.pdf",
    #using pdf format to find embedded image blocks
    extract_images_in_pdf=True,
    #use layout model to get bounding boxes for tables and get titles
    #titles are any sub-section of the document
    infer_table_structure=True,
    #post processing to aggregate text once we have title 
    chunking_strategy="by_title",
    #chunking params to aggregate text blocks
    #hard max on chunks
    max_characters=4000,
    #attempt to create a new chunk 3800 chars
    new_after_n_chars=3800,
    #attempt to keep chunks > 2000 chars
    combine_text_under_n_chars=2000,
    image_output_dir_path=dir,
)

#create dictionary to store counts of each type
category_counts = {}
for element in raw_pdf_elements:
    category = str(type(element))
    if category in category_counts:
        category_counts[category] += 1
    else: 
        category_counts[category] = 1

#unique categories will have unique elements
#table chunk if table > max chars set 
unique_categories = set(category_counts.keys())

# %%
class Element(BaseModel):
    type: str
    text: Any


# Categorize by type
categorized_elements = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        categorized_elements.append(Element(type="table", text=str(element)))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        categorized_elements.append(Element(type="text", text=str(element)))

# Tables
table_elements = [e for e in categorized_elements if e.type == "table"]
print(len(table_elements))

# Text
text_elements = [e for e in categorized_elements if e.type == "text"]
print(len(text_elements))

#%%
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# %%
# Prompt
prompt_text = """You are an assistant tasked with summarizing tables and text. \
Give a concise summary of the table or text. Table or text chunk: {element} """
prompt = ChatPromptTemplate.from_template(prompt_text)

# Summary chain
model = ChatOllama(model="llama3.1:latest")
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
# %%
# Apply to text
texts = [i.text for i in text_elements if i.text != ""]
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
# %%
# Apply to tables
tables = [i.text for i in table_elements]
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

# %%
import os
import re

import base64
from io import BytesIO

from IPython.display import HTML, display
from PIL import Image

def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings
    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def plt_img_base64(img_base64):
    """
    Display base64 encoded string as image
    :param img_base64: Base64 string
    """
    #create an HTML img tag w/ base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}"/>'
    #display the image by rendering HTML
    display(HTML(image_html))

#%%
from langchain_ollama import OllamaLLM

img_dir = os.fsencode("./figures")
prompt_text = """Describe the image in detail. Be specific about graphs, such as bar plots. If there is text, output the text in the exact format as what is represented in the picture."""
img_summaries = {}

img_model = OllamaLLM(model='llava:latest')
for file in os.listdir(img_dir):
    filename = os.fsdecode(file)
    base = re.split(r"\.", filename)[0]
    pil_image = Image.open(os.fsdecode(img_dir) + '/' + filename)
    image_b64 = convert_to_base64(pil_image)
    llm_with_image_context = img_model.bind(images=[image_b64])
    summary = llm_with_image_context.invoke(prompt_text)
    img_summaries[base] = summary

# %%
import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.documents import Document

# create vector store to index chunks 
vectorstore = Chroma(
    collection_name="summaries", embedding_function=GPT4AllEmbeddings()
)

#storage layer for parent documents
store = InMemoryStore() 
id_key = "doc_id"

#retriever object
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key
)

#add texts + summaries to vectorstore 
doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [
    Document(page_content=s, metadata={id_key: doc_ids[i]}) 
    for i, s in enumerate(text_summaries)
]
retriever.vectorstore.add_documents(summary_texts)      #llm summaries
retriever.docstore.mset(list(zip(doc_ids, texts)))      #original texts 

#add tables 
table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [
    Document(page_content=s, metadata={id_key: table_ids[i]})
    for i, s in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, tables)))
