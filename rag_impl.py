import bs4
import torch
import os
import getpass

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline

from langchain.prompts import PromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.1,
    max_new_tokens=500,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=False,
)
llm = HuggingFacePipeline(pipeline=pipe)



loader = WebBaseLoader(
    web_path=("https://www.bbc.com/portuguese/articles/cd19vexw0y1o")
)
docs = loader.load()
#print("------------------------>{qt}", len(docs[0].page_content))
#print(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, add_start_index = True)

splitts = text_splitter.split_documents(docs)

#print(splitts)

hf_embbedings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

input_test = "Um teste apenas"
result = hf_embbedings.embed_query(input_test)

vectostore = Chroma.from_documents(documents=splitts, embedding=hf_embbedings)

retriever = vectostore.as_retriever(search_type = "similarity", search_kwargs={
    "k": 6
})
template_rag = """
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    Você é um assistente virtual prestativo e está respondendo perguntas gerais.
    Use os seguintes pedaços de contexto recuperado para responder à pergunta.
    Se você não sabe a resposta, apenas diga que não sabe. Mantenha a resposta concisa.
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Pergunta: {pergunta}
    Contexto: {contexto}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
"""

prompt_rag = PromptTemplate(
    input_variables=["pergunta", "contexto"],
    template=template_rag
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain_rag = ({"contexto": retriever | format_docs, "pergunta" : RunnablePassthrough()}
             | prompt_rag
             | llm 
             | StrOutputParser)

