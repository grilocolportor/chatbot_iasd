# print("----->teste")

from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import  HuggingFaceEndpoint
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


load_dotenv()

#Exemplo hugging face

llm = HuggingFaceEndpoint( 
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.1, 
    return_full_text=False, 
    max_new_tokens=512, 
    # "stop": ["<|eot_id|>"], 
)

system_prompt = "Você é um assistente prestatico e esta respondendo perguntas gerais."
user_prompt = "{input}"

token_s, token_e = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

prompt = ChatPromptTemplate.from_messages([
    ("system", token_s + system_prompt),
    ("user", user_prompt + token_e)
])

chain = prompt | llm

input = "Explique para mim em até 1 parágrafo o conceito de redes nurais, de forma clara e objetiva"

res = chain.invoke({"input": input})
print(res)

print("------------")


#Exemplo com ollama

llm = ChatOllama(
    model="phi3",
    temperature=0.1
)

chain3 = prompt | llm

res = chain3.invoke({"input": input})
print(res.content)


