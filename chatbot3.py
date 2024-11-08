import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document
import torch
from langchain_huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub
import faiss
import tempfile
import os
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Converse com documentos 📚", page_icon="📚")
st.title("Converse com documentos 📚")

pdf_directory = "./pdfs"

model_class = "hf_hub" # @param ["hf_hub", "openai", "ollama"]

## Provedores de modelos
def model_hf_hub(model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1):
  llm = HuggingFaceHub(
      repo_id=model,
      model_kwargs={
          "temperature": temperature,
          "return_full_text": False,
          "max_new_tokens": 512
          #"stop": ["<|eot_id|>"],
          # demais parâmetros que desejar
        }
    ) 
  return llm

def model_openai(model="gpt-4o-mini", temperature=0.1):
    llm = ChatOpenAI(
        model=model,
        temperature=temperature
        # demais parâmetros que desejar
    )
    return llm

def model_ollama(model="phi3", temperature=0.1):
    llm = ChatOllama(
        model=model,
        temperature=temperature,
    )
    return llm

 # Exemplo de limpeza
def clean_text(text):
    # Remover texto desnecessário
    unwanted_phrases = [
        "Oscar2024", 
        "CRÉDITO", 
        "GETTYIMAGES", 
        "Le", 
        "Legendadafoto", 
        "AssassinosdaLuadasFlores",
        "BlueNexusIndustries", 
        "RuaBakerHolmes", 
        "Website", 
        "Apresentação", 
        "ReceitaeFaturamento", 
        "Prêmios", 
        "Reconhecimentos", 
        "aumento de capital",
        "SegurançaInterna", 
        "Amostras", 
        "Legendas", 
        "Distribuição", 
        "Contato com a imprensa",
        "notícia", 
        "página de negócios", 
        "Tabela 1", 
        "Tabela 2", 
        "Figura 1", 
        "Figura 2", 
        "apresentação da empresa", 
        "deve ser removido",  # Adicione outras palavras-chave específicas
    ]
    
    # Remover as frases/termos indesejados do texto
    for phrase in unwanted_phrases:
        text = text.replace(phrase, "")
    
    # Remover quebras de palavras
    text = text.replace("\n", " ").replace("\r", "")
    
    # Remover múltiplos espaços
    text = " ".join(text.split())

    return text

# Configuração da função retriever
def config_retriever():
    docs = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            loader = PyPDFLoader(filepath)
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
            
            # Aplicar a limpeza de texto
            cleaned_docs = [Document(page_content=clean_text(doc.page_content), metadata=doc.metadata) for doc in loaded_docs]
            docs.extend(cleaned_docs)
            
            # Verifique o conteúdo carregado após a limpeza
            # for doc in cleaned_docs:
            #     print(doc.page_content[:100]) 

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local('vectorstore/db_faiss')
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 4})
    return retriever

def config_rag_chain(model_class, retriever):
    if model_class == "hf_hub":
        llm = model_hf_hub()
    elif model_class == "openai":
        llm = model_openai()
    elif model_class == "ollama":
        llm = model_ollama()
    
    if model_class.startswith("hf"):
        token_s = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        token_e = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        token_s = ""
        token_e = ""
        
    # token_s, token_e = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>" if model_class.startswith("hf") else "", ""
    context_q_system_prompt = "Given the following chat history and the follow-up question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    context_q_system_prompt = token_s + context_q_system_prompt
    context_q_user_prompt = "Question: {input}" + token_e
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", context_q_user_prompt),
        ]
    )
    
       # Chain para contextualização
    history_aware_retriever = create_history_aware_retriever(llm = llm,
                                                             retriever=retriever,
                                                             prompt=context_q_prompt)
    
    # Prompt para perguntas e respostas (Q&A)
    qa_prompt_template = """Você é um assistente virtual prestativo e está respondendo perguntas gerais. 
    Use os seguintes pedaços de contexto recuperado para responder à pergunta. 
    Se você não sabe a resposta, apenas diga que não sabe. Mantenha a resposta concisa. 
    Responda em português. \n\n
    Pergunta: {input} \n
    Contexto: {context}"""

    qa_prompt = PromptTemplate.from_template(token_s + qa_prompt_template + token_e)

    # Configurar LLM e Chain para perguntas e respostas (Q&A)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain,)

    return rag_chain


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá, sou o seu assistente virtual! Como posso ajudar você?"),
    ]

if "docs_list" not in st.session_state:
    st.session_state.docs_list = None

if "retriever" not in st.session_state:
    # st.session_state.retriever = None
    st.session_state.retriever = config_retriever()

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

start = time.time()
user_query = st.chat_input("Digite sua mensagem aqui...")

if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        # Usando o retriever já configurado para buscar e responder à consulta
        rag_chain = config_rag_chain(model_class, st.session_state.retriever)

        result = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})

        resp = result['answer']
        st.write(resp)

        # Mostrar a fonte do conteúdo recuperado
        sources = result['context']
        for idx, doc in enumerate(sources):
            source = doc.metadata['source']
            file = os.path.basename(source)
            page = doc.metadata.get('page', 'Página não especificada')

            # Fonte 1: documento.pdf - p. 2
            ref = f":link: Fonte {idx}: *{file} - p. {page}*"
            with st.popover(ref):
                st.caption(doc.page_content)

    st.session_state.chat_history.append(AIMessage(content=resp))

# if user_query is not None and user_query != "" and docs is not None:
#     st.session_state.chat_history.append(HumanMessage(content=user_query))

#     with st.chat_message("Human"):
#         st.markdown(user_query)

#     with st.chat_message("AI"):
#         if st.session_state.docs_list != uploads:
#             st.session_state.docs_list = uploads
#             st.session_state.retriever = config_retriever(uploads)
        
#         rag_chain = config_rag_chain(model_class, st.session_state.retriever)

#         result = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})

#         resp = result['answer']
#         st.write(resp)

#         # Mostrar a fonte
#         sources = result['context']
#         for idx, doc in enumerate(sources):
#             source = doc.metadata['source']
#             file = os.path.basename(source)
#             page = doc.metadata.get('page', 'Página não especificada')

#             # Fonte 1: documento.pdf - p. 2
#             ref = f":link: Fonte {idx}: *{file} - p. {page}*"
#             with st.popover(ref):
#                 st.caption(doc.page_content)

#     st.session_state.chat_history.append(AIMessage(content=resp))

end = time.time()
print("Tempo: ", end - start)
