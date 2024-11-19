import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import wikipediaapi
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Configuração inicial do Streamlit
st.set_page_config(page_title="Converse com a Wikipedia 🌐", page_icon="📚")
st.title("Converse com a Wikipedia 🌐")

model_class = "hf_hub"  # Escolha entre "hf_hub", "openai", "ollama"

# Função para carregar o modelo
def model_hf_hub(model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1):
    from langchain_community.llms import HuggingFaceHub
    llm = HuggingFaceHub(
        repo_id=model,
        model_kwargs={
            "temperature": temperature,
            "return_full_text": False,
            "max_new_tokens": 512,
        }
    )
    return llm

def model_openai(model="gpt-4o-mini", temperature=0.1):
    llm = ChatOpenAI(
        model=model,
        temperature=temperature
    )
    return llm

def model_ollama(model="phi3", temperature=0.1):
    llm = ChatOllama(
        model=model,
        temperature=temperature,
    )
    return llm

# Função para buscar artigos na Wikipedia
def fetch_wikipedia_pages(query, language="pt"):
    valid_languages = ["en", "pt", "es", "fr", "de", "it"]  # Adicione mais idiomas se necessário
    if language not in valid_languages:
        raise ValueError(f"Idioma '{language}' não suportado. Use um dos seguintes: {', '.join(valid_languages)}")
    
    wiki_wiki = wikipediaapi.Wikipedia(language)
    page = wiki_wiki.page(query)
    
    if not page.exists():
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document = Document(page_content=page.text, metadata={"title": page.title})
    return text_splitter.split_documents([document])


# Configuração do retriever
def config_retriever(query):
    docs = fetch_wikipedia_pages(query)

    if not docs:
        st.error("Nenhuma página encontrada para a consulta.")
        return None

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 4})
    return retriever

# Configuração do RAG Chain
def config_rag_chain(model_class, retriever):
    if model_class == "hf_hub":
        llm = model_hf_hub()
    elif model_class == "openai":
        llm = model_openai()
    elif model_class == "ollama":
        llm = model_ollama()

    token_s, token_e = "", ""
    if model_class.startswith("hf"):
        token_s = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        token_e = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    # Configuração do prompt para QA
    qa_prompt_template = """Você é um assistente virtual prestativo e está respondendo perguntas gerais. 
    Use os seguintes pedaços de contexto recuperado para responder à pergunta. 
    Se você não sabe a resposta, apenas diga que não sabe. Mantenha a resposta concisa. 
    Responda em português. \n\n
    Pergunta: {input} \n
    Contexto: {context}"""

    qa_prompt = PromptTemplate.from_template(token_s + qa_prompt_template + token_e)

    # Configurar LLM e Chain para perguntas e respostas (Q&A)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    return rag_chain

# Inicializar histórico
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá, sou o seu assistente virtual! Como posso ajudar você?"),
    ]

# Exibição do histórico
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Entrada do usuário
start = time.time()
user_query = st.chat_input("Digite sua mensagem aqui...")

if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    # Configurar retriever e RAG Chain
    retriever = config_retriever(user_query)
    if retriever:
        rag_chain = config_rag_chain(model_class, retriever)

        result = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
        resp = result['answer']

        with st.chat_message("AI"):
            st.write(resp)

            # Mostrar fontes
            sources = result.get('context', [])
            for idx, doc in enumerate(sources):
                title = doc.metadata.get('title', 'Título desconhecido')
                ref = f":link: Fonte {idx}: *{title}*"
                with st.popover(ref):
                    st.caption(doc.page_content)

        st.session_state.chat_history.append(AIMessage(content=resp))

end = time.time()
print("Tempo: ", end - start)
