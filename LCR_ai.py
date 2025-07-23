import tempfile

import streamlit as st
from langchain.memory import ConversationBufferMemory


from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama


from loaders import *

TIPOS_ARQUIVOS_VALIDOS = [
    'Site', 'Youtube', 'Pdf', 'Csv', 'Txt']

CONFIG_MODELOS = {
                    'Groq': {'modelos': ['llama-3.1-8b-instant', 'gemma2-9b-it', 'deepseek-r1-distill-llama-70b', 'whisper-large-v3-turbo'],
                         'chat': ChatGroq},
                    'OpenAI':{'modelos': ['gpt-4o-mini', 'gpt-4o', 'o1-preview', 'o1-mini'],
                         'chat': ChatOpenAI},
                    'Ollama - Local':{'modelos': ['gemma3:latest', 'llama3:latest'],
                    'chat': ChatOllama}}

MEMORIA = ConversationBufferMemory()

# Função para carregar os arquivos de acordo com o tipo
def carrega_arquivos(tipo_arquivo, arquivo):
    if tipo_arquivo == 'Site':
        documento = carrega_site(arquivo)
    if tipo_arquivo == 'youtube':
        documento = carrega_youtube(arquivo)
    if tipo_arquivo == 'Pdf':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_pdf(nome_temp)
    if tipo_arquivo == 'Csv':
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_csv(nome_temp)
    if tipo_arquivo == 'Txt':
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_txt(nome_temp)
    return documento

# Função para carregar o modelo de acordo com o provedor e o modelo selecionado
def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
    documento = carrega_arquivos(tipo_arquivo, arquivo)

    system_message = '''Você é um assistente amigável chamado LCR ai.
    Você possui acesso às seguintes informações vindas 
    de um documento {nome_documento}: 

    ####
    {conteudo_documento}
    ####

    Você deve responder de forma clara e objetiva, 
    sempre com base nas informações do documento carregado. Quando for solicitado informações do documento.'''

    print(system_message)

    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])

    chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)
    chain = template | chat

    st.session_state['chain'] = chain
    st.session_state['nome_documento'] = tipo_arquivo
    st.session_state['conteudo_documento'] = documento

# Pagina de configuração do modelo e chat
def pagina_chat():
    st.header('🤖Bem-vindo ao LCR ai!', divider=True)

    chain = st.session_state.get('chain')
    if chain is None:
        st.error('Carrege o documento!')
        st.stop()

    memoria = st.session_state.get('memoria', MEMORIA)
    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    input_usuario = st.chat_input('Fale com o LCR ai')
    if input_usuario:
        chat = st.chat_message('human')
        chat.markdown(input_usuario)

        chat = st.chat_message('ai')
        resposta = chat.write_stream(chain.stream({
            'input': input_usuario, 
            'chat_history': memoria.buffer_as_messages,
            'nome_documento': st.session_state.get('nome_documento', ''),
            'conteudo_documento': st.session_state.get('conteudo_documento', '')
            }))
        
        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria

def sidebar():
    tabs = st.tabs(['Upload de Arquivos', 'Seleção de Modelos'])
    with tabs[0]:
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo', TIPOS_ARQUIVOS_VALIDOS)
        if tipo_arquivo == 'Site':
            arquivo = st.text_input('Digite a url do site')
        if tipo_arquivo == 'Youtube':
            arquivo = st.text_input('Digite a url do vídeo')
        if tipo_arquivo == 'Pdf':
            arquivo = st.file_uploader('Faça o upload do arquivo pdf', type=['.pdf'])
        if tipo_arquivo == 'Csv':
            arquivo = st.file_uploader('Faça o upload do arquivo csv', type=['.csv'])
        if tipo_arquivo == 'Txt':
            arquivo = st.file_uploader('Faça o upload do arquivo txt', type=['.txt'])
    with tabs[1]:
        provedor = st.selectbox('Selecione o provedor dos modelo', CONFIG_MODELOS.keys())
        modelo = st.selectbox('Selecione o modelo', CONFIG_MODELOS[provedor]['modelos'])
        api_key = st.text_input(
            f'Adicione a api key para o provedor {provedor}',
            value=st.session_state.get(f'api_key_{provedor}'))
        st.session_state[f'api_key_{provedor}'] = api_key
    
    if st.button('Inicializar LCR ai', use_container_width=True):
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    if st.button('Apagar Histórico de Conversa', use_container_width=True):
        st.session_state['memoria'] = MEMORIA

def main():
    with st.sidebar:
        sidebar()

    pagina_chat()

if __name__ == '__main__':
    main()
