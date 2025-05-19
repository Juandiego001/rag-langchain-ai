import streamlit as st
from rag import get_response_model, CHAT_MODEL

st.title('RAG AI With Chatbot')

if 'model_selection' not in st.session_state:
    st.session_state.model_selection = CHAT_MODEL

if 'historial' not in st.session_state:
    st.session_state.historial = []

pdf_file = st.file_uploader('Sube un PDF', type=['pdf'])

for mensaje in st.session_state.historial:
    with st.chat_message(mensaje['role']):
        st.write(mensaje['content'])

user_input = st.chat_input('Haz tu pregunta: ')

if user_input and pdf_file:
    st.session_state.historial.append({
        'role': 'user',
        'content': user_input
    })

    with st.chat_message('user'):
        st.write(user_input)

    try:
        with st.spinner('Cargando...'):
            response = get_response_model(user_input, pdf_file)

            st.session_state.historial.append({
                'role': 'assistant',
                'content': response['answer'],
            })

            with st.chat_message('assistant'):
                st.write(response['answer'])

    except Exception as e:
        st.error(f'Error: {e}')
