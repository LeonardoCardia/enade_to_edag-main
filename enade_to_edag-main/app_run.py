# Código digitado por Leonardo Cardia da Cruz => referência app.py


#TODO: Refator para few-show prompting
#O "few-shot prompting" => é uma técnica de engenharia de prompts que envolve fornecer um modelo de linguagem com um pequeno 
# número de exemplos para guiar sua resposta a uma tarefa específica. 
# "zero-shot prompting" => onde nenhum exemplo é fornecido
# "one-shot prompting" => onde apenas um exemplo é dado. 



# Execution
# streamlit run app.py

#####################

# IMPORTAÇÕES

#####################

import streamlit as st
import pandas as pd
import ast
import glob
import os
import base64
from openai import OpenAI
import re
import time


##################

# INITIALIZING API CLIENT

##################

# Initializing API client
key = st.secrets["groq"]["key"]
os.environ['OPENAI_API_KEY'] = key
client = OpenAI(
    base_url='https://api.groq.com/openai/v1',
    api_key=os.environ['OPENAI_API_KEY'],
)



#############

# HELPER FUNCTIONS

##############

# Function to load dataframe's edag topics information (and save it in cache)
@st.cache_data
def load_edag_topics(path='data/enade_data.csv'):
    df = pd.read_csv(path, converters={'test_content_edag': ast.literal_eval})
    
# Function to load files
def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    
# Function to encode images (file objects) coming from the user
def encode_image_fileobj(file_obj):
    file_obj.seek(0)
    return base64.b64encode(file_obj.read()).decode('utf-8') 

# Function to encode images so we can send them to the model
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8') 

# TODO: Realizar o refactoring
# Function to validate question format
def validate_question_format(text, fmt):
    # Question starts with “ENUNCIADO:” and contains “JUSTIFICATIVA:”?
    if not text.startswith("ENUNCIADO:") or "JUSTIFICATIVA:" not in text:
        return False

    # Trying to get rid of questions on the introduction text without filtering too much for the model's creativity
    try:
        intro = text.split("ENUNCIADO:\n",1)[1].split("\n\n",1)[0]
        if "?" == intro[-1]:
            return False
    except:
        return False

    # Testing regex for question structure
    if fmt == 'resposta_unica':
        pattern = r"(?s)^ENUNCIADO:\n[^\n]+\n\n(?:```.+?```\n\n)?[^\n]+\n\n\(A\) [^\n]+\n\(B\) [^\n]+\n\(C\) [^\n]+\n\(D\) [^\n]+\n\(E\) [^\n]+\n\nJUSTIFICATIVA:\n\(A\) [^\n]+\n\(B\) [^\n]+\n\(C\) [^\n]+\n\(D\) [^\n]+\n\(E\) [^\n]+$"

    elif fmt == 'resposta_multipla':
        pattern = r"(?s)^ENUNCIADO:\n[^\n]+\n\n(?:```.+?```\n\n)?I\. [^\n]+\nII\. [^\n]+\nIII\. [^\n]+\nIV\. [^\n]+\n\nÉ correto apenas o que se afirma em:\n\n\(A\) I\n\(B\) II e IV\n\(C\) III e IV\n\(D\) I, II e III\n\(E\) I, II, III e IV\n\nJUSTIFICATIVA:\nI\. [^\n]+\nII\. [^\n]+\nIII\. [^\n]+\nIV\. [^\n]+\n\nPortanto a alternativa correta é \(?[A,B,C,D,E]\)?$"

    elif fmt == 'discursiva':
        pattern = r"(?s)^ENUNCIADO:\n[^\n]+\n\n(?:```.+?```\n\n)?[^\n]+\n\nJUSTIFICATIVA:\n.+$"

    elif fmt == 'assercao_razao':
        pattern = r"(?s)^ENUNCIADO:\n[^\n]+\n\n(?:```.+?```\n\n)?Nesse contexto, avalie as asserções a seguir e a relação proposta entre elas:\n\nI\. [^\n]+\n\n\*\*PORQUE\*\*\n\nII\. [^\n]+\n\nÀ respeito dessas asserções, assinale a opção correta:\n\n\(A\) As asserções I e II são proposições verdadeiras, e a II é uma justificativa correta da I\.\n\(B\) As asserções I e II são proposições verdadeiras, mas a II não é uma justificativa correta da I\.\n\(C\) A asserção I é uma proposição verdadeira, e a II é uma proposição falsa\.\n\(D\) A asserção I é uma proposição falsa, e a II é uma proposição verdadeira\.\n\(E\) As asserções I e II são proposições falsas\.\n\nJUSTIFICATIVA:\nI\. [^\n]+\nII\. [^\n]+\n\n[^\n]+$"
    
    return bool(re.match(pattern, text, flags=0)) 



# Function modal with decorator to show new question
@st.dialog('Nova Questão Gerada')
def show_new_q():
    # Rendering question generation error
    if st.session_state.modal_error:
        st.error(st.session_state.modal_error)
        st.session_state.modal_error = None

    # Question editing mode
    if st.session_state.editing_question:
        new_md = st.text_area("Edite sua questão:", value=st.session_state.modal_content, height=500, key="md_editor")
        
        # Buttons to save or cancel edit
        col_save, col_cancel = st.columns([1, 1], gap="small")
        with col_save:
            if st.button("Salvar Edição", key="save_edit"):
                st.session_state.modal_content = new_md
                st.session_state.editing_question = False
                st.rerun()

        with col_cancel:
            if st.button("Cancelar Edição", key="cancel_edit"):
                st.session_state.editing_question = False
                st.rerun()

    # Buttons for either downloading generated question, editing it or closing the modal
    else:
        # Rendering generated question
        # st.text(st.session_state.modal_content)
        st.markdown(st.session_state.modal_content, unsafe_allow_html=True)

        col_ed, col_dl, col_close = st.columns([1, 1, 1], gap="small")
        with col_ed:
            if st.button("Editar Questão", key="edit_modal"):
                st.session_state.editing_question = True
                st.rerun()
        
        with col_dl:
            st.download_button(label="Baixar Questão", data=st.session_state.modal_content, file_name="nova_questao.md", mime="text/markdown")

        with col_close:
            if st.button("Fechar", key="close_modal"):
                st.session_state.show_modal_question = False
                st.rerun()




########################

# PAGE CONFIGURATION 

#######################
st.set_page_config(page_title='Gerador de Questões', layout='wide', initial_sidebar_state='collapsed')

# TODO: Refatorar -> Colocar em um arquivo separado

# Custom CSS for streamlit
st.markdown(
    """
    <style>
        /* Centering the main title */
        div[data-testid="stMarkdown"] {
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
        }

        div[data-testid="stHeadingWithActionElements"] {
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
        }

        /* Bigger labels for selectors */
        div[data-testid="stMarkdownContainer"] {
            font-size: 24px !important;
        }

        /* Reducing spacing between labels and text areas */
        div[data-testid="stSubHeader"] > label {
            margin-bottom: 0px !important;
        }

        /* Creating a bit more space between block in a horizontal block */
        div[data-testid="stHorizontalBlock"] {
            gap: 1.2rem !important;
        }

        /* Centralizing columns for button that generates questions */
        div[data-testid="stColumn"]:has(> div > div > div > div[data-testid="stButton"]) {
            display: flex !important;
            justify-content: center !important; 
            align-items: center !important;
        }

        /* Centralizing slider and making it smaller */
        div[data-baseweb="slider"] {
            padding-left: 1em !important;
            width: 90% !important;
        }

        /* Making buttons larger */
        button[data-testid="stBaseButton-secondary"] {
            height: 4em !important;
            width: 12em !important;
        }

        /* Centralizing drag and drop of files */
        section[data-testid="stFileUploaderDropzone"] {
            align-items: center !important;
        }

        div[data-testid="stFileUploaderDropzoneInstructions"] {
            margin-right: 0 !important;
        }    

        /* Centering provas antigas title */
        h2 {
            text-align: center;
            font-weight: bold !important;
            margin-top: 2em !important;
        }

        /* Making year separation bigger and centralized */
        .year-separation {
            font-size: 40px !important;
            line-height: 1.5em !important;
        }

        /* Centralizing grid buttons */
        div[data-testid="stButton"] {
            display: flex !important;
            justify-content: center !important;
        }

        /* Separating grid lines */
        div[data-testid="stHorizontalBlock"]:has(> div > div > div > div > div > div > div[data-testid="stImage"]) {
            margin-bottom: 3em !important;
            padding-bottom: 2em !important;
            border-bottom: 2px dashed #aaa !important;
        }

        /* Resizing images on grid */
        div[data-testid="stHorizontalBlock"] > div > div > div > div > div > div > div > div[data-testid="stImageContainer"] {
            height: 30em !important;
            width: 20em !important;
        }
        
        img {
            height: 100% !important;
            object-fit: contain !important;
        }

        /* Removing toolbar from images */
        div[data-testid="stElementToolbar"] {
            display: none !important;
        }
        
        /* Resizing selected image */
        div[data-testid="stElementContainer"] > div > div > div > div[data-testid="stImageContainer"] {
            height: 60em !important;
        }

        /* Centralizing selected image */
        div[data-testid="stFullScreenFrame"] {
            display: flex !important;
            justify-content: center !important;
        }

        /* Increasing legend size for selected image */
        div[data-testid="stCaptionContainer"] {
            font-size: x-large !important;
            colot: black !important
        }

        /* Changing size of modal for newly generated question */
        div[data-testid="stDialog"] > div > div {
            width: 100em !important;
        }

        /* Removing small 'x' from modal to better control enviroment variable */
        button[aria-label="Close"] {
            display: None !important
        }

        /* Fixing download button positioning on new question modal */
        div[data-testid="stDownloadButton"] {
            display: flex !important;
            justify-content: center !important;
        }

        /* Centering the tab container */
        div[data-baseweb="tab-list"] {
          display: flex !important;
          justify-content: space-around !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)

# State variable later sed for card (question) selection
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = None
    
# State variables for new question modal
if 'show_modal_question' not in st.session_state:
    st.session_state.show_modal_question = False
    
if 'modal_content' not in st.session_state:
    st.session_state.modal_content = ""
    
if 'editing_question' not in st.session_state:
    st.session_state.editing_question = False
    
if 'modal_error' not in st.session_state:
    st.session_state.modal_error = None
    
# State variables for ENADE historic analysis modal
if 'show_modal_enade' not in st.session_state:
    st.session_state.show_modal_enade = False
    
# TODO: Colocar em uma arquivo JSON
# Hard coded list of content from CIMATEC's perspective
edag_content_list = ['algoritmos e estrutura de dados', 'arquitetura de computadores', 'banco de dados', 'cibersegurança', 'ciência de dados', 'elétrica e eletrônica', 'engenharia de software', 'grafos', 'inteligência artificial', 'iot', 'lógica de programação', 'processamento de sinais', 'redes de computadores', 'robótica, automação e controle', 'sistemas digitais', 'sistemas distribuídos e programação paralela', 'sistemas embarcados', 'sistemas operacionais e compiladores', 'outros']

# Loading topics
edag_topics_by_year = load_edag_topics()

# Addition to the title
st.markdown(
    """
    <div>
        <h1>Gerador de Questões</h1>
        <h4>Um Estudo de Caso para Engenharia de Computação</h4>
    </div>
    """,
    unsafe_allow_html=True, help="""Ferramenta designada para geração de questões utilizando modelos generativos e dentro de alguns parâmetros pré-estabelecidos: formato da questão, dificuldade, instruções adicionais opcionais através de prompt e suporte gráfico opcional através de imputação de imagem. A ferramenta também conta com um grid de questões de provas antigas do ENADE, as quais podem ser selecionadas para geração mais guiada de um nova questão.\n\nEsse software é um projeto experimental na área de Engenharia de Computação, com conteúdo focado na matriz curricular do curso no SENAI CIMATEC e, mais especificamente, pensado para ajudar no desenvolvimento de novas provas do EDAG, o exame interno da universidade para avaliação dos estudantes de graduação.""")

# Topic selector
topics_display = st.multiselect('Escolha um ou mais tópicos', [t.title() for t in edag_content_list], placeholder='Todos os tópicos', help='Seletor de tópicos para gerar uma nova questão: o modelo usará os tópicos marcados (ou todos, se nenhum for escolhido) para criar uma pergunta. No caso de visualização de provas antigas do ENADE, o seletor filtra as questões por tópico. Note que a geração de uma questão pode combinar vários tópicos selecionados, então se você precisa de uma pergunta focada em um único tópico, selecione apenas aquele tópico.')
topics = [t.lower() for t in topics_display]

# Text box for format selection, additional instructions and button to generate question
fmt_col, gen_col, upload_col, btn_col = st.columns([1, 2, 1, 1])

# Format mapping
format_files = glob.glob('data/edag_question_formats/*.txt')
format_names = [os.path.basename(f) for f in format_files]
display_formats = [os.path.splitext(name)[0].replace('_', ' ').title() for name in format_names]
fmt_map = dict(zip(display_formats, format_names))

chosen_fmt = fmt_col.selectbox('Formato da Nova Questão', display_formats, help='Seletor de formato da nova questão baseado nos direcionamentos de padrão do EDAG.')
fmt_filter = fmt_map[chosen_fmt]

difficulty = fmt_col.select_slider('Nível de Dificuldade', ['Fácil', 'Médio', 'Difícil'], help='Seletor do nível de dificuldade da nova questão a ser gerada. Por conta da complexidade e subjetividaded inata em determinar o nível de dificuldade de uma questão, atente-se ao fato de que esse slider não garante uma questão fácil ou difícil.')

user_prompt = gen_col.text_area('Instruções Adicionais (opcional)', height=155, help='Área de texto para prompts adicionais e pedidos específicos que o usuário possar ter ao modelo quando da geração de uma nova questão.')

# TODO: Criar uma abordagem que o próprio modelo  gera uma imagem de suporte a partir de especificaçoes passadas, ou até a partir do tipo de questão gerada só apresenta sugestões.
uploaded_graphic = upload_col.file_uploader('Suporte Gráfico (opcional)', type=['png'], help='Opção de upload de imagem (em formato PNG) para que o modelo possa utilizar quando da geração de uma nova questão. Note que a imagem aqui fornecida será necessariamente adicionada à questão, não apenas utilizada como inspiração.')

generate_clicked = btn_col.button('Gerar Questão')

# TODO: Criar um arquivo JSON que vai ter a estrutura para o agente CONTENT, ROLE, etc.

#TODO: Criar outro botão para ver novamente a questão gerada anteriormente.

# Question Generation Logic
if generate_clicked:
    # Building up pipeline message
    msgs = []
    sys_content = (
        "Sua função é gerar uma nova questão de prova dentro dos [TÓPICOS] fornecidos, na [DIFICULDADE] fornecida e seguindo exatamente o [FORMATO DE SAÍDA] \
        fornecido através do preenchimento dos trechos indicados por '<>'. Não adicione comentários, cabeçalhos, explicações, saudações ou qualquer texto extra. \
        Retorne apenas o texto da nova questão, nada mais. Caso haja [INSTRUÇÕES ADICIONAIS], siga exatamente o que for pedido. Caso haja uma imagem [ANEXO GRÁFICO], \
        use como suporte gráfico na geração da nova questão. Por fim, caso haja uma imagem [QUESTÃO BASE], faça uma nova versão da questão base, ainda seguindo \
        o [FORMATO DE SAÍDA] fornecido."
    )
    
    # ROLE IS 'SYSTEM'
    msgs.append({'role':'system', 'content':sys_content})
    
    # Basic text with topics and format
    if len(topics) != 0:
        text_block = f"\n\n[TÓPICOS]\n{topics}\n\n[DIFICULDADE]\n{difficulty}\n\n[FORMATO DE SAÍDA]\n{load_file(f'data/edag_question_formats/{fmt_filter}') }"
    else:
        text_block = f"\n\n[TÓPICOS]\n{edag_content_list}\n\n[DIFICULDADE]\n{difficulty}\n\n[FORMATO DE SAÍDA]\n{load_file(f'data/edag_question_formats/{fmt_filter}') }"

    # Adjust for user instructions
    if user_prompt:
        text_block += f"\n\n[INSTRUÇÕES ADICIONAIS]\n{user_prompt}"
    
    # Adding user content
    content_list = []
    
    # Adjut for graphic support
    if uploaded_graphic is not None:
        graphic_b64 = encode_image_fileobj(uploaded_graphic)
        content_list.append({'type': 'text', 'text': '\n\n[ANEXO GRÁFICO]\n'})
        content_list.append({'type':'image_url','image_url':{'url':f"data:image/png;base64,{graphic_b64}"}})
        
    # Adjust for selected question
    if st.session_state.selected_question:
        path = st.session_state.selected_question['path']
        img_b64 = encode_image(path)
        content_list.append({'type': 'text', 'text': '\n\n[QUESTÃO BASE]\n'})
        content_list.append({'type':'image_url','image_url':{'url': st.session_state.selected_question['url']}})
    
    # ROLE IS 'user'
    # Creating message for groq models
    msgs.append({'role':'user','content':[{'type':'text','text':text_block}] + content_list})
    
    # Trying to generate question
    max_attempts = 3
    new_q = None
    server_error = False
    
    for attempt in range(max_attempts):
        # API call
        try:
            resp = client.chat.completions.create(
                model='meta-llama/llama-4-maverick-17b-128e-instruct',
                messages=msgs,
                temperature=0.8,
                max_tokens=1024
            )
        except:
            server_error = True
            break
        
        # Validating question
        candidate = resp.choices[0].message.content.strip()
        
        #TODO: Try except ficaria melhor
        if validate_question_format(candidate, fmt_filter.split('.')[0]):
            new_q = candidate.replace('\n', ' \n')
            break
        
        # If failed, changing message to try again
        if attempt == 0:
            msgs.append({
                'role':'user',
                'content': ("O formato da questão não seguiu exatamente o template. Por favor, gere novamente exatamente no formato fornecido.")
            })
        time.sleep(5)
        
    if new_q:
        # Adding Anexo Gráfico to question if it exists
        if uploaded_graphic is not None:
            new_q = new_q.replace('  \n  \n', f'  \n  \n![Anexo Gráfico](data:image/png;base64,{graphic_b64})  \n  \n', 1)

        st.session_state.show_modal_question = True
        st.session_state.modal_content = new_q
        
    elif server_error:
        server_error = False
        st.session_state.show_modal_question = True
        st.session_state.modal_content = ''
        st.session_state.modal_error = f"Não consegui gerar a questão por problemas no servidor."
        
    else:
        candidate = candidate.replace('\n', '  \n')

        # Adding Anexo Gráfico to question if it exists
        if uploaded_graphic is not None:
            candidate = candidate.replace('  \n  \n', f"  \n  \n![Anexo Gráfico](data:image/png;base64,{graphic_b64})  \n  \n", 1)

        st.session_state.show_modal_question = True
        st.session_state.modal_content = candidate
        st.session_state.modal_error = f"Não consegui gerar a questão no formato correto após {max_attempts} tentativas, mas segue uma questão candidata."
        
# Creating modal with new question
if st.session_state.show_modal_question:
    show_new_q()

