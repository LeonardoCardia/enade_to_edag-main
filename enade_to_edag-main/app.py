import os
import re
import time
import base64
import ast
from pathlib import Path
from collections import Counter

from PIL import Image
import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FORMATS_DIR = DATA_DIR / "edag_question_formats"
VISUAL_DIR = DATA_DIR / "visual_approach"
ENADE_CSV = DATA_DIR / "enade_data.csv"


def load_file(file_path: Path | str) -> str:
    file_path = Path(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_image(path: Path | str):
    return Image.open(Path(path))


def encode_image(image_path: Path | str) -> str:
    image_path = Path(image_path)
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_image_fileobj(file_obj) -> str:
    file_obj.seek(0)
    return base64.b64encode(file_obj.read()).decode("utf-8")


@st.cache_data
def load_edag_topics():
    df = pd.read_csv(
        ENADE_CSV,
        converters={"test_content_edag": ast.literal_eval},
    )
    return {row["year"]: row["test_content_edag"] for _, row in df.iterrows()}


@st.cache_data
def load_question_formats():
    if not FORMATS_DIR.exists():
        return [], [], {}

    format_files = sorted(FORMATS_DIR.glob("*.txt"))
    display_formats = [f.stem.replace("_", " ").title() for f in format_files]
    format_names = [f.name for f in format_files]
    fmt_map = {disp: fname for disp, fname in zip(display_formats, format_names)}

    return display_formats, format_names, fmt_map


@st.cache_data
def get_year_dirs():
    if not VISUAL_DIR.exists():
        return []
    return sorted([p for p in VISUAL_DIR.glob("prova_*") if p.is_dir()])


@st.cache_data
def get_years():
    return sorted([int(p.name.split("_")[-1]) for p in get_year_dirs()])


@st.cache_data
def get_raw_types():
    raw_types = set()
    for year_dir in get_year_dirs():
        for file_path in year_dir.iterdir():
            if file_path.is_file():
                raw_types.add(file_path.name.split("_")[0])
    return sorted(raw_types)


def validate_question_format(text, fmt):
    if not text.startswith("ENUNCIADO:") or "JUSTIFICATIVA:" not in text:
        return False

    try:
        intro = text.split("ENUNCIADO:\n", 1)[1].split("\n\n", 1)[0]
        if intro and intro[-1] == "?":
            return False
    except Exception:
        return False

    if fmt == "resposta_unica":
        pattern = r"(?s)^ENUNCIADO:\n[^\n]+\n\n(?:```.+?```\n\n)?[^\n]+\n\n\(A\) [^\n]+\n\(B\) [^\n]+\n\(C\) [^\n]+\n\(D\) [^\n]+\n\(E\) [^\n]+\n\nJUSTIFICATIVA:\n\(A\) [^\n]+\n\(B\) [^\n]+\n\(C\) [^\n]+\n\(D\) [^\n]+\n\(E\) [^\n]+$"

    elif fmt == "resposta_multipla":
        pattern = r"(?s)^ENUNCIADO:\n[^\n]+\n\n(?:```.+?```\n\n)?I\. [^\n]+\nII\. [^\n]+\nIII\. [^\n]+\nIV\. [^\n]+\n\nÉ correto apenas o que se afirma em:\n\n\(A\) I\n\(B\) II e IV\n\(C\) III e IV\n\(D\) I, II e III\n\(E\) I, II, III e IV\n\nJUSTIFICATIVA:\nI\. [^\n]+\nII\. [^\n]+\nIII\. [^\n]+\nIV\. [^\n]+\n\nPortanto a alternativa correta é \(?[A,B,C,D,E]\)?$"

    elif fmt == "discursiva":
        pattern = r"(?s)^ENUNCIADO:\n[^\n]+\n\n(?:```.+?```\n\n)?[^\n]+\n\nJUSTIFICATIVA:\n.+$"

    elif fmt == "assercao_razao":
        pattern = r"(?s)^ENUNCIADO:\n[^\n]+\n\n(?:```.+?```\n\n)?Nesse contexto, avalie as asserções a seguir e a relação proposta entre elas:\n\nI\. [^\n]+\n\n\*\*PORQUE\*\*\n\nII\. [^\n]+\n\nÀ respeito dessas asserções, assinale a opção correta:\n\n\(A\) As asserções I e II são proposições verdadeiras, e a II é uma justificativa correta da I\.\n\(B\) As asserções I e II são proposições verdadeiras, mas a II não é uma justificativa correta da I\.\n\(C\) A asserção I é uma proposição verdadeira, e a II é uma proposição falsa\.\n\(D\) A asserção I é uma proposição falsa, e a II é uma proposição verdadeira\.\n\(E\) As asserções I e II são proposições falsas\.\n\nJUSTIFICATIVA:\nI\. [^\n]+\nII\. [^\n]+\n\n[^\n]+$"

    else:
        return False

    return bool(re.match(pattern, text, flags=0))


@st.dialog("Nova Questão Gerada")
def show_new_q():
    if st.session_state.modal_error:
        st.error(st.session_state.modal_error)
        st.session_state.modal_error = None

    if st.session_state.editing_question:
        new_md = st.text_area(
            "Edite sua questão:",
            value=st.session_state.modal_content,
            height=500,
            key="md_editor",
        )

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

    else:
        st.markdown(st.session_state.modal_content, unsafe_allow_html=True)

        col_ed, col_dl, col_close = st.columns([1, 1, 1], gap="small")
        with col_ed:
            if st.button("Editar Questão", key="edit_modal"):
                st.session_state.editing_question = True
                st.rerun()

        with col_dl:
            st.download_button(
                label="Baixar Questão",
                data=st.session_state.modal_content,
                file_name="nova_questao.md",
                mime="text/markdown",
            )

        with col_close:
            if st.button("Fechar", key="close_modal"):
                st.session_state.show_modal_question = False
                st.rerun()


def adjust_layout(fig, all_x=False):
    if not all_x:
        fig.update_xaxes(showticklabels=False, title_text="Tópicos")

    fig.update_layout(
        xaxis=dict(type="category"),
        xaxis_title_font_size=20,
        yaxis_title_font_size=20,
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16,
        hoverlabel=dict(font_size=16),
        legend_title_font_size=20,
        legend_font_size=16,
        bargap=0.4,
        legend_traceorder="reversed",
    )

    if all_x:
        fig.update_layout(legend=dict(itemclick="toggleothers"))
        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Representou %{customdata[1]:.1f}% do exame<extra></extra>"
            )
        )
    else:
        fig.update_layout(legend=dict(itemclick=False, itemdoubleclick=False))
        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]} questões no exame<extra></extra>"
            )
        )

    return fig


@st.dialog("Análise Histórica ENADE")
def show_history():
    edag_topics_by_year = load_edag_topics()
    records = []

    year_order = [2019, 2017, 2014, 2023]
    for year in year_order:
        if year not in edag_topics_by_year:
            continue

        qdict = edag_topics_by_year[year]
        total_q = len(qdict)

        counter = Counter()
        for topics in qdict.values():
            counter.update(topics)
        total_occ = sum(counter.values()) if counter else 1

        for topic, occ in sorted(counter.items(), key=lambda kv: kv[0], reverse=True):
            pct = occ / total_occ
            records.append(
                {
                    "year": str(year),
                    "topic": topic.title(),
                    "percent": pct * 100,
                    "occurrences": occ,
                    "total_q": total_q,
                }
            )

    if not records:
        st.warning("Não há dados históricos suficientes para exibir.")
        if st.button("Fechar", key="close_history"):
            st.session_state.show_modal_enade = False
            st.rerun()
        return

    enade_exam_df = pd.DataFrame(records)

    st.markdown("")
    st.markdown(
        "<p style='text-align:center; font-weight:bold;'>Distribuição de Conteúdos do SENAI CIMATEC por Ano de Prova do ENADE</p>",
        unsafe_allow_html=True,
        help="Note que as questões podem conter múltiplos conteúdos, portanto a somatória do número de questões por conteúdo não resulta necessariamente no número total de questões da prova. As provas do ENADE para Engenharia de Computação sempre têm exatamente 40 questões.",
    )

    year_options = sorted({int(y) for y in enade_exam_df["year"]})
    topics_sorted = enade_exam_df["topic"].unique()

    tab_overview, tab_by_year = st.tabs(
        ["Perfil Geral das Provas", "Perfil de Prova por Ano"]
    )

    with tab_overview:
        fig1 = px.bar(
            enade_exam_df,
            x="year",
            y="percent",
            color="topic",
            custom_data=["topic", "percent"],
            barmode="stack",
            labels={
                "year": "Ano do Exame",
                "percent": "Percentual da Prova Representado",
                "topic": "Tópico",
            },
            height=600,
            category_orders={"year": year_options, "topic": topics_sorted},
        )
        fig1 = adjust_layout(fig1, all_x=True)
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    with tab_by_year:
        col1, col2, col3 = st.columns([1, 4, 2])
        with col2:
            selected_year = st.select_slider("", options=year_options, value=year_options[0])
            df_year = enade_exam_df[enade_exam_df["year"] == str(selected_year)]

        fig2 = px.bar(
            df_year,
            x="topic",
            y="occurrences",
            color="topic",
            custom_data=["topic", "occurrences"],
            labels={"topic": "Tópico", "occurrences": "Número de Questões"},
            category_orders={"topic": topics_sorted},
            height=600,
        )
        fig2 = adjust_layout(fig2)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    if st.button("Fechar", key="close_history"):
        st.session_state.show_modal_enade = False
        st.rerun()


st.set_page_config(
    page_title="Gerador de Questões",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "groq" not in st.secrets or "key" not in st.secrets["groq"]:
    st.error("Configure o segredo groq.key no Streamlit Cloud.")
    st.stop()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["groq"]["key"],
)

st.markdown(
    """
    <style>
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

        div[data-testid="stMarkdownContainer"] {
            font-size: 24px !important;
        }

        div[data-testid="stSubHeader"] > label {
            margin-bottom: 0px !important;
        }

        div[data-testid="stHorizontalBlock"] {
            gap: 1.2rem !important;
        }

        div[data-testid="stColumn"]:has(> div > div > div > div[data-testid="stButton"]) {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }

        div[data-baseweb="slider"] {
            padding-left: 1em !important;
            width: 90% !important;
        }

        button[data-testid="stBaseButton-secondary"] {
            height: 4em !important;
            width: 12em !important;
        }

        section[data-testid="stFileUploaderDropzone"] {
            align-items: center !important;
        }

        div[data-testid="stFileUploaderDropzoneInstructions"] {
            margin-right: 0 !important;
        }

        h2 {
            text-align: center;
            font-weight: bold !important;
            margin-top: 2em !important;
        }

        .year-separation {
            font-size: 40px !important;
            line-height: 1.5em !important;
        }

        div[data-testid="stButton"] {
            display: flex !important;
            justify-content: center !important;
        }

        div[data-testid="stHorizontalBlock"]:has(> div > div > div > div > div > div > div[data-testid="stImage"]) {
            margin-bottom: 3em !important;
            padding-bottom: 2em !important;
            border-bottom: 2px dashed #aaa !important;
        }

        div[data-testid="stHorizontalBlock"] > div > div > div > div > div > div > div > div[data-testid="stImageContainer"] {
            height: 30em !important;
            width: 20em !important;
        }

        img {
            height: 100% !important;
            object-fit: contain !important;
        }

        div[data-testid="stElementToolbar"] {
            display: none !important;
        }

        div[data-testid="stElementContainer"] > div > div > div > div[data-testid="stImageContainer"] {
            height: 60em !important;
        }

        div[data-testid="stFullScreenFrame"] {
            display: flex !important;
            justify-content: center !important;
        }

        div[data-testid="stCaptionContainer"] {
            font-size: x-large !important;
        }

        div[data-testid="stDialog"] > div > div {
            width: 100em !important;
        }

        button[aria-label="Close"] {
            display: none !important;
        }

        div[data-testid="stDownloadButton"] {
            display: flex !important;
            justify-content: center !important;
        }

        div[data-baseweb="tab-list"] {
            display: flex !important;
            justify-content: space-around !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None
if "show_modal_question" not in st.session_state:
    st.session_state.show_modal_question = False
if "modal_content" not in st.session_state:
    st.session_state.modal_content = ""
if "editing_question" not in st.session_state:
    st.session_state.editing_question = False
if "modal_error" not in st.session_state:
    st.session_state.modal_error = None
if "show_modal_enade" not in st.session_state:
    st.session_state.show_modal_enade = False

edag_content_list = [
    "algoritmos e estrutura de dados",
    "arquitetura de computadores",
    "banco de dados",
    "cibersegurança",
    "ciência de dados",
    "elétrica e eletrônica",
    "engenharia de software",
    "grafos",
    "inteligência artificial",
    "iot",
    "lógica de programação",
    "processamento de sinais",
    "redes de computadores",
    "robótica, automação e controle",
    "sistemas digitais",
    "sistemas distribuídos e programação paralela",
    "sistemas embarcados",
    "sistemas operacionais e compiladores",
    "outros",
]

edag_topics_by_year = load_edag_topics()

st.markdown(
    """
    <div>
        <h1>Gerador de Questões</h1>
        <h4>Um Estudo de Caso para Engenharia de Computação</h4>
    </div>
    """,
    unsafe_allow_html=True,
    help="""Ferramenta designada para geração de questões utilizando modelos generativos e dentro de alguns parâmetros pré-estabelecidos: formato da questão, dificuldade, instruções adicionais opcionais através de prompt e suporte gráfico opcional através de imputação de imagem. A ferramenta também conta com um grid de questões de provas antigas do ENADE, as quais podem ser selecionadas para geração mais guiada de uma nova questão.""",
)

topics_display = st.multiselect(
    "Escolha um ou mais tópicos",
    [t.title() for t in edag_content_list],
    placeholder="Todos os tópicos",
    help="Seletor de tópicos para gerar uma nova questão e também filtrar as questões antigas do ENADE.",
)
topics = [t.lower() for t in topics_display]

fmt_col, gen_col, upload_col, btn_col = st.columns([1, 2, 1, 1])

display_formats, format_names, fmt_map = load_question_formats()

if not display_formats:
    st.error(f"Nenhum formato foi carregado em {FORMATS_DIR}")
    st.stop()

chosen_fmt = fmt_col.selectbox(
    "Formato da Nova Questão",
    display_formats,
    index=0,
    help="Seletor de formato da nova questão baseado nos direcionamentos de padrão do EDAG.",
)
fmt_filter = fmt_map[chosen_fmt]

difficulty = fmt_col.select_slider(
    "Nível de Dificuldade",
    ["Fácil", "Médio", "Difícil"],
    help="Seletor do nível de dificuldade da nova questão a ser gerada.",
)

user_prompt = gen_col.text_area(
    "Instruções Adicionais (opcional)",
    height=155,
    help="Área de texto para prompts adicionais.",
)

uploaded_graphic = upload_col.file_uploader(
    "Suporte Gráfico (opcional)",
    type=["png"],
    help="Upload de imagem PNG para servir de suporte gráfico.",
)

generate_clicked = btn_col.button("Gerar Questão")

if generate_clicked:
    msgs = []
    sys_content = (
        "Sua função é gerar uma nova questão de prova dentro dos [TÓPICOS] fornecidos, "
        "na [DIFICULDADE] fornecida e seguindo exatamente o [FORMATO DE SAÍDA] fornecido. "
        "Não adicione comentários, cabeçalhos, explicações, saudações ou qualquer texto extra. "
        "Retorne apenas o texto da nova questão, nada mais. Caso haja [INSTRUÇÕES ADICIONAIS], "
        "siga exatamente o que for pedido. Caso haja uma imagem [ANEXO GRÁFICO], use como suporte "
        "gráfico na geração da nova questão. Caso haja uma imagem [QUESTÃO BASE], faça uma nova "
        "versão da questão base, ainda seguindo o [FORMATO DE SAÍDA] fornecido."
    )
    msgs.append({"role": "system", "content": sys_content})

    selected_topics = topics if topics else edag_content_list
    format_template = load_file(FORMATS_DIR / fmt_filter)

    text_block = (
        f"\n\n[TÓPICOS]\n{selected_topics}"
        f"\n\n[DIFICULDADE]\n{difficulty}"
        f"\n\n[FORMATO DE SAÍDA]\n{format_template}"
    )

    if user_prompt:
        text_block += f"\n\n[INSTRUÇÕES ADICIONAIS]\n{user_prompt}"

    content_list = []

    if uploaded_graphic is not None:
        graphic_b64 = encode_image_fileobj(uploaded_graphic)
        content_list.append({"type": "text", "text": "\n\n[ANEXO GRÁFICO]\n"})
        content_list.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{graphic_b64}"}}
        )

    if st.session_state.selected_question:
        img_b64 = encode_image(st.session_state.selected_question["path"])
        content_list.append({"type": "text", "text": "\n\n[QUESTÃO BASE]\n"})
        content_list.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        )

    msgs.append({"role": "user", "content": [{"type": "text", "text": text_block}] + content_list})

    max_attempts = 3
    new_q = None
    candidate = None
    server_error = False

    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                messages=msgs,
                temperature=0.8,
                max_tokens=4096,
            )
        except Exception:
            server_error = True
            break

        candidate = resp.choices[0].message.content.strip()
        if validate_question_format(candidate, fmt_filter.split(".")[0]):
            new_q = candidate.replace("\n", "  \n")
            break

        if attempt == 0:
            msgs.append(
                {
                    "role": "user",
                    "content": "O formato da questão não seguiu exatamente o template. Gere novamente exatamente no formato fornecido.",
                }
            )
        time.sleep(5)

    if new_q:
        if uploaded_graphic is not None:
            new_q = new_q.replace(
                "  \n  \n",
                f"  \n  \n![Anexo Gráfico](data:image/png;base64,{graphic_b64})  \n  \n",
                1,
            )

        st.session_state.show_modal_question = True
        st.session_state.modal_content = new_q

    elif server_error:
        st.session_state.show_modal_question = True
        st.session_state.modal_content = ""
        st.session_state.modal_error = "Não consegui gerar a questão por problemas no servidor."

    else:
        candidate = (candidate or "").replace("\n", "  \n")
        if uploaded_graphic is not None:
            candidate = candidate.replace(
                "  \n  \n",
                f"  \n  \n![Anexo Gráfico](data:image/png;base64,{graphic_b64})  \n  \n",
                1,
            )

        st.session_state.show_modal_question = True
        st.session_state.modal_content = candidate
        st.session_state.modal_error = (
            f"Não consegui gerar a questão no formato correto após {max_attempts} tentativas, "
            "mas segue uma questão candidata."
        )

if st.session_state.show_modal_question:
    show_new_q()

if st.session_state.show_modal_enade:
    show_history()

raw_types = get_raw_types()
type_map = {"closed": "Fechada", "open": "Discursiva"}

if st.session_state.selected_question:
    st.markdown("")
    st.markdown("<h2>Questão Base Selecionada</h2>", unsafe_allow_html=True)
    st.markdown("")

    sq = st.session_state.selected_question

    if st.button("Voltar"):
        st.session_state.selected_question = None
        st.rerun()

    st.image(
        str(sq["path"]),
        caption=f"Questão {type_map.get(sq['type'], sq['type'].title())} {sq['number']:02d}",
        output_format="PNG",
    )

else:
    st.markdown("")
    st.markdown("<h2>Questões de Provas Antigas do ENADE</h2>", unsafe_allow_html=True)
    st.markdown("")

    cols = st.columns([2, 2, 1])

    years = get_years()

    with cols[0]:
        year_filter = st.selectbox(
            "Ano da Prova",
            ["Todos"] + [str(y) for y in years],
            help="Filtro do ano da prova antiga.",
        )

    display_types = ["Todos"] + [type_map.get(t, t.title()) for t in raw_types]
    with cols[1]:
        type_filter = st.selectbox(
            "Tipo de Questão",
            display_types,
            help="Filtro do tipo de questão.",
        )

    with cols[2]:
        if st.button(
            "Análise Histórica",
            key="btn_history",
            help="Abre painel interativo para análise histórica das provas do ENADE.",
        ):
            st.session_state.show_modal_enade = True
            st.rerun()

    all_qs = []
    for y in years:
        year_topics = edag_topics_by_year.get(y, {})
        year_dir = VISUAL_DIR / f"prova_{y}"

        if not year_dir.exists():
            continue

        for file_path in sorted(year_dir.iterdir()):
            if not file_path.is_file():
                continue

            fname = file_path.name
            qtype_raw, _, num_ext = fname.partition("_question_")
            if not num_ext:
                continue

            try:
                num = int(num_ext.split(".")[0])
            except ValueError:
                continue

            qkey = f"{qtype_raw}_question_{num:02d}"
            qtopics = year_topics.get(qkey, [])

            if topics and not set(qtopics).intersection(topics):
                continue

            if year_filter != "Todos" and str(y) != year_filter:
                continue

            disp_type = type_map.get(qtype_raw, qtype_raw.title())
            if type_filter != "Todos" and disp_type != type_filter:
                continue

            all_qs.append(
                {
                    "year": y,
                    "type": qtype_raw,
                    "number": num,
                    "path": file_path,
                }
            )

    grouped = {}
    for q in all_qs:
        grouped.setdefault(q["year"], []).append(q)

    for y in sorted(grouped.keys(), reverse=True):
        cols = st.columns([1, 14])
        with cols[0]:
            st.markdown("")
            st.markdown(f"<b class='year-separation'>{y}</b>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown("---")

        qs = grouped[y]
        for i in range(0, len(qs), 4):
            row = qs[i : i + 4]
            cols = st.columns(len(row))
            for col, q in zip(cols, row):
                with col:
                    st.image(str(q["path"]), output_format="PNG")
                    label = f"Questão {type_map.get(q['type'], q['type'].title())} {q['number']:02d}"
                    if st.button(label, key=f"select_{y}_{q['type']}_{q['number']}"):
                        st.session_state.selected_question = q
                        st.rerun()
