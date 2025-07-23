import time
import re
import os
import io
import base64

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
import fitz
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
from openai import OpenAI
from pdf2image import convert_from_bytes, convert_from_path
import pytesseract
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3Model
import matplotlib.pyplot as plt


# Normalizing text for when we want to compare strings
def normalize(text):
    return text.strip().lower()

# Funciton to click and reject cookies
def reject_cookies(page):
    try:
        page.wait_for_selector("button:has-text('Rejeitar cookies')", timeout=5000)
        page.click("button:has-text('Rejeitar cookies')")
    except:
        print("Aba de cookies não apareceu?\n")

# Function to try and handle failing get requests
def safe_get(url, retries=5, backoff_factor=1.2):
    for i in range(retries):
        try:
            return requests.get(url, timeout=10)
        except Exception as e:
            print(f"Falha no request {i+1} de {retries}: {e}\n")
            time.sleep(backoff_factor*(2**i))

# Function to load files
def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Function to load images
def load_image(path):
    return Image.open(path)

# Function to encode images so we can send them to the model
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Function to figure out if a text is garbage we did not manage to decode or proper text
def garbage_text(text, threshold=0.5):
    # Removing spaces and punctuation
    cleaned_text = re.sub(r'[\W_]+', '', text)
    
    # Check if the text is either empty or only symbols (garbage)
    if not cleaned_text:
        return True

    # Counting latin alphabet characters
    latin_chars = re.findall(r'[A-Za-zÀ-ÿ]', cleaned_text)
    ratio = len(latin_chars)/len(cleaned_text)

    return ratio<threshold

# Parsing PDF to extract ENADE theoretical content
def parse_pdf_theoretical(url, course, year):
    try:
        # Getting PDF
        response = safe_get(url)
        pdf_stream = io.BytesIO(response.content)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")

        # Finding all lines of the PDF and accumulating contents for Art. 6 or 7
        art_string = 'art. 6' if year > 2017 else 'art. 7'
        next_art_string = 'art. 7' if year > 2017 else 'art. 8'

        # Start looking for content only after matching our course (to parse even in Diários Oficiais)
        seen_course = False

        is_cont_art = False
        content_list = []
        for page in doc:
            # Fixing potential PDF broken lines (e.g. I-\nAdministração e Economia)
            raw_lines = page.get_text().split('\n')
            lines = []
            i = 0
            while i < len(raw_lines):
                current_line = raw_lines[i].strip()

                # Checking if line looks like "I-", "II -", etc. but doesn't have real content
                if re.match(r"^(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX|XXI)[\s\-–:.]*$", current_line, re.IGNORECASE):
                    if i+1 < len(raw_lines):
                        current_line += " " + raw_lines[i+1].strip()
                        i += 1

                lines.append(current_line)
                i += 1

            for line in lines:
                line = normalize(line.strip())
                
                # Wait until the course appears before parsing anything else
                if not seen_course:
                    if course in line:
                        seen_course = True
                    continue

                # Testing Art 6 or 7 start
                if not is_cont_art and line.startswith(art_string):
                    is_cont_art = True

                # Getting Art 6 or 7 content
                elif is_cont_art:
                    # Stop accumulating content
                    if line.startswith(next_art_string):
                        break

                    match = re.match(r"^(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX|XXI)[\s\-–:.]+(.+)", line, re.IGNORECASE)
                    if match:
                        content = match.group(2).strip()[:-1]
                        content_list.append(content)

        content_list.append('outros')
        return content_list

    except Exception as e:
        print(f"Não foi possível parsear o PDF {url}: {e}\n")
        return []

# Parsing HTML to extract ENADE theoretical content
def parse_html_theoretical(url, course, year):
    try:
        # Getting HTML
        response = safe_get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # Finding all lines of the HTML and accumulating contents for Art. 6 or 7
        paragraphs = soup.find_all("p", class_="dou-paragraph")

        art_string = 'art. 6' if year > 2017 else 'art. 7'
        next_art_string = 'art. 7' if year > 2017 else 'art. 8'

        # Start looking for content only after matching our course (to parse even in Diários Oficiais)
        seen_course = False

        is_cont_art = False
        content_list = []
        for p in paragraphs:
            text = normalize(p.get_text(strip=True))

            # Wait until the course appears before parsing anything else
            if not seen_course:
                if course in text:
                    seen_course = True
                continue

            # Testing Art. 6 or 7 start
            if not is_cont_art and text.startswith(art_string):
                is_cont_art = True

            # Getting Art. 6 or 7 content
            elif is_cont_art:
                # Stop acumulating content
                if text.startswith(next_art_string):
                    break

                match = re.match(r"^(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX|XXI)[\s\-–:.]+(.+)", text, re.IGNORECASE)
                if match:
                    content = match.group(2).strip()[:-1]
                    content_list.append(content)

        content_list.append('outros')
        return content_list

    except Exception as e:
        print(f"Não foi possível parsear o HTML {url}: {e}\n")
        return []

# Parsing PDF to extract ENADE test content
def parse_pdf_test_textual(url, year):
    try:
        # Getting PDF
        response = safe_get(url)
        pdf_stream = io.BytesIO(response.content)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")

        # Creating folder to store individual questions
        output_folder = f"../data/textual_approach/prova_{year}/raw"
        os.makedirs(output_folder, exist_ok=True)

        # Getting all text from all pages
        full_text = "\n".join([page.get_text() for page in doc])
        full_text = normalize(full_text)

        # Using OCR if we cannot properly decode PDF
        if garbage_text(full_text, threshold=0.5):
            images = convert_from_bytes(response.content)
            ocr_texts = []
            for img in images:
                text = pytesseract.image_to_string(img, lang="por")
                ocr_texts.append(text)
            full_text = "\n".join(ocr_texts)
            full_text = normalize(full_text)
            # Trimming document
            end_marker = "questionário de percepção da prova"
            end_index = full_text.rfind(end_marker)
            if end_index != -1:
                full_text = full_text[:end_index]

        # In case we don't need OCR, we need to get full text one page at a time to remove end_marker's page 
        else:
            end_marker = "questionário de percepção da prova"
            pages_before_marker = []
            for page in doc:
                raw = page.get_text()
                norm = normalize(raw)
                if end_marker in norm:
                    break
                pages_before_marker.append(raw)

            # Now joinning just those pages
            full_text = "\n".join(pages_before_marker)

        # Finding all question headers
        question_pattern = re.compile(r"(questão(?: discursiva)?\s+\d+)", re.IGNORECASE)

        matches = list(question_pattern.finditer(full_text))

        # Processing individual questions
        for i, match in enumerate(matches):
            question_title = match.group(1).lower()
            start_pos = match.end()

            # Defining end of current question
            end_pos = matches[i+1].start() if i+1 < len(matches) else len(full_text)

            # Extracting content
            question_text = full_text[start_pos:end_pos].strip()
            full_question = question_title + "\n" + question_text

            # Determining type (discursiva -> open or múltipla-escolha -> closed)
            is_open = "discursiva" in question_title
            q_type = "open" if is_open else "closed"
            q_number = re.findall(r"\d+", question_title)[0]

            # Saving to file
            filename = f"{q_type}_question_{q_number}.txt"
            filepath = os.path.join(output_folder, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(full_question)

        print(f"Extração das questões de prova de {year} completa: {len(matches)} questões salvas\n")

    except Exception as e:
        print(f"Não foi possível parsear o PDF {url}: {e}\n")

# Function to test if any other question has similar height to the first question
def test_heights(indices, boxes):
    if len(indices) > 1:
        for idx in indices[1:]:
            if torch.abs(boxes[indices[0], 1]-boxes[idx, 1]) < 5:
                return True
    return False

# Function to extract questions from PDF
def deep_learning_ocr(pages, processor, device, visual=-1):
    # Running LayoutLMv3 OCR
    tokens = []
    boxes = []
    final_pages = []
    for i, page in enumerate(pages, start=1):
        # Running model
        enc = processor(page, return_tensors="pt", truncation=True, \
        max_length=2048).to(device)
        
        # Extracting tokens and boundig boxes
        page_tokens = [token.replace('Ġ', '') for token in enc.tokens()]
        page_boxes  = enc.bbox.squeeze(0)
        
         # Stopping test reading if "Questionário de Percepção da Prova" is reached
        indices = [j for j, token in enumerate(page_tokens) if token == 'QUEST']
        if len(indices) >= 1 and page_tokens[indices[0]+1] == 'ION':
            return final_pages, tokens, boxes

        # Splitting page in case it's a vertically split page        
        if len(indices) >= 2 and test_heights(indices, page_boxes):
            w, h  = page.size
            mid_x = w//2
            halves = [page.crop((0, 0, mid_x, h)), page.crop((mid_x, 0, w, h))]

            # Rerunning model now on 2 pages
            for half in halves:
                sub_enc = processor(half, return_tensors="pt", truncation=True,\
                                    max_length=1024).to(device)
                sub_tokens = [token.replace('Ġ','') for token in sub_enc.tokens()]
                sub_boxes  = sub_enc.bbox.squeeze(0)

                tokens.append(sub_tokens)
                boxes.append(sub_boxes)
                final_pages.append(half)

            continue

        tokens.append(page_tokens)
        boxes.append(page_boxes)
        final_pages.append(page)
        
        # Drawing bounding boxes on questions
        if visual != -1 and i == visual:
            orig_w, orig_h = page.size
            draw = ImageDraw.Draw(page)
            for idx in range(len(page_tokens)):
                x0, y0, x1, y1 = page_boxes[idx]
                draw.rectangle([(orig_w*x0/1000, orig_h*y0/1000), \
                                (orig_w*x1/1000, orig_h*y1/1000)], \
                               outline="red", width=2)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(page)

            return final_pages, tokens, boxes
    
    return final_pages, tokens, boxes

# Parsing PDF to extract ENADE test content
def parse_pdf_test_visual(url, year, dpi=300, visual=-1):
    try:
        # Getting PDF as PIL images 
        response = safe_get(url)
        pages = convert_from_bytes(response.content, dpi=dpi)
        pages = pages[1:-2]

        # Creating folder to store individual questions
        output_folder = f"../data/visual_approach/prova_{year}"
        os.makedirs(output_folder, exist_ok=True)

        # Initializing processor and device
        processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-large", apply_ocr=True, ocr_lang="por")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Extracting questions
        pages, tokens, boxes = deep_learning_ocr(pages, processor, device, visual=visual)

        # Saving screenshot of question for later usage on an LLM
        page_overflow = False
        prev = {'left': None, "right": None, "discursiva": None}
        open_counter = 1
        closed_counter = 1
        for i, t in enumerate(tokens):
            # Getting indices of questions on a page
            indices = [j for j, token in enumerate(t) if token == 'QUEST']

            # Saving continuation of a question on another second page
            if page_overflow is True:
                try:
                    bot_limit = boxes[i][indices[0], 1]
                except:
                    bot_limit = torch.max(boxes[i][:-1, 1])
                
                img = pages[i]
                orig_w, orig_h = img.size
                left = (prev["left"].item()/1000)*orig_w - 5
                right = (prev["right"].item()/1000)* orig_w + 5
                top = 5
                bottom = (bot_limit.item()/1000)*orig_h + 5

                crop = img.crop((left, top, right, bottom))
                if discursiva:
                    fname = f"../data/visual_approach/prova_{year}/open_question_{open_counter-1:02d}_p2.png"
                else:
                    fname = f"../data/visual_approach/prova_{year}/closed_question_{closed_counter-1:02d}_p2.png"
                crop.save(fname)
                
                page_overflow = False

            # Going through questions on a page
            for j, idx in enumerate(indices):
                # Checking if its closed or open
                discursiva = (t[idx+3] == 'DIS')

                # Computing limits for cropping
                left_limit, right_limit = boxes[i][idx, 0], torch.max(boxes[i][:-1, 2])
                top_limit = boxes[i][idx, 1]
                
                # Two questions on the same page
                if j < len(indices)-1:
                    bot_limit = boxes[i][indices[j+1], 1]
                
                # A question that either finishes the page or goes to another page
                else:
                    page_overflow = True
                    bot_limit = boxes[i][-10, 3]

                # Saving question
                img = pages[i]
                orig_w, orig_h = img.size
                left = (left_limit.item()/1000)*orig_w - 5
                right = (right_limit.item()/1000)* orig_w + 5
                top = (top_limit.item()/1000)*orig_h - 5
                bottom = (bot_limit.item()/1000)*orig_h + 5

                crop = img.crop((left, top, right, bottom))
                if discursiva:
                    fname = f"../data/visual_approach/prova_{year}/open_question_{open_counter:02d}_p1.png"
                    open_counter += 1
                else:
                    fname = f"../data/visual_approach/prova_{year}/closed_question_{closed_counter:02d}_p1.png"
                    closed_counter += 1
                crop.save(fname)

                # Updating previous dictionary if needed
                if page_overflow:
                    prev.update({"left": left_limit, "right": right_limit, \
                                "discursiva": discursiva})

        # "Glueing" multiple page questions
        folder = f"../data/visual_approach/prova_{year}"
        open_pattern = re.compile(r"open_question_(\d{2})_p[12]\.png$")
        closed_pattern = re.compile(r"closed_question_(\d{2})_p[12]\.png$")

        # Gathering all files and group by question number
        groups = {}
        for fname in os.listdir(folder):
            open_m = open_pattern.match(fname)
            closed_m = closed_pattern.match(fname)
            
            if open_m:
                qnum = open_m.group(1)
                try:
                    groups[f'open_{qnum}'].append(fname)
                except:
                    groups[f'open_{qnum}'] = [fname]

            elif closed_m:
                qnum = closed_m.group(1)
                try:
                    groups[f'closed_{qnum}'].append(fname)
                except:
                    groups[f'closed_{qnum}'] = [fname]

        # Stacking images vertically when question has more than one part
        for qnum, fnames in groups.items():
            if len(fnames) < 2:
                path = os.path.join(folder, fnames[0])
                os.rename(path, os.path.join(folder, f"{fnames[0].split('_p')[0]}.png"))
                continue

            # Organizing order to stack correctly
            fnames_sorted = sorted(fnames)

            # Computing dimensions for the new canvas
            imgs = [Image.open(os.path.join(folder, f)) for f in fnames_sorted]
            widths, heights = zip(*(im.size for im in imgs))
            max_width = max(widths)
            total_height = sum(heights)

            # Creating a blank canvas and pasting each part
            combined = Image.new("RGB", (max_width, total_height), (255, 255, 255))
            y_offset = 0
            for im in imgs:
                combined.paste(im, (0, y_offset))
                y_offset += im.height

            # Saving the stacked image
            combined.save(os.path.join(folder, f"{fnames[0].split('_p')[0]}.png"))

            # Deleting the old files with parts
            for f in fnames:
                os.remove(os.path.join(folder, f))

    except Exception as e:
        print(f"Não foi possível parsear o PDF {url}: {e}\n")

# Function to extract the test content by topics, both on ENADE's format and on CIMATEC's formart
def extract_test_content(df, year):
    # Hard coded list of content from CIMATEC's perspective
    edag_content_list = ['algoritmos e estrutura de dados', 'arquitetura de computadores', 'banco de dados', 'cibersegurança', 'ciência de dados', 'elétrica e eletrônica', 'engenharia de software', 'grafos', 'inteligência artificial', 'iot', 'lógica de programação', 'processamento de sinais', 'redes de computadores', 'robótica, automação e controle', 'sistemas digitais', 'sistemas distribuídos e programação paralela', 'sistemas embarcados', 'sistemas operacionais e compiladores', 'outros']

    # Getting enade's content list
    idx = None
    for i, y in enumerate(df['year']):
        if int(y) == year:
            idx = i
            break
    enade_content_list = df['theoretical_content'][idx]

    # Initializing API client
    groq_key = load_file('../data/keys/groq').strip()
    os.environ['OPENAI_API_KEY'] = groq_key
    client = OpenAI(
        base_url='https://api.groq.com/openai/v1',
        api_key=os.environ['OPENAI_API_KEY'],
    )

    # Iterating through images and extracting content
    enade_dictionary = {}
    edag_dictionary = {}
    for question in os.listdir(f'../data/visual_approach/prova_{year}'):
        base64_image = encode_image(f'../data/visual_approach/prova_{year}/{question}')
        try_counter = 0
        enade_check = False
        edag_check = False
        while True:
            try:
                if not enade_check:
                    response_enade = client.chat.completions.create(
                        model='meta-llama/llama-4-maverick-17b-128e-instruct',
                        messages=[
                            {
                                'role': 'system',
                                'content': "Sua função é analisar a questão fornecida e determinar a quais categorias ela pertence dada uma lista de conteúdos. Tente achar o menor número de categorias possível por questão, ou seja categorias diretas da questão, não tangenciais. Retorne apenas as categorias, nada mais. Para as questões que não apresentam conteúdo técnico, atribua exclusivamente a categoria 'outros'. As diferentes categorias devem ser separadas por '; '.",
                            },
                            {
                                'role': 'user',
                                'content': [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{base64_image}",
                                        },
                                    },
                                    {
                                        'type': 'text',
                                        'text': f"""Lista de conteúdos: {enade_content_list}""",
                                    }
                                ]
                            },
                        ],
                        temperature=1.2,
                        max_tokens=4096,
                    )
                    enade_content = response_enade.choices[0].message.content.strip().split('; ')

                if not edag_check:
                    response_edag = client.chat.completions.create(
                        model='meta-llama/llama-4-maverick-17b-128e-instruct',
                        messages=[
                            {
                                'role': 'system',
                                'content': "Sua função é analisar a questão fornecida e determinar a quais categorias ela pertence dada uma lista de conteúdos. Tente achar o menor número de categorias possível por questão, ou seja categorias diretas da questão, não tangenciais. Retorne apenas as categorias, nada mais. Para as questões que não apresentam conteúdo técnico, atribua exclusivamente a categoria 'outros'. As diferentes categorias devem ser separadas por '; '.",
                            },
                            {
                                'role': 'user',
                                'content': [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{base64_image}",
                                        },
                                    },
                                    {
                                        'type': 'text',
                                        'text': f"""Lista de conteúdos: {edag_content_list}""",
                                    }
                                ]
                            },
                        ],
                        temperature=1.2,
                        max_tokens=4096,
                    )
                    edag_content = response_edag.choices[0].message.content.strip().split('; ')

                # Checking if the contents are actually part of their proper lists
                enade_check = True
                for content in enade_content:
                    if content not in enade_content_list:
                        enade_check = False
                        break

                edag_check = True
                for content in edag_content:
                    if content not in edag_content_list:
                        edag_check = False
                        break

                if not enade_check or not edag_check:
                    try_counter += 1
                    if try_counter >= 3:
                        try_counter = 0
                        print(f"Por favor, revise os conteúdos da questão {question.split('.')[0]}")
                        print(question.split('.')[0])
                        print(enade_content)
                        print(edag_content)
                        print()

                        time.sleep(5)
                        break
                    
                    time.sleep(5)
                    continue

                print(question.split('.')[0])
                print(enade_content)
                print(edag_content)
                print()
                time.sleep(5)
                break
            
            except Exception as e:
                print(f'Problema no servidor... {e}')
                time.sleep(15)
        
        enade_dictionary[question.split('.')[0]] = enade_content
        edag_dictionary[question.split('.')[0]] = edag_content

    return enade_dictionary, edag_dictionary
    
# Function to parse and extract content. Basically a wrapper of the other functions
def parse_and_extract(page, df, target_courses, extraction_type='edital'):
    # Getting all year tabs and filtering everything before 2014
    year_elements = page.locator(".govbr-tabs a").all()
    for year_element in year_elements:
        year = year_element.inner_text().strip()
        if not year.isdigit() or int(year) < 2014:
            continue

        # Simulating year click to extract information (and letting content load)
        year_element.click()
        time.sleep(1.5)

        # Selecting currently visible content block
        active_tab = page.locator(".tab-content.active")

        if extraction_type == 'edital':
            # Getting all <a> elements within it
            links = active_tab.locator("a").all()

            # Finding the target courses
            for link in links:
                link_text = normalize(link.inner_text())
                href = link.get_attribute("href")

                for course in list(target_courses.keys()):
                    if any(tc in link_text for tc in target_courses[course]):
                        print(f"Edital do curso {link_text} encontrado em {year}\n")
                        
                        if href.endswith(".pdf"):
                            theoretical_content = parse_pdf_theoretical(href, course, int(year))
                        else:
                            theoretical_content = parse_html_theoretical(href, course, int(year))
                        
                        # Populating dataframe
                        df['course'].append(course)
                        df['year'].append(year)
                        df['theoretical_content'].append(theoretical_content)

        elif extraction_type == 'prova':
            # Getting all <p> elements with class 'callout' inside the active tab
            callouts = active_tab.locator("p.callout").all()

            # Finding the target courses
            for callout in callouts:
                callout_text = normalize(callout.inner_text())

                for course in list(target_courses.keys()):
                    if any(tc in callout_text for tc in target_courses[course]):
                        # Look for the next sibling <ul>
                        ul_element = callout.locator("xpath=following-sibling::ul").first

                        # Now find <a> tags inside <li> elements under that <ul>
                        prova_links = ul_element.locator("li a").all()

                        # Now find the 'prova' link
                        for link in prova_links:
                            link_text = normalize(link.inner_text())
                            if link_text == 'prova':
                                href = link.get_attribute("href")
                                print(f"Prova do curso '{course}' encontrada em {year}\n")

                                parse_pdf_test_visual(href, int(year))

                                # After parsing the test, we procees to extract its content
                                content_enade, content_edag = extract_test_content(df, int(year))
                                df['test_content_enade'].append(content_enade)
                                df['test_content_edag'].append(content_edag)

# Starting interaction with browser
with sync_playwright() as p:
    # Choosing what courses to target
    target_courses = {'engenharia de computação': ["engenharia da computação", "engenharia de computação"]}

    # Initializing dataframe
    df = {'course': [], 'year': [], 'theoretical_content': [], 'test_content_enade': [], 'test_content_edag': []}

    # Launchung browser and navigating to desired URL
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # Navigating to edital page
    url = "https://www.gov.br/inep/pt-br/centrais-de-conteudo/legislacao/enade"
    page.goto(url)
    reject_cookies(page)

    # Waiting for the the carousel containing years to load
    page.wait_for_selector(".govbr-tabs")

    # Parsing and extracting content from edital page
    parse_and_extract(page, df, target_courses, extraction_type='edital')

    # Repeating the process for the prova page
    url = "https://www.gov.br/inep/pt-br/areas-de-atuacao/avaliacao-e-exames-educacionais/enade/provas-e-gabaritos"
    page.goto(url)
    page.wait_for_selector(".govbr-tabs")
    parse_and_extract(page, df, target_courses, extraction_type='prova')

    browser.close()

    # Creating and saving dataframe
    df = pd.DataFrame(df)
    df.to_csv("../data/enade_data.csv", index=False)