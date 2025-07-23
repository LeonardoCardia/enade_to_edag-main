import os
import time
from openai import OpenAI

# Function to load files
def load_file(file_path):
  with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
  return content

# Connecting to groq
os.environ["OPENAI_API_KEY"] = load_file('../data/keys/groq').strip()
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["OPENAI_API_KEY"],
)

# Reading questions
root_dir = '../data/'
for folder in os.listdir(root_dir):
    folder = os.path.join(root_dir, folder)

    if os.path.isdir(folder) and 'prova' in folder:
        raw_folder = os.path.join(folder, 'raw')
        clean_folder = os.path.join(folder, 'clean')
        os.makedirs(clean_folder, exist_ok=True)

        for questao in os.listdir(raw_folder):
            raw_questao = os.path.join(raw_folder, questao)
            clean_questao = os.path.join(clean_folder, questao)
            original_question = load_file(raw_questao)

            # Making request to effectively clean questions
            response = client.chat.completions.create(
                #model="meta-llama/llama-3.3-70b-versatile",
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um assistente que limpa questões do ENADE. Remova apenas ruídos como textos inválidos, palavras malformadas ou trechos repetidos. Mantenha absolutamente inalterados o formato, a estrutura e o conteúdo da questão. NÃO responda à pergunta. NÃO adicione comentários ou introduções. Retorne SOMENTE a questão corrigida, limpa e nada mais.",
                    },
                    {
                        "role": "user",
                        "content": f"{original_question}"
                    },
                ],
                temperature=0.3,
                max_tokens=1024,
            )

            # Saving question to file
            with open(clean_questao, "w", encoding="utf-8") as f:
                f.write(response.choices[0].message.content)

            # Avoiding requesting more than the free tier allows
            time.sleep(2.5)

        print(f'Limpeza feita para {folder.split('/')[-1]}\n')