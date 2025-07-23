import os
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

# Reading question and format
test_year = 2023
question_type = 'closed'
test_question = 1
original_question = load_file(f'../data/prova_{test_year}/clean/{question_type}_question_{test_question:02d}.txt')

question_format = load_file('../data/edag_question_formats/resposta_unica.txt')

# Making request
response = client.chat.completions.create(
    #model="meta-llama/llama-3.3-70b-versatile",
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "system",
            "content": "Você é um assistente cuja única função é gerar uma nova versão de uma questão do ENADE exatamente no formato de saída fornecido. Não responda ao conteúdo da questão original. Não adicione comentários, cabeçalhos, explicações, saudações ou qualquer texto extra (como “Aqui está” ou “Nova questão:”). Mantenha estritamente a estrutura, numeração, pontuação e estilo do formato de saída. Retorne apenas o texto da nova questão, nada mais.",
        },
        {
            "role": "user",
            "content": f"[QUESTÃO ORIGINAL]\n{original_question}\n\n[FORMATO DE SAÍDA]\n{question_format}"
        },
    ],
    temperature=0.3,
    max_tokens=1024,
)

print('Nova Questão Gerada:\n')
print(response.choices[0].message.content)
print()