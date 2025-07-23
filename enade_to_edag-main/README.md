# Análise e Geração de Questões para o EDAG

Este projeto tem como objetivo geração de questões para a prova do EDAG (Exame de Desempenho Acadêmico Geral do Senai CIMATEC), com estudo de caso para o curso de Engenharia da Computação. Também mineramos e analisamos o conteúdo dos editais e das provas do ENADE (Exame Nacional de Desempenho dos Estudantes) com a intenção de possibilitar a geração de questões inspiradas no exame nacional.

<p align="center">
  <img src="assets/usando_app.gif" alt="Visão geral do uso do aplicativo." width="60%" style="margin-top: 20px;" />
</p>
<p align="center" style="margin-bottom: 15px;">
  Animação de uso do aplicativo para uma visão geral de interatividade e das ferramentas existentes no mesmo.
</p>


Se quiser entender mais sobre o desenvolvimento das pipelines, leia a [documentação anexa](https://github.com/Luizerko/enade_to_edag/tree/main/docs/MINERACAO_GERACAO_ANALISE.md).

## Etapas do Projeto

1. **Mineração de Dados**
   - Raspagem dos editais do ENADE (portarias oficiais) para extração do conteúdo teórico da prova.
   - Raspagem das provas aplicadas do ENADE por ano para extração e categorização automática das questões com base no conteúdo programático.

2. **Geração de Questões com IA**
   - Uso de LLMs para sugerir novas questões para o EDAG, utilizando sua formatação específica e com possibilidade de basear-se em questões do ENADE.

3. **Dashboard Interativo**
   - Visualizações da evolução dos conteúdos e questões do ENADE ao longo dos anos.
   - Comparações quantitativas por áreas temáticas.

## Status Atual

- Raspagem e extração dos conteúdos teóricos dos editais **concluída**.
- Raspagem de questões das provas **concluída**.
- Classificação das questões das provas nas áreas temáticas **concluída**.
- Geração automática de novas questões com LLMs **concluída**.
- Desenvolvimento do dashboard interativo para conteúdo **concluída**.

## Hierarquia do Repositório
 
    .
    ├── assets                     # Coleção de imagens e GIFs utilizados para a documentação
    ├── data                       # Coleção de questões, formatos e o CSV com metadados de provas antigas
    |   ├── edag_question_formats  # Coleção de formatos das questões do EDAG
    |   ├── textual_approach       # Questões antigas do ENADE extraídas com método de processamento textual
    |   ├── visual_approach        # Questões antigas do ENADE extraídas com método de processamento visual
    |   ├── enade_data.csv         # CSV com metadados de provas antigas
    ├── docs                       # Arquivos de documentação
    ├── notebooks                  # Coleção ded notebooks IPython usados para testar código
    ├── py_scripts                 # Coleção de scripts Python utilizados para testar código e extrair e processar dados  
    ├── app.py                     # Código Python do aplicativo
    └── ...

---

Projeto desenvolvido por **Luis Vitor Zerkowski** com inspiração nas ideias de **Sanval Ebert**.