import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Configuração da chave da API OpenAI e do modelo LLM
llm_name = 'gpt-4o-mini'
model = ChatOpenAI(api_key="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", model=llm_name)

# Carregar arquivo CSV e tratar valores ausentes
cols_to_use = ['fx_etaria', 'SRAG', 'SARS2', 'VSR', 'FLU', 'RINO', 'ADNO', 'BOCA', 'METAP', 'OUTROS', 'DS_UF_SIGLA', 'epiyear', 'epiweek']
df = pd.read_csv('https://gitlab.fiocruz.br/marcelo.gomes/infogripe/-/raw/master/Dados/InfoGripe/casos_semanais_fx_etaria_virus_sem_filtro_febre.csv', sep = ';', usecols=cols_to_use).fillna(value=0)

# Criar agente pandas para manipulação do DataFrame
agent = create_pandas_dataframe_agent(llm=model, df=df, verbose=True, allow_dangerous_code=True)

# Definir prefixo e sufixo do prompt para o agente
PROMPT_PREFIX = """
Você é um cientista de dados prestativo e suas respostas são concisas.

Você será fornecido um banco de dados em formato CSV de casos de Síndrome Respiratória Aguda Grave ('SRAG') que ocorreram no Brasil, em diferentes faixas etárias. Cada linha do banco de dados representa os casos de SRAG que ocorreram em uma determinada faixa etária (coluna 'fx_etaria'), em uma determinada Unidade Federativa ou no Brasil (coluna 'DS_UF_SIGLA'), em uma determinada semana epidemiológica (coluna 'epiweek'), em um determinado ano (coluna 'epiyear').

    O banco de dados possui as seguintes colunas:
    - 'SG_UF_NOT': ignore essa variável.
    - 'fx_etaria': A faixa etária, em anos, em que ocorreram os casos de SRAG. As categorias são: '< 2' para menores de 2 anos de idade; '2 a 4' para crianças de 2 a 4 anos de idade; '5 a 14' para pessoas de 5 a 14 anos de idade; '15 a 49' para pessoas de 15 a 49 anos, '50 a 64' para pessoas de 50 a 64 anos de idade; '65+' para pessoas com 65 ou mais anos de idade.
    - 'SRAG': O número de casos total de Síndrome Respiratória Aguda Grave (SRAG).
    - 'SARS2': O número de casos causados por SARS-CoV-2 (Covid 19).
    - 'VSR': O número de casos causados por Vírus Sincicial Respiratório.
    - 'FLU': O número total de casos causados por todos os vírus Influenza
    - 'RINO': O número de casos causados por Rinovírus.
    - 'ADNO': O número de casos causados por Adenovírus.
    - 'BOCA': O número de casos causados por Bocavírus.
    - 'METAP': O número de casos causados por Metapneumovírus.
    - 'OUTROS': O número de casos causados por outros vírus respiratórios.
    - 'DS_UF_SIGLA': Sigla das Unidades Federativas do Brasil, onde a sigla BR significa o somatório total para o Brasil inteiro.
    - 'epiyear': ano epidemiológico.
    - 'epiweek': semana epidemiológica.

Primeiro, remova todas as linhas onde fx_etaria seja 'Total' e remova todas as linhas onde DS_UF_SIGLA seja 'BR'. Ajuste as configurações de exibição do pandas para mostrar todas as colunas e as primeiras 6 linhas. Recupere os nomes das colunas e, em seguida, prossiga para responder à pergunta com base nos dados fornecidos.
"""

PROMPT_SUFFIX = """
- Antes de fornecer a resposta final sempre tente pelo menos um método adicional.
- Inicie a análise identificando se o usuário indicou alguma das faixas etárias corretas como constam no banco: '< 2' para menores de 2 anos de idade; '2 a 4' para crianças de 2 a 4 anos de idade; '5 a 14' para pessoas de 5 a 14 anos de idade; '15 a 49' para pessoas de 15 a 49 anos, '50 a 64' para pessoas de 50 a 64 anos de idade; '65+' para pessoas com 65 ou mais anos de idade.
- Se o usuário tiver pedido uma faixa etária que não seja um intervalo que conste do banco, interrompa a análise e diga quais são as faixas etárias possíveis.
- Reflita sobre os métodos e assegure-se de que os resultados respondem à pergunta original com precisão.
- Retorne sempre números, a menos que o usuário peça explicitamente um gráfico.
- Se o usuário pedir um gráfico, utilize a biblioteca seaborn.
- Formate qualquer número com quatro ou mais dígitos utilizando pontos para facilitar a leitura.
- Formate qualquer número real com casas decimais utilizando vírgula e arredondando para 3 casas decimais para facilitar a leitura.
- Se os resultados dos métodos forem diferentes, reflita e tente outra abordagem até que ambos os métodos se alinhem.
- Se ainda não conseguir chegar a um resultado consistente, reconheça a incerteza em sua resposta.
- Quando tiver confiança na resposta correta, elabore uma explicação detalhada e bem estruturada utilizando markdown.
- Sob nenhuma circunstância deve-se usar conhecimento prévio — confie unicamente nos resultados derivados dos dados e cálculos realizados.
- Como parte de sua resposta final, inclua uma seção de **Explicação** que descreva claramente como a resposta foi alcançada, mencionando nomes de colunas específicos utilizados nos cálculos.
- Como parte de sua resposta final, inclua o resultado dos cálculos e/ou filtragens realizadas.
"""

# Exemplo de pergunta ao agente
QUESTION = "Quantos casos de gripe ocorreram no Mato Grosso do Sul em 2024?"
QUERY = PROMPT_PREFIX + QUESTION + PROMPT_SUFFIX

# Consulta e execução da resposta do agente
res = agent.invoke(QUERY)

# Aplicação Streamlit
st.title("Agente de IA do InfoGripe")
st.write("### Preview do banco")
st.write(df.head())

# Entrada de pergunta pelo usuário
st.write('### Faça uma pergunta relacionada a SRAG no Brasil')
question = st.text_input(
    "O que você deseja saber sobre o banco de dados:",
    "Quantos casos de gripe ocorreram no Mato Grosso do Sul em 2024?"
)

# Ação ao clicar no botão
if st.button("Run Query"):
    QUERY = PROMPT_PREFIX + question + PROMPT_SUFFIX
    res = agent.invoke(QUERY)
    st.write("### Resposta")
    st.markdown(res["output"])

# streamlit run app.py