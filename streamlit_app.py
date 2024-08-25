import os
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta, date
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults


def fetch_stock_price(ticker, weeks_ago=52):
    today = date.today()
    date_weeks_ago = today - timedelta(weeks=weeks_ago)

    stock_data = yf.download(ticker, start=date_weeks_ago, end=today)

    return stock_data


yahoo_finance_tool = Tool(
    name="Yahoo Finance Tool",
    description="Obtenha os preços da ação com o ticker {ticker} de um intervalo de dados do Yahoo Finance",
    func=lambda ticker: fetch_stock_price(ticker),
)

llm = ChatOpenAI(model="gpt-3.5-turbo")

stock_price_analyst = Agent(
    role="Analista sênior de preços de ações",
    goal="Encontre o preço da ação {ticker} e faça uma análise de tendência",
    backstory="Você tem muita experiência em analisar o preço de uma ação específica e fazer previsões sobre seu preço futuro.",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    tools=[yahoo_finance_tool],
    allow_delegation=False,
)

get_stock_price_task = Task(
    description="Analisar o histórico de preços da ação {ticker} e crie uma análise de tendências de alta, baixa ou para os lados",
    agent=stock_price_analyst,
    expected_output="""Especifique a tendência atual do preço das ações - para cima, para baixo ou para os lados.
    por exemplo, ação = 'APPL, price UP'""",
)

search_tool = DuckDuckGoSearchResults(
    backend="news",
    num_results=20,
)

news_analyst_agent = Agent(
    role="Analista de notícias de ações",
    goal="""Crie um breve resumo das notícias do mercado relacionadas as ações da empresa com o ticker {ticker}. Especifique a tendência atual - para cima, para baixo ou para os lados com
    o contexto das notícias. Para cada ativo de ação solicitado, especifique um número entre 0 e 100, onde 0 é medo extremo e 100 é ganância extrema.""",
    backstory="""Você é altamente experiente em analisar tendências de mercado e notícias e acompanha ativos há mais de 10 anos.

    Você também é um analista de nível senior nos mercados tradicionais e tem profundo conhecimento da psicologia humana.

    Você entende notícias, seus títulos e informações, mas olha para elas com uma dose saudável de ceticismo.

    Você também considera a fonte dos artigos de notícias.
    """,
    verbose=True,
    llm=llm,
    max_iter=100,
    memory=True,
    tools=[search_tool],
)

get_news_task = Task(
    agent=news_analyst_agent,
    description=f"""Pegue os dados da ação.
    Use a ferramenta de busca para pesquisar sobre ela.

    A data atual é {datetime.now()}.

    Componha os resultados em um relatório útil""",
    expected_output=""""Um resumo do mercado geral e um resumo de uma frase para cada ativo solicitado.
    Inclua uma pontuação de medo/ganância para cada ativo com base nas notícias.

    Use o formato:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>""",
    allow_delegation=False,
)

stock_analyst_write_agent = Agent(
    role="Escritor analista sênior de ações",
    goal="""Análise as tendências de preços e notícias e escreva uma análise
    informativa, envolvente e perspicaz, de até 5 parágrafos, com base no
    relatório da ação e na tendência de preços.""",
    backstory="""Você é amplamente aceito como o melhor analista de ações do
    mercado. Você entende conceitos complexos e cria histórias e narrativas
    atraentes que ressoam com públicos mais amplos.

    Você entende fatores macro e combina múltiplas teorias - por exemplo, teoria
    do ciclo e análise fundamentalista.

    Você é capaz de sustentar múltiplas opiniões ao analisar qualquer coisa.""",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=True,
)

write_analyses_task = Task(
    description="""Use a tendência do preço das ações e o relatório de notícias
    sobre ações para criar uma análise e escrever um resumo sobre a empresa com
    o ticker {ticker} que seja breve e destaque os pontos mais importantes.

    Concentre-se na tendência do preço das ações, notícias e pontuação de
    medo/ganância.

    Quais são as considerações para o futuro próximo?

    Inclua as análises anteriores da tendência das ações e o resumo das notícias.
    """,
    expected_output=""""Um boletim análise de até 5 parágrafos formatado como
    markdown de forma fácil de ler. Deve conter:

    - resumo de até 8 marcadores de lista
    - introdução - defina o quadro geral e aumente o interesse
    - a parte principal fornece o cerne da análise, incluindo o resumo das notícias e as pontuações de feed/greed
    - resumo - fatos-chave e previsão concreta de tendências futuras - para cima, para baixo ou para os lados.
    - importante informar todas as fontes consultadas para realizar a análise

    A resposta deve estar em português (brasileiro).
    """,
    agent=stock_analyst_write_agent,
    context=[get_stock_price_task, get_news_task],
)

crew = Crew(
    agents=[stock_price_analyst, news_analyst_agent, stock_analyst_write_agent],
    tasks=[get_stock_price_task, get_news_task, write_analyses_task],
    verbose=2,
    process=Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15,
)

with st.sidebar:
    st.header("Informe o ticker do ativo a ser analisado")

    with st.form(key="research_form"):
        openai_api_key = st.text_input(
            "API key da OpenAI",
            help="Crie suas chave em https://platform.openai.com/api-keys",
            placeholder="sk-proj-***",
        )
        ticker = st.text_input(
            label="Ticker do ativo",
            help='Para ações brasileiras adicione o sufixo ".SA". Exemplo: WEGE3.SA',
        )
        submit_button = st.form_submit_button(label="Realizar análise")

if submit_button:
    if not ticker:
        st.error("Informe o ticker do ativo a ser analisado")
    elif not openai_api_key:
        st.error("Informe a chave de API da OpenAI")
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        results = crew.kickoff(inputs={"ticker": ticker})

        st.subheader("Resultado do análise")
        st.write(results["final_output"])
