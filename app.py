import os
import yfinance as yf
import streamlit as st
# from dotenv import load_dotenv
from datetime import datetime, timedelta, date
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

# load_dotenv()

print()

def fetch_stock_price(ticker, weeks_ago = 52):
  today = date.today()
  date_weeks_ago = today - timedelta(weeks = weeks_ago)

  stock_data = yf.download(ticker, start = date_weeks_ago, end = today)

  return stock_data

yahoo_finance_tool = Tool(
  name = 'Yahoo Finance Tool',
  description = 'Fetch stock prices for {ticker} from a data range from Yahoo Finance',
  func = lambda ticker: fetch_stock_price(ticker)
)

os.environ['OPENAI_API_KEY'] = st.secrets('OPENAI_API_KEY')

llm = ChatOpenAI(model = "gpt-3.5-turbo")

stockPriceAnalyst = Agent(
  role = "Senior stock price analyst",
  goal = "Find the {ticker} stock price and analysis trends",
  backstory = "You're highly experienced in analyzing the price of an specific stock and make predictions about its future price.",
  verbose = True,
  llm = llm,
  max_iter = 5,
  memory = True,
  tools = [yahoo_finance_tool]
)

get_stock_price_task = Task(
  description = "Analyze the stock {ticker} price history and create a trend analyses of up, down or sideways",
  agent = stockPriceAnalyst,
  expected_output = """Specify the current trend stock price - up, down or sideways.
  eg. stock = 'APPL, price UP'
  """,
)

search_tool = DuckDuckGoSearchResults(backend = 'news', num_results = 10)

news_analyst_agent = Agent(
    role= "Stock News Analyst",
    goal="""Create a short summary of the market news related to the stock {ticker} company. Specify the current trend - up, down or sideways with
    the news context. For each request stock asset, specify a numbet between 0 and 100, where 0 is extreme fear and 100 is extreme greed.""",
    backstory="""You're highly experienced in analyzing the market trends and news and have tracked assest for more then 10 years.

    You're also master level analyts in the tradicional markets and have deep understanding of human psychology.

    You understand news, theirs tittles and information, but you look at those with a health dose of skepticism.
    You consider also the source of the news articles.
    """,
    verbose = True,
    llm = llm,
    max_iter = 100,
    memory = True,
    tools = [search_tool]
)

get_news_task = Task(
    description= f"""Take the stock.
    Use the search tool to search each one individually.

    The current date is {datetime.now()}.

    Compose the results into a helpfull report""",
    expected_output = """"A summary of the overall market and one sentence summary for each request asset.
    Include a fear/greed score for each asset based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
""",
    agent= news_analyst_agent
)

stock_analyst_write_agent = Agent(
    role = "Senior Stock Analyts Writer",
    goal = """"Analyze the trends price and news and write an insighfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend. """,
    backstory = """You're widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stories
    and narratives that resonate with wider audiences.

    You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses.
    You're able to hold multiple opinions when analyzing anything.
""",
    verbose = True,
    llm = llm,
    max_iter = 5,
    memory = True,
    allow_delegation = True
)

write_analyses_task = Task(
    description = """Use the stock price trend and the stock news report to create an analyses and write the newsletter about the {ticker} company
    that is brief and highlights the most important points.
    Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
    Include the previous analyses of stock trend and news summary.
""",
    expected_output = """"An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner. It should contain:

    - 3 bullets executive summary
    - Introduction - set the overall picture and spike up the interest
    - main part provides the meat of the analysis including the news summary and fead/greed scores
    - summary - key facts and concrete future trend prediction - up, down or sideways.

	The response must be in PT-BR.
""",
    agent = stock_analyst_write_agent,
    context = [get_stock_price_task, get_news_task]
)

crew = Crew(
    agents = [stockPriceAnalyst, news_analyst_agent, stock_analyst_write_agent],
    tasks = [get_stock_price_task, get_news_task, write_analyses_task],
    verbose = 2,
    process = Process.hierarchical,
    full_output = True,
    share_crew = False,
    manager_llm = llm,
    max_iter = 15
)

with st.sidebar:
  st.header('Enter the ticker to research')

  with st.form(key = 'research_form'):
    ticker = st.text_input('Select the ticker')
    submit_button = st.form_submit_button(label = "Run reseach")

if submit_button:
  if not ticker:
    st.error('Please fill the ticker')
  else:
    results = crew.kickoff(inputs = { 'ticker': ticker })

    st.subheader('Results of your research:')
    st.write(results['final_output'])
