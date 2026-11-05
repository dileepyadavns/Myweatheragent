import os
import certifi
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
import requests
import streamlit as st
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv(Path(__file__).with_name(".env"))

tavily_api_key = os.getenv("TAVILY_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

search_tool = TavilySearch(
	max_results=5,
	topic="news",
	time_range="week",
	search_depth="advanced",
	tavily_api_key=tavily_api_key,
)

llm = ChatGoogleGenerativeAI(
	model="gemini-3.5-flash-lite",
	google_api_key=gemini_api_key,
)
prompt = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"Use web search for current information. For recent-event questions, prioritize the newest dated reports. Only rely on sources that explicitly refer to Bigg Boss Telugu Season 10; ignore Tamil and other regional editions. Answer in exactly one concise line and include the source URL. If matching evidence is unavailable, say you cannot verify the answer. Do not guess.",
		),
		("human", "{input}"),
		MessagesPlaceholder(variable_name="agent_scratchpad"),
	]
)


@tool
def mathematical_calcutions(a, b, operation):
    """Perform basic mathematical calculations."""
    try:
        a = float(a)
        b = float(b)
    except ValueError:
        return "Invalid input: a and b must be numbers."

    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            return "Error: Division by zero."
        return a / b
    else:
        return f"Invalid operation: {operation}. Supported operations are add, subtract, multiply, divide."
@tool
def weather_information(city):
	"""Get current weather information for a city."""
	weather_api_key = os.getenv("Weather_API_KEY")
	if not weather_api_key:
		return "Weather API key is not set."

	try:
		response = requests.get(
			"https://api.weatherstack.com/current",
			params={"access_key": weather_api_key, "query": city, "units": "m"},
			timeout=15,
		)
		response.raise_for_status()
		data = response.json()
	except (requests.RequestException, ValueError):
		return "Could not retrieve weather information."

	if "error" in data:
		return f"Weather service error: {data['error'].get('info', 'Unknown error.')}"

	current = data.get("current")
	if not current or current.get("temperature") is None:
		return "Weather data was unavailable."

	temp = current["temperature"]
	descriptions = current.get("weather_descriptions") or []
	description = descriptions[0] if descriptions else "conditions unavailable"
	return f"The current temperature in {city} is {temp}°C with {description}."

tools = [search_tool,weather_information, mathematical_calcutions]

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

create_agent = AgentExecutor(agent=agent, tools=tools, verbose=False)

st.set_page_config(page_title="LangChain Assistant", layout="centered")
st.title("LangChain Assistant")

with st.form("question_form"):
	question = st.text_input("Question", placeholder="Ask about current events or weather")
	submitted = st.form_submit_button("Ask", type="primary")

if submitted:
	if not question.strip():
		st.warning("Enter a question first.")
	elif not gemini_api_key or not tavily_api_key:
		st.error("Set GEMINI_API_KEY and TAVILY_API_KEY in the .env file.")
	else:
		with st.spinner("Finding an answer..."):
			try:
				response = create_agent.invoke({"input": question.strip()})
				answer = response["output"]
				if isinstance(answer, list):
					answer = " ".join(
						item.get("text", "") if isinstance(item, dict) else str(item)
						for item in answer
					)
			except Exception as exc:
				st.error(f"Could not get an answer: {exc}")
			else:
				st.subheader("Answer")
				st.write(" ".join(str(answer).split()))