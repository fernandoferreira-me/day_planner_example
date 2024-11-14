
import streamlit as st

# Agent and LLM
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ConversationalAgent
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
# Memory
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
# Tools
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.utilities import OpenWeatherMapAPIWrapper

import os


st.title("Smart Day Planner")


# Set Tools
SERPER_API_KEY = os.getenv("SERPER_API_KEY") 
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
weather = OpenWeatherMapAPIWrapper(openweathermap_api_key=OPENWEATHERMAP_API_KEY)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to get current, up to date answers.",
    ),
    Tool(
        name="Weather",
        func=weather.run,
        description="Useful for when you need to get the current weather in a location.",
    )
]

# Set Chat Conversation

prefix = """ You are a friendly modern day planner.
You can help users to find activities in a given city based
on their preferences and the weather.
You have access to the two tools:
"""

suffix = """
Chat History:
{chat_history}
Latest Question: {input}
{agent_scratchpad}
"""

prompt = ConversationalAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input",
                     "chat_history",
                     "agent_scratchpad"],
)

# Set Memory

msg = StreamlitChatMessageHistory()

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        messages=msg,
        memory_key="chat_history",
        return_messages=True
    )
memory = st.session_state.memory

# Set Agent

llm_chain = LLMChain(
    llm=ChatOpenAI(temperature=0.8, model_name="gpt-4"),
    prompt=prompt,
    verbose=True
)

agent = ConversationalAgent(
    llm_chain=llm_chain,
    memory=memory,
    verbose=True,
    max_interactions=3,
    tools=tools
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                    tools=tools,
                                                    memory=memory,
                                                    verbose=True)

query = st.text_input("O que você quer fazer hoje?", placeholder="Digite aqui...")

if query:
    with st.spinner("Estou pensando..."):
        result = agent_executor.run(query)
        st.info(result, icon="🤖")

with st.expander("My thinking"):
    st.write(st.session_state.memory.chat_memory.messages)