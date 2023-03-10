from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent
import src.util.templates as tp

search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name = "Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history")
llm=OpenAI(temperature=0, model_name = "gpt-3.5-turbo")
agent_chain = initialize_agent(tools, llm, agent="conversational-react-description", verbose=True, memory=memory)

def queryAgent(input_text):
    print(agent_chain.run(input_text))