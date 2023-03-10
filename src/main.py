from langchain.agents import tool, Tool, ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)

import src.util.templates as tp

info = tp.generate_info()
self_description = input("Describe your personality and other facts in one paragraph: \n")
personality = tp.generate_personality(self_description)
gen_prompt = tp.generate_template(info['user'], info['recipient'], personality=personality)

search = GoogleSearchAPIWrapper()

@tool("Contact", return_direct=True)
def contactAPI(query: str) -> str:
    """useful for when you are asked personal questions that you do not know the answer to"""
    return "Contact Here"

tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),

    contactAPI()
]

prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=gen_prompt['prefix'], 
    suffix=gen_prompt['suffix'], 
    input_variables=["input", "agent_scratchpad"]
)

messages = [
    SystemMessagePromptTemplate(prompt=prompt),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}\n\nThis was your previous work "
                f"(but I haven't seen any of it! I only see what "
                "you return as final answer):\n{agent_scratchpad}")
]

prompt = ChatPromptTemplate.from_messages(messages)

tool_names = [tool.name for tool in tools]

memory = ConversationBufferMemory()
llm = ChatOpenAI(temperature=0)

conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
agent = ZeroShotAgent(llm_chain=conversation, allowed_tools=tool_names)
conversation_agent = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

def queryAgent(input_text):
    print(conversation_agent.run(input_text))