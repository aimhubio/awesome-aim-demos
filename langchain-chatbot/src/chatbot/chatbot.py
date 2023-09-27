from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent

from aimstack.langchain_debugger.callback_handlers import GenericCallbackHandler

def chatbot(serpapi_key, openai_key):
    # Configs
    model_name = 'gpt-4-0613'

    # Simple chatbot implementation
    memory = ConversationBufferMemory(memory_key="chat_history")

    search = SerpAPIWrapper(serpapi_api_key=serpapi_key)
    tools = [
        Tool(
            name = "Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world"
        ),
    ]
    # TODO: log [tool.__dict__ for tool in tools]

    llm = ChatOpenAI(temperature=0, openai_api_key=openai_key, model_name=model_name)
    # TODO: log llm.__dict__

    agent_chain = initialize_agent(
        tools, llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors="Check your output and make sure it conforms!"
    )
    # TODO: log agent_chain.__dict__

    # Init the callback
    aim_cb = GenericCallbackHandler()

    # Run the bot
    while True:
        msg = input('Message:\n')
        response = agent_chain.run(input=msg, callbacks=[aim_cb])
        aim_cb.flush()
        # try:
        #     response = agent_chain.run(input=msg, callbacks=[aim_cb])
        # except ValueError as e:
        #     response = str(e)
        #     if not response.startswith("Could not parse LLM output: `"):
        #         raise e
        # response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
