# https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-1-build-a-basic-chatbot
from typing import Annotated
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_community.tools.tavily_search import TavilySearchResults
import os

load_dotenv()
# load tools

tools = []
if (os.getenv("TAVILY_API_KEY")):
    tavily = TavilySearchResults(max_results=2)
    # Test tavily tool
    # print(tavily.invoke("What's a 'node' in LangGraph?"))
    tools.append(tavily)    
                 
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)
print(llm_with_tools.invoke("What's a 'node' in LangGraph?"))

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    response = llm.invoke(state["messages"])
    #res = llm_with_tools.invoke(state["messages"])
    #print("Response:", res)
    return {"messages": [response]}
    #return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node('chatbot', chatbot)
graph_builder.add_edge(START, 'chatbot')
graph_builder.add_edge('chatbot', END)
graph = graph_builder.compile()

try:
    image_path = "graph.png"
    display(graph.get_graph().draw_ascii())
except Exception as e:
    pass

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except Exception as e:
        # fallback if input() is not available
        print(e)
        break