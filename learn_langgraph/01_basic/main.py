# https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-1-build-a-basic-chatbot
import json
from typing import Annotated
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
import os
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

load_dotenv()
# load tools

tools = []
tavily = None
if (os.getenv("TAVILY_API_KEY")):
    tavily = TavilySearchResults(max_results=2)
    print('Tavily tool loaded')
    tools.append(tavily)    
                 
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

tool_node = BasicToolNode(tools=[tavily])

def chatbot(state: State):    
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

graph_builder = StateGraph(State)
graph_builder.add_node('chatbot', chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools
)
graph_builder.set_entry_point('chatbot')
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile(checkpointer=memory)

try:
    image_path = "graph.png"
    display(graph.get_graph().draw_ascii())
except Exception as e:
    pass

def stream_graph_updates(user_input: str):
    events = graph.stream(
        {"messages": [("user", user_input)]}, 
        {"configurable": {"thread_id": "1"}}, 
        stream_mode="values")
    for event in events:
        event["messages"][-1].pretty_print()
        # for value in event.values():
        #     print("Assistant:", value["messages"][-1].content)

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