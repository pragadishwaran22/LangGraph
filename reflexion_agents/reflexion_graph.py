from langchain.messages import AIMessage
from langchain_core.messages import BaseMessage,ToolMessage
from typing import List
from langgraph.graph import END,MessageGraph
from chains import first_responder_chain,revisor_chain
from tool_excutor import tool_executor

graph = MessageGraph()

def first_responder(state):
    result=first_responder_chain.invoke({
        "messages":state
    })
    return AIMessage(content=result.model_dump_json())

def revisor_node(state):
    result = revisor_chain.invoke({
        "messages": state 
    })

    return AIMessage(content=result.model_dump_json())

graph.add_node("draft",first_responder)
graph.add_node("tool",tool_executor)
graph.add_node("revisor",revisor_node)

max_iteration = 1

def should_continue(state:List[BaseMessage])->str :
    count_tool_visit = sum(isinstance(item,ToolMessage)for item in state)
    num_iteration = count_tool_visit
    if num_iteration > max_iteration:
        return END
    return"tool"

graph.add_edge("draft","tool")
graph.add_edge("tool","revisor")
graph.add_conditional_edges("revisor",should_continue,{END:END,"tool":"tool"})

graph.set_entry_point("draft")

app=graph.compile()
print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

response = app.invoke(
    "write about how a traditional local jwellery shop can turn into a big enterprise with leverage of AI"
)

print(response)

