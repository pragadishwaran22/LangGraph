from langchain_core.messages import BaseMessage,ToolMessage
from typing import List
from langgraph.graph import END,MessageGraph
from chains import first_responder_chain,revisor_chain
from tool_excutor import tool_executor

graph = MessageGraph()

graph.add_node("draft",first_responder_chain)
graph.add_node("tool",tool_executor)
graph.add_node("revisor",revisor_chain)

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



