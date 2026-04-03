from typing import List, Sequence
from chains import generation_chain, reflection_chain
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import END,MessageGraph



REFLECT = "reflect"
GENERATE = "generate"
graph = MessageGraph()

def generate_node(state):
    return generation_chain.invoke({
        "messages": state
    })

def reflection_node(messages):
    response = reflection_chain.invoke({
        "messages": messages
    })
    return [HumanMessage(content=response.content)] 

graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflection_node)

graph.set_entry_point(GENERATE)

def should_continue(state):
    if len(state)>6:
        return END
    return REFLECT

graph.add_conditional_edges(GENERATE, should_continue,{END:END , REFLECT:REFLECT})
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

response = app.invoke(HumanMessage(content = "Write a tweet about the importance of AI in the modern world"))
print(response)