import json
from os import name
from sys import exception
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_tool = TavilySearchResults(max_results=5)

def tool_executor(state:List[BaseMessage]) -> List[BaseMessage]:
    last_ai_message : AIMessage = state[-1]

    if not hasattr(last_ai_message, "content") or not last_ai_message.content:
        return[]
        
    data =json.loads(last_ai_message.content)    
    search_queries = data.get("search_queries",[])

    if not search_queries:
        return[]
       
    query_results = {}

    for query in search_queries:
        try:
            result = tavily_tool.invoke(query)
            query_results[query]=result
        except Exception as e:
            query_results[query] = {"error": str(e)}

    return [
        ToolMessage(content=json.dumps(query_results),tool_call_id="manual")
    ]

        
            


# test_state = [
#     HumanMessage(
#         content="Write about how small business can leverage AI to grow"
#     ),
#     AIMessage(
#         content="", 
#         tool_calls=[
#             {
#                 "name": "AnswerQuestion",
#                 "args": {
#                     'answer': '', 
#                     'search_queries': [
#                             'AI tools for small business', 
#                             'AI in small business marketing', 
#                             'AI automation for small business'
#                     ], 
#                     'reflection': {
#                         'missing': '', 
#                         'superfluous': ''
#                     }
#                 },
#                 "id": "call_KpYHichFFEmLitHFvFhKy1Ra",
#             }
#         ],
#     )
# ]
    
# Execute the tools
# results = execute_tools(test_state)

# print("Raw results:", results)
# if results:
#     parsed_content = json.loads(results[0].content)
#     print("Parsed content:", parsed_content)
