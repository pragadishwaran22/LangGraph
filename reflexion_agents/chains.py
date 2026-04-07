from langchain.messages import HumanMessage
import datetime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from schema import AnswerQuestion, Reflection, ReviseAnswer
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

load_dotenv()

pydantic_parser_responder = PydanticOutputParser(pydantic_object=AnswerQuestion)
format_instructions_1 = pydantic_parser_responder.get_format_instructions()

pydantic_parser_revisor = PydanticOutputParser(pydantic_object=ReviseAnswer)
format_instruction_2 = pydantic_parser_revisor.get_format_instructions()


actor_prompt_template = ChatPromptTemplate.from_messages(
[
    (
        "system",
        """You are expert AI researcher.
Current time: {time}

{format_instructions}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, list 1-3 search queries separately.
Return ONLY valid JSON.
""",
    ),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Answer the user's question above using the required format."),
]
).partial(time=lambda: datetime.datetime.now().isoformat())

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction = "provide a detailed -250 word answer",
    format_instructions=format_instructions_1
)

revisor_prompt_template = actor_prompt_template.partial(format_instructions = format_instruction_2,
first_instruction = revise_instructions
)

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

first_responder_chain = first_responder_prompt_template | llm | pydantic_parser_responder
revisor_chain = revisor_prompt_template | llm | pydantic_parser_revisor


query = "write me a blog on how small bussiness can leverage the use of AI for their growth"
response = first_responder_chain.invoke({
    "messages": [HumanMessage(content=query)]
})
print(response)
revised = revisor_chain.invoke({
    "messages":[HumanMessage(content = query),
    HumanMessage(content = response.answer)]
})
print(revised)


