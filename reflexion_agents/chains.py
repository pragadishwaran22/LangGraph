from langchain.messages import HumanMessage
import datetime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from schema import AnswerQuestion
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

load_dotenv()

pydantic_parser = PydanticOutputParser(pydantic_object=AnswerQuestion)
format_instructions = pydantic_parser.get_format_instructions()


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
).partial(
time=lambda: datetime.datetime.now().isoformat(),
format_instructions=format_instructions
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction = "provide a detailed -250 word answer"
)

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

first_responder_chain = first_responder_prompt_template | llm | pydantic_parser 

response = first_responder_chain.invoke({
    "messages": [HumanMessage(content="write me a blog on how small bussiness can leverage the use of AI for their growth")]
})
print(response.reflection)


