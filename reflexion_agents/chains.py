from langchain.messages import HumanMessage,AIMessage
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
Do Not return empty output

- Use the critique to improve the answer.
- Keep the answer under 250 words.
- Include numerical citations in the answer like [1], [2].

STRICT OUTPUT RULES:

- Return ONLY valid JSON.
- Do NOT include any text outside JSON.
- Do NOT include a "References" section inside the answer.

- You MUST include a separate field:
  "references": ["https://example.com", "https://example.com"]

- The "references" field must contain ONLY URLs.
- The "answer" field must NOT contain URLs.
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








