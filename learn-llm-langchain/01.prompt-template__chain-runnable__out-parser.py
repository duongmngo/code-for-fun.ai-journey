import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template), 
        ("user", "{text}")
    ]
)

output_parser = StrOutputParser()
chain = prompt_template | model | output_parser
result = chain.invoke({"language": "Vietnamese", "text": "hi!"})
print(result)
