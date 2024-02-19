from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
import dotenv
import os

dotenv.load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Initiating the chat model with API key
chat = ChatOpenAI(api_key=api_key)

# Defines a context and query using SystemMessage and HumanMessage
messages = [
    SystemMessage(content="You are a math tutor who provides answers with a bit of sarcasm."),
    HumanMessage(content="What is the square of 2?"),
]
 
response = chat.invoke(messages)
print(response.content)
