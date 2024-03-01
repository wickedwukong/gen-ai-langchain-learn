# importing LangChain modules
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import dotenv
import os

dotenv.load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')


# Insert your key here
llm = OpenAI(temperature=0.0,
            openai_api_key = api_key)

memory = ConversationBufferMemory()
memory.save_context({"input": "Alex is a 9-year old boy."}, 
                    {"output": "Hello Alex! How can I assist you today?"})
memory.save_context({"input": "Alex likes to play football"}, 
                    {"output": "That's great to hear! "})

conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=True
)

print(conversation.predict(input="How old is Alex?"))