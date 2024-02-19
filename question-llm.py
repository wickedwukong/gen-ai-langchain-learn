from langchain_openai import OpenAI
import dotenv
import os

dotenv.load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Initiating the OpenAI LLM with API key
llm = OpenAI(api_key=api_key)



# Query the model
response = llm.invoke("What is the tallest building in the world?")
print(response)