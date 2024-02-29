import os
import dotenv
from langchain.llms import OpenAI

dotenv.load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# creating the model to be used
client = OpenAI(api_key=api_key)

# defining the model name and messages to display to check if the key works
completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi, my name is Alex."}
    ]
)

# printing the first message
print(completion.choices[0].message)

# Initiating the OpenAI LLM with API key
llm = OpenAI(api_key="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")

# Query the model
response = llm.invoke("What is the tallest building in the world?")
print(response)
