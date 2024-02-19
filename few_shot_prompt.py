# importing LangChain modules
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts import PromptTemplate

import dotenv
import os

dotenv.load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Initiating the chat model with API key
chat = ChatOpenAI(temperature=0.0, openai_api_key = api_key)

examples = [
  {
    "review": "I absolutely love this product! It exceeded my expectations.",
    "sentiment": "Positive"
  },
  {
    "review": "I'm really disappointed with the quality of this item. It didn't meet my needs.",
    "sentiment": "Negative"
  },
  {
    "review": "The product is okay, but there's room for improvement.",
    "sentiment": "Neutral"
  }
]

example_prompt = PromptTemplate(
                        input_variables=["review", "sentiment"], 
                        template="Review: {review}\n{sentiment}")

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Review: {input}",
    input_variables=["input"]
)

message = prompt.format(input="The machine worked okay without much trouble.")

response = chat.invoke(message)
print(response.content)