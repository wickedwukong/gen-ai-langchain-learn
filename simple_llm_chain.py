# importing the modules
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import dotenv
import os

dotenv.load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')


# defining the LLM model
llm = OpenAI(temperature=0.0, openai_api_key=api_key)

# creating the prompt template
prompt_template = PromptTemplate(
    input_variables=["book"],
    template="Name the author of the book {book}?",
)

# creating the chain
chain = LLMChain(llm=llm,
                 prompt=prompt_template,
                 verbose=True)

# calling the chain
print(chain.run("The Da Vinci Code"))
