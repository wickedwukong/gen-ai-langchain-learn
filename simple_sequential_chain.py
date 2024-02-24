# importing the modules
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
import dotenv
import os

dotenv.load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')


# defining the LLM model
llm = OpenAI(temperature=0.0, openai_api_key=api_key)


prompt_1 = PromptTemplate(
    input_variables=["book"],
    template="Name the author who wrote the book {book}?"
)
chain_1 = LLMChain(llm=llm, prompt=prompt_1, verbose=True)

prompt_2 = PromptTemplate(
    input_variables=["author_name"],
    template="Write a 50-word biography for the following author:{author_name}"
)
chain_2 = LLMChain(llm=llm, prompt=prompt_2)

simple_sequential_chain = SimpleSequentialChain(chains=[chain_1, chain_2], verbose=True)

# calling the chain
print(simple_sequential_chain.run("The Da Vinci Code"))
