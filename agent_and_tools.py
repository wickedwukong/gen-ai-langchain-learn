import os
# importing LangChain modules
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
import dotenv

dotenv.load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
serp_api_key = os.getenv('MY_SERPAPI_API_KEY')

os.environ["SERPAPI_API_KEY"] = serp_api_key

# Insert your key here
llm = OpenAI(temperature=0.0,
            openai_api_key = api_key)

# loading tools
tools = load_tools(["serpapi", 
                    "llm-math"], 
                    llm=llm)

agent = initialize_agent(tools, 
                        llm, 
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                        verbose=True)

# user's query
print(agent.run("What is the current population of the world, and calculate the percentage change compared to the population five years ago"))