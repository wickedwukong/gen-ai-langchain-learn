# importing LangChain modules
from langchain_openai.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import DatetimeOutputParser, CommaSeparatedListOutputParser
import dotenv
import os

dotenv.load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')


# Insert your key here
llm = OpenAI(openai_api_key = api_key)

parser_dateTime = DatetimeOutputParser()
parser_List = CommaSeparatedListOutputParser()

# creating our prompt template
template = """Provide the response in format {format_instructions} 
            to the user's question {question}"""

prompt_dateTime = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": parser_dateTime.get_format_instructions()},
)

prompt_List = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": parser_List.get_format_instructions()},
)

# formatting the output
iphone_launch_date = llm.predict(text = prompt_dateTime.format(question="When was the first iPhone launched?"))
print(iphone_launch_date)
print(type(iphone_launch_date))
choc_list = llm.predict(text = prompt_List.format(question="What are the four famous chocolate brands?"))
# print(llm.predict(text = prompt_List.format(question="What are the four famous chocolate brands?")))
print(choc_list)
print(choc_list[0])
print(type(choc_list))

parsed_list = parser_List.parse(choc_list)
print(parsed_list)
print(type(parsed_list))