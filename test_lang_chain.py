import os

import dotenv

dotenv.load_dotenv()

from lang_chain import LangChain

openai_api_key = os.getenv('OPENAI_API_KEY')

