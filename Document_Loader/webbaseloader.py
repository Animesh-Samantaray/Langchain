from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
url='https://takeuforward.org/profile/Animesh18'

load_dotenv()

loader=WebBaseLoader(url)

docs=loader.load()

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash-lite",
    temperature=0.5
)
prompt=PromptTemplate(
    template='Answer the following question {qstn} from the following text \n {text}',
    input_variables=['qstn','text']
)

parser=StrOutputParser()

chain=prompt|model|parser
res=chain.invoke({'qstn':'WHat does this describe about and what is the name of the user ? ' , 'text':docs[0].page_content})
print(res)