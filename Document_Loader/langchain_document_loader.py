from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader=TextLoader('Document_Loader\cricket.txt',encoding='utf-8')

doc=loader.load()


model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash-lite",
    temperature=0.5
)
prompt=PromptTemplate(
    template='Write a summaryr of the following poem \n {poem}',
    input_variables=['poem']
)

parser=StrOutputParser()

chain=prompt|model|parser
res=chain.invoke({'poem':doc[0].page_content})
print(res)