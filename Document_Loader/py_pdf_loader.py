from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader=PyPDFLoader('Document_Loader\/ai.pdf')

doc=loader.load()


# model = ChatGoogleGenerativeAI(
#     model="models/gemini-2.5-flash-lite",
#     temperature=0.5
# )

print(len(doc))