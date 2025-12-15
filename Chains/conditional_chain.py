from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from typing import Literal

class Feedback(BaseModel):
    sentiment:Literal['positive','negative']  = Field(description='Give the sentiment of the following feedback')

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.2
)

parser1=StrOutputParser()
parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template='classify the sentiment of the following feedback text into possetive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

prompt_pos=PromptTemplate(
    template='Write an appropriate response to this possetive  \n {feedback} ',
    input_variables=['feedback'],
)
prompt_neg=PromptTemplate(
    template='Write an appropriate response to this negative  \n {feedback} ',
    input_variables=['feedback'],
)

classifier_chain=prompt1 | model | parser2

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive' , prompt_pos|model|parser1),
    (lambda x:x.sentiment=='negative' , prompt_neg|model|parser1),
    RunnableLambda(lambda x:'Could not find sentiment')
)

chain=classifier_chain | branch_chain

result=chain.invoke({'feedback':'It is a very beautiful mobile phone'})
print(result)