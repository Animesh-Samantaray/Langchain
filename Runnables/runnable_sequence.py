from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda,RunnableSequence

prompt=PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Explain the following joke  {text}',
    input_variables=['text']
)

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash-lite",
    temperature=0.5
)

parser=StrOutputParser()


chain=RunnableSequence(prompt,model,parser,prompt2,model,parser)


print(chain.invoke({'topic':'Bollywood'}))