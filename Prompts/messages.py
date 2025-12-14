from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.7
)

messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about Langchain')
]

result=model.invoke(messages)

print(result.content)