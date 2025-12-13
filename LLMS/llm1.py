from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    max_new_tokens=120,
    temperature=0.3,
)

chat = ChatHuggingFace(llm=llm)

res = chat.invoke("What is machine learning?")
print(res.content)
