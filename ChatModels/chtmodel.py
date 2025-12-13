from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=120,
    temperature=0.5,
)

chat = ChatHuggingFace(llm=llm)

res = chat.invoke("What is machine learning?")
print(res.content)
