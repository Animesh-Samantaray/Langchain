from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Create your chat model
model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.7)

# Create prompt template
chat_template = ChatPromptTemplate([
    ('system', 'You are a proficient {domain} expert'),
    ('human', 'Explain in simple terms, about what is {topic}')
])

# Generate messages
messages = chat_template.invoke({'domain':'cricket', 'topic':'cover drive'})

# Send to LLM
result = model.invoke(messages)

print("AI:", result.content)
