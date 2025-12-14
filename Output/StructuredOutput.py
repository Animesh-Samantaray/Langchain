from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated

load_dotenv()

model= ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.7
)

class Review(TypedDict):
    summary:str
    sentiment:str

StructuredModel_model=model.with_structured_output(Review)

result=StructuredModel_model.invoke('I am very happy with this product . It is making our life very easy .Such a bad product this is . I feel guilty tafter buying this')

print(result)