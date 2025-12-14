from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
chat_template=ChatPromptTemplate([
    ('system','You are a great customer support agent'),
    MessagesPlaceholder(
        variable_name='chat_history'
    ),

    ('human','{query}')
])

chat_history=[]

with open('chat_history.txt') as f:
    for line in f:
        chat_history.append(HumanMessage(content=line.strip()))

prompt =chat_template.invoke({
    'chat_history':chat_history,
    'query':'Where is my refund ? '
})

print(prompt)