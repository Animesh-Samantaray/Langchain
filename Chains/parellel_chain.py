from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
from langchain_core.runnables import RunnableParallel

prompt1=PromptTemplate(
    template='Generate short and simple notes from the folowing text \n {text}',
    input_variables=['text']
)

prompt2=PromptTemplate(
    template='Generate 5 simple short questions from the following text \n {text}',
    input_variables=['text']
)

prompt3=PromptTemplate(
    template='Merge the provided notes and quizes into a single document \n notes->{notes} \n quiz->{quiz}',
    input_variables=['notes','quiz']
)

model1 = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.7
)
model2 =  ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.7
)


parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz':prompt2|model2|parser
})

merge_chain=prompt3 | model1 | parser

chain=parallel_chain | merge_chain

text="""
JavaScript is a programming language used to make websites interactive. It was created in 1995 and is now one of the most popular languages in the world.

When you open a website and click a button, fill a form, or see content change without reloading the page, JavaScript is working behind the scenes. It runs directly in the web browser and controls how a page behaves.

Originally, JavaScript was only used in browsers. Later, with the introduction of Node.js, it could also run on servers. This made it possible to build both the front end and back end using the same language.

JavaScript is known for being fast, flexible, and event-based. It can handle tasks like loading data, showing animations, and responding to user actions smoothly.

Today, JavaScript is used everywhere—websites, mobile apps, servers, and even games. Popular tools like React and Node.js are built using JavaScript, making it an essential technology for modern development.

"""

result = chain.invoke({'text':text})
print(result)
chain.get_graph().print_ascii() 