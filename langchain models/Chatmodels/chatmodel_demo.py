from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI(model='gpt-4',temperature=1.5,max_completion_tokens=10) #temp for deterministic or creative

results=model.invoke("What is the capital of Nepal")

print(results.content)



