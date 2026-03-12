from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.runnables import RunnableSequence,RunnableParallel


load_dotenv()

# Define the model
llm=HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.5",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
model=ChatHuggingFace(llm=llm)


parser = StrOutputParser()


prompt1 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)


prompt2 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

parallel_chain=RunnableParallel({
    "linkedin": RunnableSequence(prompt1,model,parser),
    "tweet" :RunnableSequence(prompt2,model,parser)
})

result=parallel_chain.invoke({'topic':'AI'})

print(result)