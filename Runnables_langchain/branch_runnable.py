from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch

def word_counter(text):
    return len(text.split())

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
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)


prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

report_generation_chain=RunnableSequence(prompt1,model,parser)

branch_chain=RunnableBranch(
    (lambda x: len(x.split())>500,RunnableSequence(prompt2,model,parser)), #(condition,runnable)
    RunnablePassthrough() #default runnable after n conditions above
)

final_chain=RunnableSequence(report_generation_chain,branch_chain)

result=final_chain.invoke({'topic': 'Football'})
print(result)