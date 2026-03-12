from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough


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
    template='Write a joke about {topic}',
    input_variables=['topic']
)


prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain_joke = RunnableSequence(prompt1, model, parser)

parallel_chain=RunnableParallel({
    "Joke": RunnablePassthrough(),
    "Explanation" :RunnableSequence(prompt2,model,parser)
})

final_chain=RunnableSequence(chain_joke,parallel_chain)

print(final_chain.invoke({'topic':'Football'}))