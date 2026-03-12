from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda


load_dotenv()

def word_counter(text):
    return len(text.split())

# runnable_word_counter=RunnableLambda(word_counter)


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


chain_joke = RunnableSequence(prompt1, model, parser)

parallel_chain=RunnableParallel({
    "Joke": RunnablePassthrough(),
    "word_count" :RunnableLambda(word_counter)
})

final_chain = RunnableSequence(chain_joke, parallel_chain)

result = final_chain.invoke({'topic':'Football'})


final_result = """{} \n word count - {}""".format(result['Joke'], result['word_count'])

print(final_result)