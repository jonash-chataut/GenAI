from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
import os

load_dotenv()

llm=HuggingFacePipeline.from_model_id(
    model_id="MiniMaxAI/MiniMax-M2.5",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

model=ChatHuggingFace(llm=llm)

result=model.invoke("what is the capital of nepal")

print(result.content)
