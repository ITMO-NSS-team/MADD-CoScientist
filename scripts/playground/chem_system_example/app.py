
from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get('groq2')
from graph import app


model = ChatOpenAI(model='llama-3.3-70b-versatile',
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
                temperature=0.1)

visual_model = ChatOpenAI(model='llama-3.2-90b-vision-preview',
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key)


config = {"recursion_limit": 50,
          "configurable": {"model": model,
                           'visual_model': visual_model,
                           'img_path': 'image.png'}}


inputs = {"input": "Посчитай qed, tpsa, logp, hbd, hba свойства ацетона"}

for event in app.stream(inputs, config=config):
    for k, v in event.items():
        if k != "__end__":
            print(v)

print("\n\nFIN: ", v)
