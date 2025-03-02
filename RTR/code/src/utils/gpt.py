from openai import OpenAI
import pdb
import os

with open('./openai_api.key', 'r') as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)
def gpt_chat(model, prompt, seed=44):
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=4096,
    temperature=0.5,
    logprobs=True
    )
    
    return response.choices[0].message.content

def gpt_chat_35(prompt, seed=44):
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=4096,
    temperature=0.5,
    logprobs=True
    )
    
    return response.choices[0].message.content

if __name__ == '__main__':
    prompt = "What is the ICD-10 code for Diabetes Mellitus?"
    model = "gpt-3.5-turbo"
    response = gpt_chat(model, prompt)
    print(response)