import os
import pdb

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

with open('./claude_api_aws.key', 'r') as f:
    # two lines, one for AWS_ACCESS_KEY_ID, the next for AWS_SECRET_ACCESS_KEY
    keys = f.readlines()
    AWS_ACCESS_KEY_ID = keys[0].strip()
    AWS_SECRET_ACCESS_KEY = keys[1].strip()


AWS_DEFAULT_REGION = "us-west-2"
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
from anthropic import AnthropicBedrock
client = AnthropicBedrock(
    aws_access_key=AWS_ACCESS_KEY_ID,
    aws_secret_key=AWS_SECRET_ACCESS_KEY,
    aws_region=AWS_DEFAULT_REGION
)


def chat_haiku(prompt):

    if isinstance(prompt, str):
        message = [{
            'role': 'user',
            'content': prompt
        }]
    else:
        message = prompt
    
    message = client.messages.create(
        temperature=0,
        model="anthropic.claude-3-haiku-20240307-v1:0",
        max_tokens=1024,
        messages=message,
    )
    
    return message.content[0].text

def chat_sonnet(prompt):
    
    if isinstance(prompt, str):
        message = [{
            'role': 'user',
            'content': prompt
        }]
    else:
        message = prompt

    message = client.messages.create(
        temperature=0,
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        max_tokens=1024,
        messages=message,
    )
    
    return message.content[0].text


if __name__ == "__main__":
    prompt = "Write a haiku about the ocean."
    print(chat_haiku(prompt))