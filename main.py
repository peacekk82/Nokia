

api_key = 'Your API Token'

from openai import OpenAI

client = OpenAI(api_key=api_key)

MODEL="gpt-4o"

system_prompt = '''Your goal is check the similarity of two abstract.

You are given two abstract. 

If you think the two abstracts are too similar which has a risk of infringement, explain why.

- Structure your response in a table format, with the following columns:
  - Abstract 1
  - Abstract 2
  - Risk Level (Low, Mid, High)
  - Explanation'''

user_prompt = '''
Abstract 1: Personalized information is transferred from a first hand portable phone having a first memory means for storing said personalized information to a second hand portable phone having a second memory means for storing said personalized information.
Abstract 2: Personalized information is transferred from a first portable phone, which contains a first memory for storing this information, to 
a second portable phone equipped with a second memory for the same purpose. The process begins by establishing a connection between the first phone and a computer running a data transfer application.

'''

completion = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "system",
   "content": system_prompt},
  {"role": "user", "content": user_prompt}])


from openai import OpenAI

# Set up OpenAI client
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")  # Replace with your API key

# Prepare the request
response = client.chat.completions.create(
    model="gpt-4-vision-preview",  # Use GPT-4 Vision model
    messages=[
        {"role": "system", "content": "You are a helpful assistant that analyzes images."},
        {"role": "user", "content": [
            {"type": "text", "text": "What is shown in this image?"},
            {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
        ]}
    ],
    max_tokens=1000
)

# Print the response
print(response.choices[0].message.content)


print("Assistant: " + completion.choices[0].message.content)
