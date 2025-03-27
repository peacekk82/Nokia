import base64
from openai import OpenAI
api_key = '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'

# Base64 encode the image
image_path = "patent_photo.png"  # Replace with your image path
with open(image_path, "rb") as img_file:
    image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)  # Replace with your key

# Correctly structured API call
response = client.chat.completions.create(
    model="gpt-4o",  # Use GPT-4o model
    messages=[
        {"role": "system", "content": "You are a helpful assistant that analyzes images."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "find the abstract of this patent"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"  # Correct format
                    }
                }
            ]
        }
    ],
    max_tokens=1000
)

# Print the response
print(response.choices[0].message.content)
