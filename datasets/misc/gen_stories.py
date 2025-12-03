from openai import OpenAI
from tqdm import tqdm
import numpy as np
import pandas as pd
from google import genai

client = genai.Client(api_key="YOUR_API_KEY")



story_prompts = [
    "Old friend calls after twenty years of silence.",
    "You find someone’s wallet and recognize the photo inside.",
    "Neighbor asks you to keep a strange secret.",
    "You overhear a conversation not meant for you.",
    "Unexpected letter arrives with no return address or name.",
    "A chance meeting changes your entire daily routine.",
    "You help a stranger and they never say thanks.",
    "Lost phone reveals someone’s hidden double life.",
    "You see someone you know pretending not to notice.",
    "Childhood home is up for sale again."
]

def main():
    # key = "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"
    # client = OpenAI(api_key=key )
    
    key = "AIzaSyAPCLlG444BsPqXah-y7bIFlnZla570Afs"
    client = genai.Client(api_key=key)

    temps = np.linspace(0.05, 1, 20)

    data = pd.DataFrame(columns=["temperature", "topic", "story"])

    for temp in tqdm(temps):
        for story in tqdm(story_prompts, leave=False):
            # OpenAI API call to generate a story - switch to output_text to use
            # response = client.responses.create(
            #     model="gpt-4o-mini",
            #     input=f"Write a short story between 7-13 sentences long about the following topic: {story}. Do not include any metacommentary, only return the story.",
            #     temperature=temp
            # )

            # Gemini API call to generate a story - switch to text to use
            response = client.models.generate_content(
                model="gemini-2.0-flash", contents="Write a short story between 7-13 sentences long about the following topic: {story}. Do not include any metacommentary, only return the story.",
                temperature=temp
            )

            data = pd.concat([data, pd.DataFrame({"temperature": [temp], "topic": [story], "story": [response.text]})], ignore_index=True)

        
    data.to_csv("./syntehtic-stories-gemini.csv")
            

if __name__ == "__main__":    
    main()