import os
import ast
import openai
from openai import OpenAI
import tiktoken
import pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"

BASE_PROMPT = """
Please analyze the following customer review and extract the following metadata:
1. Sentiment: A polarity score from -1.0 (very negative) to +1.0 (very positive).
2. Emotions: Which of the following best apply: joy, anger, frustration, satisfaction (list only one that best applies).
3. Clarity: A rating from 1 (poorly written) to 5 (very clear).
4. Depth: A rating from 1 (very brief) to 5 (very detailed).
5. Subjectivity: A rating from 1 (mostly objective) to 5 (highly opinion-based).
6. Predicted_Usefulness: A numeric estimate (1-5) of how useful this review might be.

Return your result as plain text in the following format (match the keys exactly):
{"sentiment": -0.8, "emotions": 'frustration', "clarity": 4, "depth": 2, "subjectivity": 4, "predicted_usefulness": 4}
"""

def build_full_prompt(review_text):
    """
    Returns a complete prompt by appending the review text to the BASE_PROMPT.
    """
    return f"""{BASE_PROMPT}
Review text:
\"
{review_text}
\"
"""

def approximate_token_count(text, model = "gpt-3.5-turbo"):
    """
    Uses the tiktoken library to estimate the number of tokens for a given string.
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

def get_review_metadata(review_text):
    """
    Sends the review text to the LLM, parses its response, and returns the metadata as a dictionary.
    """
    client = OpenAI()
    
    user_prompt = build_full_prompt(review_text)
    
    prompt_tokens = approximate_token_count(user_prompt, model = MODEL_NAME)
    print(f"Approx. prompt tokens: {prompt_tokens}")
    
    response = client.chat.completions.create(
        model = MODEL_NAME,
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ],
        temperature = 0.0, # deterministic generation
        max_tokens = 200,
        n = 1,
    )
    
    response_text = response.choices[0].message.content    

    try:
        metadata = ast.literal_eval(response_text)
    except Exception as e:
        print("Error parsing metadata:", e)
        metadata = {}
    return metadata

def process_reviews_df(df, text_column = "text"):
    """
    Processes a DataFrame of review texts, querying the LLM for each review and appending the extracted
    metadata as new columns.
    """
    sentiments = []
    emotions = []
    clarities = []
    depths = []
    subjectivities = []
    predicted_usefulness_list = []
    
    for idx, row in df.iterrows():
        review_text = row[text_column]
        
        if len(df) <= 10:
            print(f"\nProcessing review at index {idx}")
        elif idx%10 == 0:
            print(f"\nProcessing review at index {idx}")
            
        metadata = get_review_metadata(review_text)
        
        sentiments.append(metadata.get("sentiment"))
        emotions.append(metadata.get("emotions"))
        clarities.append(metadata.get("clarity"))
        depths.append(metadata.get("depth"))
        subjectivities.append(metadata.get("subjectivity"))
        predicted_usefulness_list.append(metadata.get("predicted_usefulness"))
    
    df["sentiment"] = sentiments
    df["emotions"] = emotions
    df["clarity"] = clarities
    df["depth"] = depths
    df["subjectivity"] = subjectivities
    df["predicted_usefulness"] = predicted_usefulness_list
    
    return df