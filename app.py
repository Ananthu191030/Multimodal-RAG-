from flask import Flask, render_template, request
import os
import requests
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from groq import Groq
import pinecone
import numpy as np

app = Flask(__name__)

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize Pinecone
from pinecone import Pinecone, ServerlessSpec
index_name = "changed"
pc = Pinecone(api_key='pcsk_a7YMi_5NbA2fzbYru7WYRhu1seo9NA3fqTwNLuseL8JeC5Ftbr69d3duFgvxPLmz7s2xb')
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(index_name)

# Initialize Groq API
os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def normalize(vector):
    """Normalize a vector to unit norm."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def process_uploaded_image(image_file):
    """Process an uploaded image for CLIP model."""
    try:
        image = Image.open(image_file).convert("RGB")
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def vectorize_query(query_text, image_file, text_weight=0.2, image_weight=0.8):
    """
    Vectorizes the user query using CLIP and applies a weighted sum to combine text and image embeddings.
    """
    inputs = {}
    if query_text:
        inputs['text'] = query_text
    if image_file:
        image = process_uploaded_image(image_file)
        if image:
            inputs['images'] = image
        else:
            return None

    if 'images' in inputs and 'text' in inputs:
        preprocessed_inputs = processor(
            text=[inputs['text']], images=inputs['images'], return_tensors="pt", padding=True
        )
    elif 'images' in inputs:
        preprocessed_inputs = processor(images=inputs['images'], return_tensors="pt", padding=True)
    elif 'text' in inputs:
        preprocessed_inputs = processor(text=[inputs['text']], return_tensors="pt", padding=True)
    else:
        return None

    with torch.no_grad():
        outputs = model(**preprocessed_inputs)

    text_emb = outputs.text_embeds.detach().numpy() if 'text' in inputs else None
    image_emb = outputs.image_embeds.detach().numpy() if 'images' in inputs else None

    # Normalize embeddings
    if text_emb is not None:
        text_emb = normalize(text_emb)
    if image_emb is not None:
        image_emb = normalize(image_emb)

    # Apply weighted sum
    if text_emb is not None and image_emb is not None:
        combined_emb = (text_weight * text_emb) + (image_weight * image_emb)
    else:
        combined_emb = text_emb if text_emb is not None else image_emb

    return combined_emb
def search_pinecone(query_embedding, index, top_k=5):
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()
    response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return response['matches']

def generate_response_with_llm(user_query, document_description):
    prompt = f"User Query: {user_query}\nDocument Description: {document_description}\nGenerate a detailed and helpful response:"
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a highly knowledgeable assistant."},
            {"role": "user", "content": prompt},
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.9,
        top_p=0.99,
        max_tokens=300
    )
    return chat_completion.choices[0].message.content

@app.route('/')
def home():
    return render_template('indexes.html')

@app.route('/results', methods=['POST'])
def results():
    query_text = request.form.get('query_text')
    query_image = request.files.get('query_image')

    query_embedding = vectorize_query(query_text, query_image)
    if query_embedding is None:
        return "Error processing query. Please try again."

    results = search_pinecone(query_embedding, index, top_k=1)
    if not results:
        return "No results found."

    top_result = results[0]
    document_description = top_result.get("metadata", {}).get("description", "No description available.")
    response = generate_response_with_llm(query_text, document_description)

    return render_template('results.html', response=response, description=document_description)

if __name__ == '__main__':
    app.run(debug=True)
