from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

# بارگذاری مدل و توکنایزر
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_response(prompt, persona, drama_level):
    input_text = f"{persona}: {prompt} [Drama level: {drama_level}]"
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, temperature=1.0)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data['prompt']
    persona = data['persona']
    drama_level = data['drama_level']
    response = generate_response(prompt, persona, drama_level)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='45.156.184.37', port=8080)
