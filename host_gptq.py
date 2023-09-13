import sys
import re
from typing import List

import torch
from flask import Flask, request, jsonify
from transformers import (AutoModelForCausalLM, AutoTokenizer, 
                          StoppingCriteria, StoppingCriteriaList)

from auto_gptq import exllama_set_max_input_length

# Retrieve command line arguments
model_name_or_path, local_port, gpuid = sys.argv[1], sys.argv[2], str(sys.argv[3])

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, 
    use_fast=False, 
    local_files_only=True, 
    trust_remote_code=True
)

class StoppingCriteriaSub(StoppingCriteria):
    """Custom Stopping Criteria based on provided stop tokens."""
    def __init__(self, stops=[]):
        super().__init__()
        self.stops = [stop.to(f"cuda:{gpuid}") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        return any(torch.all((stop == input_ids[0][-len(stop):])).item() for stop in self.stops)

def convert_stopwords_to_ids(stopwords: List[str]):
    """Convert stopwords to their respective IDs."""
    return StoppingCriteriaList([
        StoppingCriteriaSub(
            [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stopwords]
        )
    ])

# Initialize model
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, 
    local_files_only=True, 
    torch_dtype=torch.float16, 
    trust_remote_code=True, 
    device_map=f"cuda:{gpuid}"
)

if 'Speechless-Llama2-13B-GPTQ' in model_name_or_path:
    model = exllama_set_max_input_length(model, 4096)

def generate_output(text, num_responses, max_new_tokens, temperature, top_p, top_k, repetition_penalty, stop_tokens):
    """Generate output text based on the provided input."""
    tokens = model.generate(
        tokenizer(text, return_tensors="pt").input_ids.to(f"cuda:{gpuid}"), 
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        top_p=top_p, 
        top_k=top_k, 
        repetition_penalty=repetition_penalty, 
        stopping_criteria=convert_stopwords_to_ids(stop_tokens), 
        do_sample=True, 
        num_return_sequences=num_responses
    )
    return [tokenizer.decode(token, skip_special_tokens=True) for token in tokens]

# Initialize Flask app
app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_text():
    """Flask endpoint to generate text."""
    data, num_responses = request.json, 3
    responses = [
        response.replace(re.sub('<\|.*?\|>', '', data['prompt']), "") 
        for response in generate_output(data['prompt'], num_responses, 200, 0.9, 1.0, 60, 1.0, [])
    ]
    for i, stop in enumerate(data.get('stopwords', [])):
        responses[i] = responses[i].replace(stop, "")
    
    return jsonify({'response': responses, "model": model_name_or_path})

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=local_port)
