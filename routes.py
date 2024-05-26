from flask import Blueprint, request, jsonify
from transformers import pipeline
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import Config

routes = Blueprint('routes', __name__)

# Load configuration data
config_data = json.load(open("config.json"))
HF_TOKEN = config_data["HF_TOKEN"]

model_name = Config.MODEL_NAME

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    token=HF_TOKEN
)

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=Config.MAX_TOKENS
)

def get_response(prompt):
    sequences = text_generator(prompt, temperature=Config.TEMPERATURE)
    gen_text = sequences[0]["generated_text"]
    return gen_text

@routes.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    response = get_response(prompt)
    return jsonify({"response": response})
