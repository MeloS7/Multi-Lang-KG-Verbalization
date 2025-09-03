import json
import yaml
import pickle
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    MarianMTModel,
)

from data_reader import read_lex

def load_config(config_path='./config/config_translator.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def translate_sentences_m2m(sentences, args):
    # Parse args
    model_name = args['model_name'] if args['model_name'] is not None else args['default_model']
    batch_size = args['model_config'][model_name]['batch_size']
    max_tokens = args['model_config'][model_name]['max_tokens']
    target_lang = args['target_language']
    src_lang = args['model_config'][model_name]['src_lang']
    
    print(f"You are translating from {src_lang} to {target_lang} using {model_name} !!!!")

    # Load tokenizer and model
    if "nllb" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif "mbart" in model_name:
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        tokenizer.src_lang = src_lang
    elif "m2m" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        tokenizer.src_lang = src_lang
    else:
        raise ValueError(f"Unsupported model name: {model_name}.")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    translated_sentences_pair = []

    for i in tqdm(range(0, len(sentences), batch_size)):
        batch_sentences = sentences[i:i+batch_size]

        # Tokenize input sentences and move to GPU if available
        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True).to(device)

        forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
        # Generate translated sentences
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            forced_bos_token_id=forced_bos_token_id,
            max_length=max_tokens
        )
        
        # Decode and remove special tokens
        translated_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated_sentences_pair.extend(zip(batch_sentences, translated_sentences))

    return translated_sentences_pair

def translate_sentences_121(sentences, args):
    # Parse args
    model_name = args['model_name'] if args['model_name'] is not None else args['default_model']
    batch_size = args['model_config'][model_name]['batch_size']
    max_tokens = args['model_config'][model_name]['max_tokens']
    target_lang = args['target_language']
    src_lang = args['model_config'][model_name].get('src_lang', None)
    
    print(f"You are translating from {src_lang} to {target_lang} using {model_name} !!!!")
    
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    translated_sentences_pair = []

    for i in tqdm(range(0, len(sentences), batch_size)):
        batch_sentences = sentences[i:i+batch_size]

        # Tokenize input sentences and move to GPU if available
        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True).to(device)

        # Generate translated sentences
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_tokens
        )
        
        # Decode and remove special tokens
        translated_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated_sentences_pair.extend(zip(batch_sentences, translated_sentences))

    return translated_sentences_pair
    

def main(args):
    # Read golden references from input json file
    lexs = read_lex(args['input_file'], lang="en")
    
    # Get translation
    model_type = args["model_type"]
    if model_type == "m2m":
        translated_sentences_pair = translate_sentences_m2m(lexs, args)
    else:
        translated_sentences_pair = translate_sentences_121(lexs, args)

    # Save pairs to json file
    with open(args['output_file'], 'w') as f:
        json.dump(translated_sentences_pair, f, indent=4)
        
if __name__ == "__main__":
    args = load_config(config_path = './config/config_translator.yaml')

    main(args)
 