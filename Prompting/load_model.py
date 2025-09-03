import torch
import numpy as np
import random
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

def load_model(args):
    print('Loading model...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    attn_implementation = 'flash_attention_2'

    seed = 42

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    padding_side = 'left'

    if args.model_type == 'llama3':
        if args.model_size.lower() == '8b':
            # model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
            # model_id = '/gpfsdswork/dataset/HuggingFace_Models/meta-llama/Meta-Llama-3.1-8B-Instruct'
            model_id = '/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Llama-3.1-8B-Instruct'
        elif args.model_size.lower() == '70b':
            # model_id = 'unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit'
            model_id = '/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Meta-Llama-3-70B-Instruct'
        elif args.model_size.lower() == '405b':
            model_id = 'unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit'
    if args.model_type == 'qwen2.5':
        if args.model_size.lower() == '7b':
            # model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
            # model_id = '/gpfsdswork/dataset/HuggingFace_Models/meta-llama/Meta-Llama-3.1-8B-Instruct'
            model_id = '/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen2.5-7B-Instruct'
        elif args.model_size.lower() == '72b':
            # model_id = 'unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit'
            model_id = '/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen2.5-72B-Instruct'

    cache_dir = args.model_cache + '/' + model_id
    hf_token = args.hf_token if args.hf_token else None

    if not os.path.exists(args.model_cache):
        print('Designated does not exist, creating new cache...')
        os.makedirs(args.model_cache)

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side = padding_side, 
            token = hf_token)
 
    tokenizer.pad_token_id = tokenizer.eos_token_id
                
    if args.model_size.lower() in ['7b', '8b']:
        model = AutoModelForCausalLM.from_pretrained(model_id, 
            cache_dir = cache_dir,
            torch_dtype = torch.bfloat16,
            # attn_implementation = attn_implementation,
            token = hf_token,
            device_map = 'auto')
    elif args.model_size.lower() in ['70b', '72b', '405b']:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, 
            cache_dir = cache_dir,
            torch_dtype = torch.bfloat16,
            # attn_implementation = attn_implementation,
            token = hf_token,
            device_map = 'auto',
            quantization_config=quantization_config)
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    model = pipeline("text-generation", model = model, tokenizer = tokenizer)

    terminators = [
            model.tokenizer.eos_token_id,
            model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    return tokenizer, model, device, terminators