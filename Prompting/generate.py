#Imports
import argparse
import json
import re
from tqdm import tqdm
from load_model import load_model
from build_prompts import PromptBuilder
import os
import torch

def run_model(args, model, messages):
    prompt = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    eos_token_ids = [model.tokenizer.eos_token_id]
    if args.model_type == 'llama3':
        eos_token_ids.append(model.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    
    with torch.inference_mode():
        outputs = model(
            prompt,
            do_sample=True,
            eos_token_id=eos_token_ids,
            **args.gen_args
        )

    prediction = outputs[0]["generated_text"][len(prompt):]

    try:
        p = json.loads(prediction)
        return p
    except:
        pattern = r'full-text(.*?)\}'

        match = re.search(pattern, prediction, re.DOTALL)
        if match:
            captured_text = match.group(1)
        else:
            captured_text = 'ERROR ' + prediction

        captured_text = captured_text.strip(' ":\n') 

        return {'full-text': captured_text}


def save_data(args, data):
    if args.output_filename:
        output_filename = utput_file_name = f'''{args.output_filename}_{args.fewshot_style}_{args.generation_technique}_{args.descriptions}_{args.labels}_{args.target_lang}.json'''
    else:
        output_filename = output_file_name = f'''{args.input_file}_{args.fewshot_style}_{args.generation_technique}_{args.descriptions}_{args.labels}_{args.target_lang}.json'''
    output_file_path = os.path.join(args.output_dir, output_filename)
    with open(output_file_path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"File saved successfully to {output_file_path}.")

def main(args):
    tokenizer, model, device, terminators = load_model(args)
    pb = PromptBuilder(args)

    print('Loading Data...')
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    print('Data Loaded...')

    print('Beginning Generations...')
    loop = tqdm(data['entries'])
    for entry in loop:
        if args.generation_technique == 'dir' or len(entry['modifiedtripleset']) == 1:
            messages = pb.build_prompt_dir(entry)
        elif args.generation_technique == 'cot':
            messages = pb.build_prompt_cot(entry)
        elif args.generation_technique == 'cml':
            lexes = []
            for messages in pb.build_prompt_cml_pre(entry):
                lexes.append(run_model(args, model, messages)["full-text"])
                if args.do_print_check:
                    for message in messages:
                        print(message["role"])
                        print(message['content'])
                    print("Prediction:\n", lexes[-1])
            messages = pb.build_prompt_cml_post(entry, lexes)

        prediction = run_model(args, model, messages)

        if args.do_print_check:
            for message in messages:
                print(message["role"])
                print(message['content'])
            print("Prediction:\n", prediction)

        entry['lexicalisations'] = {}

        if type(prediction) == dict:
            entry['lexicalisations'][args.lang_code] = [{"comment": "", 
                                                         "lex": prediction['full-text']}]
        else:
            entry['lexicalisations'][args.lang_code] = [{"comment": prediction, 
                                                         "lex": ""}]

    return data


if __name__ == '__main__':
    #Main Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='path/to/file')
    parser.add_argument('--fewshot_source', type=str, default='path/to/file')
    parser.add_argument('--output_dir', type=str, default='path/to/output_dir')
    parser.add_argument('--model_cache', type=str, default='path/to/dir')
    parser.add_argument('--hf_token', type=str, default='')
    parser.add_argument('--model_size', type=str, default='8B', help='7B, 8B, 70B, 72B, or 405B')
    parser.add_argument('--model_type', type=str, default='llama3', help='llama3, qwen2.5, or')
    parser.add_argument('--target_lang', type=str, default='en', help='en, ru, es, zh, ar, or fr')

    #Optional Arguments
    parser.add_argument('--output_filename', type=str, default='')
    parser.add_argument('--lang_code', type=str, default='')
    parser.add_argument('--fewshot_lang', type=str, default='en')
    parser.add_argument('--fewshot_style', type=str, default='overlap', help='overlap, size, or random')
    parser.add_argument('--generation_technique', type=str, default='dir', help='dir, cml, or cot')
    parser.add_argument('--descriptions', type=str, default='all', help='all, properties, or none')
    parser.add_argument('--labels', type=bool, default=False)
    parser.add_argument('--num_fewshot', type=int, default=4)
    parser.add_argument('--do_print_check', type=bool, default=False)
    args = parser.parse_args()

    language_dict = {'br':'Breton', 'cy':'Welsh', 'en': 'English', 'fr':'French', 'ga':'Irish', 'it':'Italian', 'mt':'Maltese', 'pt':'Portuguese', 'ru': 'Russian', 'es': 'Spanish', 'zh': 'Simplified Chinese',}

    args.target_language = language_dict[args.target_lang]
    args.fewshot_language = language_dict[args.fewshot_lang]

    if args.lang_code == '':
        args.lang_code = args.target_lang

    #Generation Arguments
    args.gen_args = {'max_new_tokens': 1280, 'temperature' : 0.7, 'top_p' : 0.7}

    #Generate Lexicalisations
    data = main(args)

    #Save the Output
    save_data(args, data)