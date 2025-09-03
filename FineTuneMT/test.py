# Import Needed Libraries
import json
import torch
import os

import numpy as np

from datasets import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    NllbTokenizer,
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    pipeline,
)
from transformers.pipelines.pt_utils import KeyDataset
import argparse

# Read CLI Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, choices=['mbart50', 'm2m100', 'nllb200', 'helsinki'], required=True, dest='model')
parser.add_argument('-l', '--lang', type=str, required=True, dest='lang')
parser.add_argument('-tf', '--test_file', type=str, required=True, dest='test_file')
parser.add_argument('-gf', '--gen_file', type=str, required=True, dest='gen_file')
args = parser.parse_args()


# Load Model and Tokenizer
dsdir = os.environ['DSDIR']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

lang_mapping = {}

if args.model == 'mbart50':

    if args.lang not in ['en', 'es', 'fr', 'it', 'pt', 'ru', 'zh']:
        raise Exception('Model-Language Pair not supported')
    
    save_path = f'models/mbart50-rdf2{args.lang}'
    model = MBartForConditionalGeneration.from_pretrained(
        f'{dsdir}/HuggingFace_Models/facebook/mbart-large-50',
        low_cpu_mem_usage=True,
        device_map=device,
        cache_dir='cache'
    )

    lang_mapping = {
        'en':'en_XX',
        'es':'es_XX',
        'fr':'fr_XX',
        'it':'it_IT',
        'pt':'pt_XX',
        'ru':'ru_RU',
        'zh':'zh_CN'
    }
    
    tokenizer = MBart50TokenizerFast.from_pretrained(
        f'{dsdir}/HuggingFace_Models/facebook/mbart-large-50',
        src_lang='en_XX',
        tgt_lang=lang_mapping[args.lang],
        cache_dir='cache'
    )

elif args.model == 'm2m100':
    
    if args.lang not in ['br', 'cy', 'en', 'es', 'fr', 'ga', 'it', 'pt', 'ru', 'zh']:
        raise Exception('Model-Language Pair not supported')
    
    save_path = f'models/m2m100-rdf2{args.lang}'
    model = M2M100ForConditionalGeneration.from_pretrained(
        f'{dsdir}/HuggingFace_Models/facebook/m2m100_1.2B',
        low_cpu_mem_usage=True,
        device_map=device,
        cache_dir='cache'
    )
    tokenizer = M2M100Tokenizer.from_pretrained(
        f'{dsdir}/HuggingFace_Models/facebook/m2m100_1.2B',
        src_lang='en',
        tgt_lang=args.lang,
        cache_dir='cache'
    )

elif args.model == 'nllb200':

    if args.lang not in ['cy', 'en', 'es', 'fr', 'ga', 'it', 'mt', 'pt', 'ru', 'zh']:
        raise Exception('Model-Language Pair not supported')
        
    save_path = f'models/nllb200-rdf2{args.lang}'
    model = AutoModelForSeq2SeqLM.from_pretrained(
        f'{dsdir}/HuggingFace_Models/facebook/nllb-200-distilled-1.3B',
        low_cpu_mem_usage=True,
        device_map=device,
        cache_dir='cache'
    )

    lang_mapping = {
        'cy':'cym_Latn',
        'en':'eng_Latn',
        'es':'spa_Latn',
        'fr':'fra_Latn',
        'ga':'gle_Latn',
        'it':'ita_Latn',
        'mt':'mlt_Latn',
        'pt':'por_Latn',
        'ru':'rus_Cyrl',
        'zh':'zho_Hans'
    }
    
    tokenizer = NllbTokenizer.from_pretrained(
        f'{dsdir}/HuggingFace_Models/facebook/nllb-200-distilled-1.3B',
        src_lang='eng_Latn',
        tgt_lang=lang_mapping[args.lang],
        cache_dir='cache'
    )

elif args.model == 'helsinki':
    save_path = f'models/helsinki-rdf2{args.lang}'
    
    if args.lang in ['de', 'en', 'es', 'fr', 'mt', 'zh']:
        model_path = f'opus-mt-en-{args.lang}'
    elif args.lang in ['pt']:
        model_path = f'opus-mt-tc-big-en-{args.lang}'
    elif args.lang in ['br', 'cy', 'ga']:
        model_path = f'opus-mt-en-CELTIC'
    else:
        raise Exception('Model-Language Pair not supported')

    if args.lang != 'pt':
        model = AutoModelForSeq2SeqLM.from_pretrained(
            f'{dsdir}/HuggingFace_Models/Helsinki-NLP/{model_path}',
            low_cpu_mem_usage=True,
            device_map=device,
            cache_dir='cache'
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            f'{dsdir}/HuggingFace_Models/Helsinki-NLP/opus-mt-tc-big-en-pt',
            cache_dir='cache'
        ).to_empty(device=device)
    tokenizer = AutoTokenizer.from_pretrained(
        f'{dsdir}/HuggingFace_Models/Helsinki-NLP/{model_path}',
        low_cpu_mem_usage=True,
        src_lang='en',
        tgt_lang=args.lang,
        cache_dir='cache',
    )

else:
    raise Exception('Model not supported')

# Define Linearization Code
def linearize_tripleset(tripleset, clean=True, subj_token='<S>', pred_token='<P>', obj_token='<O>'):
    linearization = ''
    for triple in tripleset:
        subj = str(triple['subject']).strip()
        prop = str(triple['property']).strip()
        obj = str(triple['object']).strip()
        linearization += f'{subj_token} {subj} {pred_token} {prop} {obj_token} {obj} '
    if clean:
        linearization = linearization.replace('"','').replace('_',' ')
    while '  ' in linearization:
        linearization = linearization.replace('  ', ' ')
    linearization = linearization.strip()
    return linearization

def load_file(file_path):
    formatted_data = {
        'input_text':[],
        'target_text':[]
    }
    
    with open(file_path) as file:
        data = json.load(file)
    
    for entry in tqdm(data['entries']):
        if args.lang in entry['lexicalisations']:
            input_text = linearize_tripleset(entry['modifiedtripleset'])
            if args.model == 'helsinki':
                if args.lang == 'po':
                    input_text = '>>pob<< '+input_text
                elif args.lang in ['br', 'cy', 'ga']:
                    input_text == f'>>{args.lang}<< '+input_text
            for lex in entry['lexicalisations'][args.lang]:
                formatted_data['input_text'].append(input_text)
                formatted_data['target_text'].append(lex['lex'])
        elif args.lang not in entry['labels']:
            raise Exception('Language not present in training file')

    return formatted_data

formatted_test_data = load_file(args.test_file)
print('formatted_test_data:',len(formatted_test_data['input_text']))

# Tokenize Datasets

def tokenize_dataset(sample):
    return tokenizer(sample['input_text'], return_tensors='pt', truncation=True)

# Prepare DataSet
test_ds = Dataset.from_dict(formatted_test_data).map(tokenize_dataset, remove_columns=['input_text', 'target_text'])

with open(args.test_file) as file:
    data = json.load(file)

output = {'entries':[]}
for entry, sample in tqdm(zip(data['entries'], test_ds), total=len(test_ds)):
    target_token_id = tokenizer(lang_mapping.get(args.lang, args.lang), add_special_tokens=False)['input_ids'][0]

    input_ids = torch.tensor(sample['input_ids']).to(device)
    attention_mask = torch.tensor(sample['attention_mask']).to(device)
    
    if args.model == 'nllb200' or args.model =='m2m100':
        translated_tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, forced_bos_token_id=target_token_id)
    elif args.model == 'mbart50':
        translated_tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, decoder_start_token_id=target_token_id)
    else:
        translated_tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask)
    lexicalization = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    new_entry = entry.copy()
    new_entry['lexicalisations'] = {args.lang:[lexicalization]}
    output['entries'].append(new_entry)

with open(args.gen_file, 'w', encoding='utf-8') as file:
    json.dump(output, file, indent=4, ensure_ascii=False)