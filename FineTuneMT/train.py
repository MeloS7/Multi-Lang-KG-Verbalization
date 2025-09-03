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
DataCollatorForSeq2Seq
)
import argparse

# Read CLI Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, choices=['mbart50', 'm2m100', 'nllb200', 'helsinki'], required=True, dest='model')
parser.add_argument('-l', '--lang', type=str, required=True, dest='lang')
parser.add_argument('-tf', '--train_file', type=str, required=True, dest='train_file')
parser.add_argument('-df', '--dev_file', type=str, required=True, dest='dev_file')
parser.add_argument('-e', '--epochs', type=int, default=2, dest='epochs')
args = parser.parse_args()


# Load Model and Tokenizer
dsdir = os.environ['DSDIR']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

if args.model in ['nllb200', 'mbart50', 'helsinki']:
    # Add New Tokens
    special_tokens_dict = {'additional_special_tokens': ['<S>', '<P>', '<O>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

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
        else:
            raise Exception('Language not present in training file')

    return formatted_data

formatted_train_data = load_file(args.train_file)
print('formatted_train_data:',len(formatted_train_data['input_text']))

# Load Validation Data
formatted_dev_data = load_file(args.dev_file)
print('formatted_dev_data:',len(formatted_dev_data['input_text']))

# Tokenize Datasets

def tokenize_dataset(sample):
    tokenized = tokenizer(sample['input_text'], text_target=sample['target_text'], truncation=True)
    return {
        'input_ids':tokenized['input_ids'],
        'labels':tokenized['labels']
    }

# Prepare Training DataSet
train_ds = Dataset.from_dict(formatted_train_data).map(tokenize_dataset, remove_columns=['input_text', 'target_text'])

# Prepare Validation DataSet
dev_ds = Dataset.from_dict(formatted_dev_data).map(tokenize_dataset, remove_columns=['input_text', 'target_text'])

# Set Training Arguments
trainer_args = Seq2SeqTrainingArguments(
    save_path,
    eval_strategy='steps',
    eval_steps=10000,
    save_strategy='steps',
    save_steps=10000,
    logging_strategy='steps',
    logging_steps=10000,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    save_total_limit=3,
    load_best_model_at_end=True,
    num_train_epochs=args.epochs,
    include_inputs_for_metrics=False,
    predict_with_generate=False,
    optim='adafactor',
    metric_for_best_model='loss',
    report_to='none',
    weight_decay=0.1,
    learning_rate=2e-5,
    save_safetensors=False,
)

# Prepare Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=trainer_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    tokenizer=tokenizer,
)

# Train Model
trainer.train()

# Save Model
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
