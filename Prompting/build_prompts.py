from dataclasses import dataclass
from typing import List, Dict
import re
from sentence_transformers import SentenceTransformer
import faiss
from bitarray import bitarray
import json
import random

lang_dict = {}

@dataclass
class Entry:
    idx: int
    props: List[str]
    lex: List[str]
    tripleset: List[Dict]
    size: int
    descriptions: Dict
    labels: Dict
    bitset: bitarray = None
    overlap: int = 0
    
    def __eq__(self, other):
        if isinstance(other, Entry):
            return self.idx == other.idx
        return False

    def __hash__(self):
        return hash(self.idx)

class PromptBuilder:
    def __init__(self, args):
        self.args = args
        self.data = self.load_data()
        self.all_props = set()
        self.formatted_data, self.max_size = self.build_datasets()
        self.model = SentenceTransformer('LaBSE')
        self.prop_to_index = {prop: i for i, prop in enumerate(self.all_props)}
        self.index = self.create_index()
        self.bitset = self.build_bitset()

    def load_data(self):
        try:
            with open(self.args.fewshot_source, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading fewshot source file: {e}")
            return None

    def simplify_descriptions(self, descriptions):
        for lang in descriptions:
            for prop in descriptions[lang]:
                text = re.sub(r'\(.*?\)', '', descriptions[lang][prop])
                descriptions[lang][prop] = re.split(r'[.;:]', text)[0].strip()
        return descriptions

    def process_entry(self, entry, idx=-1):
        props = [triple['property'] for triple in entry['modifiedtripleset']]
        if idx != -1:
            self.all_props.update(props)
        size = len(props)
        try:
            lexicalisation = entry['lexicalisations'][self.args.fewshot_lang][0]['lex']
        except:
            lexicalisation = ""
        processed_entry = Entry(
            idx=idx,
            props=props,
            lex=lexicalisation,
            tripleset=entry['modifiedtripleset'],
            size=size,
            descriptions=self.simplify_descriptions(entry.get('descriptions', {})),
            labels=entry.get('labels', {}).get(self.args.target_lang, {})
        )
        return processed_entry

    def build_datasets(self):
        formatted_data = []
        max_size = 0
        for idx, entry in enumerate(self.data['entries']):
            processed_entry = self.process_entry(entry, idx=idx)
            if max_size < processed_entry.size: max_size = processed_entry.size
            formatted_data.append(processed_entry)

        self.all_props = list(self.all_props)
        return formatted_data, max_size

    def create_index(self):
        embeddings = self.model.encode(self.all_props)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def build_bitset(self):
        for entry in self.formatted_data:
            bitset = bitarray(len(self.all_props))
            bitset.setall(0)
            for prop in entry.props:
                index = self.prop_to_index.get(prop)
                bitset[index] = 1
            entry.bitset = bitset

    def update_overlaps(self, new_properties):
        search_bitset = bitarray(len(self.all_props))
        search_bitset.setall(0)
        for prop in new_properties:
            index = self.prop_to_index.get(prop)
            if index is not None:
                search_bitset[index] = 1
        
        for entry in self.formatted_data:
            entry.overlap = (entry.bitset & search_bitset).count()

    def get_similar_props(self, search_properties):
        search_embeddings = []
        for prop in search_properties:
            search_embeddings.append(self.model.encode([prop]))

        semantic_similar = []

        for embedding in search_embeddings:
            D, I = self.index.search(embedding, 1)
            semantic_similar.extend([self.all_props[i] for i in I[0]])

        return semantic_similar

    def get_fewshot_examples_overlap(self, property_list):
        seen_properties = []
        unseen_properties = []
        for prop in property_list:
            if prop in self.all_props:
                seen_properties.append(prop)
            else:
                unseen_properties.append(prop)
        similar_properties = self.get_similar_props(unseen_properties)

        original_search_properties = list(set(seen_properties + similar_properties))

        self.update_overlaps(original_search_properties)

        original_size = len(property_list)
        size_order = [original_size]
        for i in range(1, self.max_size - original_size + 1):
            size_order.append(original_size + i)
        for i in range(1, original_size):
            size_order.append(original_size - i)

        size_rank = {size: rank for rank, size in enumerate(size_order)}

        def sorting_key(e):
            overlap_rank = -e.overlap
            size_preference = size_rank.get(e.size, float('inf'))
            return (overlap_rank, size_preference)

        sorted_entries = sorted(self.formatted_data, key=sorting_key)

        search_properties = original_search_properties[:]
        selected_examples = []

        while len(selected_examples) < self.args.num_fewshot:
            for entry in sorted_entries:
                if entry not in selected_examples:
                    selected_examples.append(entry)
                    if list(set(search_properties) - set(entry.props)) == list(set(search_properties)):
                        search_properties = original_search_properties[:]
                    break
            search_properties = list(set(search_properties) - set(selected_examples[-1].props))
            if not search_properties:
                search_properties = original_search_properties[:]
            self.update_overlaps(search_properties)
            sorted_entries = sorted(self.formatted_data, key=sorting_key)

        return selected_examples

    def get_fewshot_examples_size(self, property_list):
        selected_examples = []
        target_size = len(property_list)

        sorted_entries = self.formatted_data[:]
        random.shuffle(sorted_entries)

        sorted_entries = sorted(
            sorted_entries,
            key=lambda e: (abs(e.size - target_size), e.size)
        )

        for entry in sorted_entries:
            if len(selected_examples) >= self.args.num_fewshot:
                break
            selected_examples.append(entry)

        return selected_examples

    def get_fewshot_examples_random(self, property_list):
        sizes = []
        if self.max_size == 1:
            return [1] * self.args.num_fewshot
        else:
            step = (self.max_size - 1) / (self.args.num_fewshot - 1)
            sizes = [int(round(1 + i * step)) for i in range(self.args.num_fewshot)]

        selected_examples = []

        sorted_entries = self.formatted_data[:]
        random.shuffle(sorted_entries)

        for size in sizes:
            sorted_entries = sorted(
                sorted_entries,
                key=lambda e: (abs(e.size - size), e.size)
            )
            for entry in sorted_entries:
                if entry not in selected_examples:
                    selected_examples.append(entry)
                    break
        return selected_examples

                
    def create_user_prompt(self, entry):
        user_message = {}
        if self.args.descriptions == 'all':
            descriptions = entry.descriptions['en'].copy()
            for key in entry.descriptions['en']:
                if key in entry.descriptions[self.args.target_lang] and self.args.target_lang != 'en':
                    descriptions[key] += ". " + entry.descriptions[self.args.target_lang][key]
            user_message['descriptions'] = descriptions
        if self.args.descriptions == 'properties':
            descriptions = entry.descriptions['en'].copy()
            for key in entry.descriptions['en']:
                if key in entry.descriptions[self.args.target_lang] and self.args.target_lang != 'en':
                    descriptions[key] += ". " + entry.descriptions[self.args.target_lang][key]
            property_descriptions = {key : descriptions[key] for key in descriptions if key in entry.props}
            user_message['property-descriptions'] = property_descriptions
        if self.args.labels:
            user_message['labels'] = json.dumps(entry.labels, indent=4, ensure_ascii=False)

        tripleset_data = []

        for idx, triple in enumerate(entry.tripleset):
            tripleset_data.append({'id': idx, 'subject': triple['subject'], 'property': triple['property'], 'object': triple['object']})
        user_message['data'] = tripleset_data

        return user_message

    def get_fewshots(self, properties):
        if self.args.fewshot_style == 'overlap':
            return self.get_fewshot_examples_overlap(properties)
        elif self.args.fewshot_style == 'size':
            return self.get_fewshot_examples_size(properties)
        elif self.args.fewshot_style == 'random':
            return self.get_fewshot_examples_random(properties)
        else:
            print('An unavailable fewshot style was specified.')
            return []

    def build_prompt_dir(self, entry, entry_processed=False):
        if entry_processed:
            processed_entry = entry
        else:
            processed_entry = self.process_entry(entry)
        fewshots = self.get_fewshots(processed_entry.props)
        
        messages = []

        for fewshot in fewshots:
            user_message = self.create_user_prompt(fewshot)

            messages.append({"role": "user",
                             "content": json.dumps(user_message, indent=4, ensure_ascii=False).replace('_', ' ') + "\n\nGenerate a lexicalisation of all the " + str(len(fewshot.props)) + f" triples in {self.args.fewshot_language}.\nWhen necessary, generate text from other languages in their original script.\nInclude all the information from the triples regardless of their type or relevance."})

            assistant_message = {"full-text": fewshot.lex}

            messages.append({"role": "assistant",
                             "content": json.dumps(assistant_message, indent=4, ensure_ascii=False)})

        final_message = self.create_user_prompt(processed_entry)

        messages.append({"role": "user",
                         "content": json.dumps(final_message, indent=4, ensure_ascii=False).replace('_', ' ') + "\n\nGenerate a lexicalisation of all the " + str(len(processed_entry.props)) + " triples in " + self.args.target_language + ".\nWhen necessary, generate text from other languages in their original script.\nInclude all the information from the triples regardless of their type or relevance."})

        return messages

    def build_prompt_cot(self, entry):
        processed_entry = self.process_entry(entry)
        fewshots = self.get_fewshots(processed_entry.props)
        
        messages = []

        for fewshot in fewshots:
            user_message = self.create_user_prompt(fewshot)

            messages.append({"role": "user",
                             "content": json.dumps(user_message, indent=4, ensure_ascii=False).replace('_', ' ') + "\n\nGenerate a lexicalisation of all the " + str(len(fewshot.props)) + f" triples in {self.args.fewshot_language}.\nWhen necessary, generate text from other languages in their original script.\nInclude all the information from the triples regardless of their type or relevance."})

            individual_lexes = []
            for idx, triple in enumerate(fewshot.tripleset):
                individual_lexes.append({'id': idx, 'lex': triple['lex']})

            assistant_message = {"individual-lexicalisations": individual_lexes, 
                                 "full-text": fewshot.lex}

            messages.append({"role": "assistant",
                             "content": json.dumps(assistant_message, indent=4, ensure_ascii=False)})

        final_message = self.create_user_prompt(processed_entry)

        messages.append({"role": "user",
                         "content": json.dumps(final_message, indent=4, ensure_ascii=False).replace('_', ' ') + "\n\nGenerate a lexicalisation of all the " + str(len(processed_entry.props)) + " triples in " + self.args.target_language + ".\nWhen necessary, generate text from other languages in their original script.\nInclude all the information from the triples regardless of their type or relevance."})

        return messages

    def build_prompt_cml_pre(self, entry):
        processed_entry = self.process_entry(entry)
        
        messages_collection = []

        for idx, tripleset in enumerate(processed_entry.tripleset):
            individual_entry = Entry(
                idx=idx,
                props=[tripleset['property']],
                lex='',
                tripleset=[tripleset],
                size=1,
                descriptions = {
                    "en": {
                        key: processed_entry.descriptions.get('en', {}).get(key, '')
                        for key in [tripleset['subject'], tripleset['property'], tripleset['object']]
                        if key in processed_entry.descriptions.get('en', {})
                    },
                    self.args.target_lang: {
                        key: processed_entry.descriptions.get(self.args.target_lang, {}).get(key, '')
                        for key in [tripleset['subject'], tripleset['property'], tripleset['object']]
                        if key in processed_entry.descriptions.get(self.args.target_lang, {})
                    }
                },

                labels = {
                    "en": {
                        key: processed_entry.labels.get('en', {}).get(key, '')
                        for key in [tripleset['subject'], tripleset['property'], tripleset['object']]
                        if key in processed_entry.labels.get('en', {})
                    },
                    self.args.target_lang: {
                        key: processed_entry.labels.get(self.args.target_lang, {}).get(key, '')
                        for key in [tripleset['subject'], tripleset['property'], tripleset['object']]
                        if key in processed_entry.labels.get(self.args.target_lang, {})
                    }
                }
            )
            messages_collection.append(self.build_prompt_dir(individual_entry, entry_processed=True))
        return messages_collection

        

    def build_prompt_cml_post(self, entry, lexes):
        processed_entry = self.process_entry(entry)
        fewshots = self.get_fewshots(processed_entry.props)

        messages = []

        for fewshot in fewshots:

            lexicalisations = []

            for idx, triple in enumerate(fewshot.tripleset):
                lexicalisations.append({"idx": idx, "lex": triple["lex"]})

            user_message = {"sentences": lexicalisations}

            messages.append({"role": "user",
                             "content": json.dumps(user_message, indent=4, ensure_ascii=False).replace('_', ' ') + "\n\nGenerate a combined paragraph of all the " + str(len(fewshot.props)) + f" sentences in {self.args.fewshot_language}.\nInclude all the information from the sentences regardless of their type or relevance."})


            assistant_message = {"full-text": fewshot.lex}

            messages.append({"role": "assistant",
                             "content": json.dumps(assistant_message, indent=4, ensure_ascii=False)})

        lexicalisations = []
        for idx, lex in enumerate(lexes):
            lexicalisations.append({"idx": idx, "lex": lex})
        
        final_message = {"sentences": lexicalisations}

        messages.append({"role": "user",
                "content": json.dumps(final_message, indent=4, ensure_ascii=False).replace('_', ' ') + "\n\nGenerate a combined paragraph of all the " + str(len(fewshot.props)) + " sentences in " + self.args.target_language + ".\nInclude all the information from the sentences regardless of their type or relevance."})

        return messages
        