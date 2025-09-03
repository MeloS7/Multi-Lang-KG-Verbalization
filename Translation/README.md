# Translation Page


## Text Translation
In this folder, you can find the usage of the following Machine Translation (MT) models that we tested in our paper:
- m2m:
    - facebook/nllb-200-distilled-600M
    - facebook/nllb-200-3.3B
    - facebook/mbart-large-50-many-to-many-mmt
    - facebook/m2m100_418M
    - facebook/m2m100_1.2B
- 121:
    - Helsinki-NLP/opus-mt-en-zh
    - Helsinki-NLP/opus-mt-en-fr
    - Helsinki-NLP/opus-mt-en-es
    - Helsinki-NLP/opus-mt-en-ru
    - Helsinki-NLP/opus-mt-en-ar
    - Helsinki-NLP/opus-mt-en-it
    - Helsinki-NLP/opus-mt-tc-big-en-pt
    - Helsinki-NLP/opus-mt-en-mt
    - Helsinki-NLP/opus-mt-en-ga
    - Helsinki-NLP/opus-mt-en-cy
    - Helsinki-NLP/opus-mt-en-CELTIC

Please check and modify the (config file)[./config/config_translator.yaml] and then run the (source code)[translator.py]. 
The input file should be a .txt file that each line is a sentence to translate, and the output will be a same .txt file with their corresponding translation in target files.

Don't forget to switch MT mode ('m2m' or '121') and the target language when using it. (Note. the abbrevations for langugaes might be differnet across MT models)

## Translation Evaluation
We also provide the cross-lang translation (seq2seq/text2text) evaluation script. Bascially, we use the following automatic metrics:
- NMTScore
    - NMT-Direct
    - NMT-Pivot
    - NMT-Cross_likelihood
- SBERT
- BERTScore
- COMETKiwi ("Unbabel/wmt23-cometkiwi-da-xl")

Here is the format of input JSON file with lists of (original text, translation):
```json
[
    [
        "Aarhus airport serves the city of Aarhus whose leader is Jacob Bundsgaard.",
        "L'a\u00e9roport d'Aarhus dessert la ville d'Aarhus, dont le chef est Jacob Bundsgaard."
    ],
    [
        "Aarhus airport serves the city of Aarhus where Jacob Bundsgaard is a leader.",
        "L'a\u00e9roport d'Aarhus dessert la ville d'Aarhus o\u00f9 Jacob Bundsgaard est un leader."
    ],
    [
        "Aarhus airport serves the city of Aarhus, its leader is Jacob Bundsgaard.",
        "L'a\u00e9roport d'Aarhus dessert la ville d'Aarhus, son leader est Jacob Bundsgaard."
    ],
]
```



