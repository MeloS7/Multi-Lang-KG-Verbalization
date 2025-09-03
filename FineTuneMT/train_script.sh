## helsinki
python -u train.py -l br -m helsinki -tf data/br_train.json -df data/br_dev.json
python -u train.py -l es -m helsinki -tf data/es_train.json -df data/es_dev.json
python -u train.py -l fr -m helsinki -tf data/fr_train.json -df data/fr_dev.json
python -u train.py -l mt -m helsinki -tf data/mt_train.json -df data/mt_dev.json

## m2m100
python -u train.py -l cy -m m2m100 -tf data/cy_train.json -df data/cy_dev.json
python -u train.py -l ru -m m2m100 -tf data/ru_train.json -df data/ru_dev.json
python -u train.py -l zh -m m2m100 -tf data/zh_train.json -df data/zh_dev.json

## nllb200
python -u train.py -l en -m nllb200 -tf data/en_train.json -df data/en_dev.json
python -u train.py -l ga -m nllb200 -tf data/ga_train.json -df data/ga_dev.json
