# Llama 8B

## 472
python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/en_train+dev.json --output_dir generations --model_cache .cache --target_lang en --lang_code en --fewshot_lang en --fewshot_style overlap --generation_technique dir --descriptions none --model_type llama3 --model_size 8b --output_filename llama3_8b_472_en

python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/fr_train+dev.json --output_dir generations --model_cache .cache --target_lang fr --lang_code fr --fewshot_lang fr --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 8b --output_filename llama3_8b_472_fr

 python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/ru_train+dev.json --output_dir generations --model_cache .cache --target_lang ru --lang_code ru --fewshot_lang ru --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 8b --output_filename llama3_8b_472_ru

python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/es_train+dev.json --output_dir generations --model_cache .cache --target_lang es --lang_code es --fewshot_lang es --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 8b --output_filename llama3_8b_472_es

python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/es_train+dev.json --output_dir generations --model_cache .cache --target_lang zh --lang_code zh --fewshot_lang zh --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 8b --output_filename llama3_8b_472_zh

## WebNLG
python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/br_train+dev.json --output_dir generations --model_cache .cache --target_lang br --lang_code br --fewshot_lang br --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 8b --output_filename llama3_8b_webnlg_br

python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/cy_train+dev.json --output_dir generations --model_cache .cache --target_lang cy --lang_code cy --fewshot_lang cy --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 8b --output_filename llama3_8b_webnlg_cy

python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/en_train+dev.json --output_dir generations --model_cache .cache --target_lang en --lang_code en --fewshot_lang en --fewshot_style overlap --generation_technique dir --descriptions none --model_type llama3 --model_size 8b --output_filename llama3_8b_webnlg_en

python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/ga_train+dev.json --output_dir generations --model_cache .cache --target_lang ga --lang_code ga --fewshot_lang ga --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 8b --output_filename llama3_8b_webnlg_ga

python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/mt_train+dev.json --output_dir generations --model_cache .cache --target_lang mt --lang_code mt --fewshot_lang mt --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 8b --output_filename llama3_8b_webnlg_mt

python -u generate.py --input_file data/expanded-test-webnlg-russian.json --fewshot_source data/ru_train+dev.json --output_dir generations --model_cache .cache --target_lang ru --lang_code ru --fewshot_lang ru --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 8b --output_filename llama3_8b_webnlg_ru

python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/es_train+dev.json --output_dir generations --model_cache .cache --target_lang es --lang_code es --fewshot_lang es --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 8b --output_filename llama3_8b_webnlg_es


# Llama 70B

## 472
python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/en_train+dev.json --output_dir generations --model_cache .cache --target_lang en --lang_code en --fewshot_lang en --fewshot_style overlap --generation_technique dir --descriptions none --model_type llama3 --model_size 70b --output_filename llama3_70b_472_en

python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/fr_train+dev.json --output_dir generations --model_cache .cache --target_lang fr --lang_code fr --fewshot_lang fr --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 70b --output_filename llama3_70b_472_fr

python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/ru_train+dev.json --output_dir generations --model_cache .cache --target_lang ru --lang_code ru --fewshot_lang ru --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 70b --output_filename llama3_70b_472_ru

python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/es_train+dev.json --output_dir generations --model_cache .cache --target_lang es --lang_code es --fewshot_lang es --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 70b --output_filename llama3_70b_472_es

python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/es_train+dev.json --output_dir generations --model_cache .cache --target_lang zh --lang_code zh --fewshot_lang zh --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 70b --output_filename llama3_70b_472_zh

## WebNLG
python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/br_train+dev.json --output_dir generations --model_cache .cache --target_lang br --lang_code br --fewshot_lang br --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 70b --output_filename llama3_70b_webnlg_br

python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/cy_train+dev.json --output_dir generations --model_cache .cache --target_lang cy --lang_code cy --fewshot_lang cy --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 70b --output_filename llama3_70b_webnlg_cy

python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/en_train+dev.json --output_dir generations --model_cache .cache --target_lang en --lang_code en --fewshot_lang en --fewshot_style overlap --generation_technique dir --descriptions none --model_type llama3 --model_size 70b --output_filename llama3_70b_webnlg_en

 python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/ga_train+dev.json --output_dir generations --model_cache .cache --target_lang ga --lang_code ga --fewshot_lang ga --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 70b --output_filename llama3_70b_webnlg_ga

python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/mt_train+dev.json --output_dir generations --model_cache .cache --target_lang mt --lang_code mt --fewshot_lang mt --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 70b --output_filename llama3_70b_webnlg_mt

python -u generate.py --input_file data/expanded-test-webnlg-russian.json --fewshot_source data/ru_train+dev.json --output_dir generations --model_cache .cache --target_lang ru --lang_code ru --fewshot_lang ru --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 70b --output_filename llama3_70b_webnlg_ru

python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/es_train+dev.json --output_dir generations --model_cache .cache --target_lang es --lang_code es --fewshot_lang es --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type llama3 --model_size 70b --output_filename llama3_70b_webnlg_es

# Qwen 7B

## 472
python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/en_train+dev.json --output_dir generations --model_cache .cache --target_lang en --lang_code en --fewshot_lang en --fewshot_style overlap --generation_technique dir --descriptions none --model_type qwen2.5 --model_size 7b --output_filename qwen2.5_7b_472_en

python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/fr_train+dev.json --output_dir generations --model_cache .cache --target_lang fr --lang_code fr --fewshot_lang fr --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 7b --output_filename qwen2.5_7b_472_fr

python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/ru_train+dev.json --output_dir generations --model_cache .cache --target_lang ru --lang_code ru --fewshot_lang ru --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 7b --output_filename qwen2.5_7b_472_ru

python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/es_train+dev.json --output_dir generations --model_cache .cache --target_lang es --lang_code es --fewshot_lang es --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 7b --output_filename qwen2.5_7b_472_es

python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/es_train+dev.json --output_dir generations --model_cache .cache --target_lang zh --lang_code zh --fewshot_lang zh --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 7b --output_filename qwen2.5_7b_472_zh

## WebNLG
python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/br_train+dev.json --output_dir generations --model_cache .cache --target_lang br --lang_code br --fewshot_lang br --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 7b --output_filename qwen2.5_7b_webnlg_br

python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/cy_train+dev.json --output_dir generations --model_cache .cache --target_lang cy --lang_code cy --fewshot_lang cy --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 7b --output_filename qwen2.5_7b_webnlg_cy

python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/en_train+dev.json --output_dir generations --model_cache .cache --target_lang en --lang_code en --fewshot_lang en --fewshot_style overlap --generation_technique dir --descriptions none --model_type qwen2.5 --model_size 7b --output_filename qwen2.5_7b_webnlg_en

python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/ga_train+dev.json --output_dir generations --model_cache .cache --target_lang ga --lang_code ga --fewshot_lang ga --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 7b --output_filename qwen2.5_7b_webnlg_ga

python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/mt_train+dev.json --output_dir generations --model_cache .cache --target_lang mt --lang_code mt --fewshot_lang mt --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 7b --output_filename qwen2.5_7b_webnlg_mt

python -u generate.py --input_file data/expanded-test-webnlg-russian.json --fewshot_source data/ru_train+dev.json --output_dir generations --model_cache .cache --target_lang ru --lang_code ru --fewshot_lang ru --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 7b --output_filename qwen2.5_7b_webnlg_ru

python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/es_train+dev.json --output_dir generations --model_cache .cache --target_lang es --lang_code es --fewshot_lang es --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 7b --output_filename qwen2.5_7b_webnlg_es

# Qwen 72B

## 472
python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/en_train+dev.json --output_dir generations --model_cache .cache --target_lang en --lang_code en --fewshot_lang en --fewshot_style overlap --generation_technique dir --descriptions none --model_type qwen2.5 --model_size 72b --output_filename qwen2.5_72b_472_en

python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/fr_train+dev.json --output_dir generations --model_cache .cache --target_lang fr --lang_code fr --fewshot_lang fr --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 72b --output_filename qwen2.5_72b_472_fr

python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/ru_train+dev.json --output_dir generations --model_cache .cache --target_lang ru --lang_code ru --fewshot_lang ru --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 72b --output_filename qwen2.5_72b_472_ru

python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/es_train+dev.json --output_dir generations --model_cache .cache --target_lang es --lang_code es --fewshot_lang es --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 72b --output_filename qwen2.5_72b_472_es

python -u generate.py --input_file data/expanded-test-472-full.json --fewshot_source data/es_train+dev.json --output_dir generations --model_cache .cache --target_lang zh --lang_code zh --fewshot_lang zh --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 72b --output_filename qwen2.5_72b_472_zh

## WebNLG
python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/br_train+dev.json --output_dir generations --model_cache .cache --target_lang br --lang_code br --fewshot_lang br --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 72b --output_filename qwen2.5_72b_webnlg_br

python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/cy_train+dev.json --output_dir generations --model_cache .cache --target_lang cy --lang_code cy --fewshot_lang cy --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 72b --output_filename qwen2.5_72b_webnlg_cy

python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/en_train+dev.json --output_dir generations --model_cache .cache --target_lang en --lang_code en --fewshot_lang en --fewshot_style overlap --generation_technique dir --descriptions none --model_type qwen2.5 --model_size 72b --output_filename qwen2.5_72b_webnlg_en

python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/ga_train+dev.json --output_dir generations --model_cache .cache --target_lang ga --lang_code ga --fewshot_lang ga --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 72b --output_filename qwen2.5_72b_webnlg_ga

python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/mt_train+dev.json --output_dir generations --model_cache .cache --target_lang mt --lang_code mt --fewshot_lang mt --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 72b --output_filename qwen2.5_72b_webnlg_mt

python -u generate.py --input_file data/expanded-test-webnlg-russian.json --fewshot_source data/ru_train+dev.json --output_dir generations --model_cache .cache --target_lang ru --lang_code ru --fewshot_lang ru --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 72b --output_filename qwen2.5_72b_webnlg_ru

python -u generate.py --input_file data/expanded-test-webnlg-full.json --fewshot_source data/es_train+dev.json --output_dir generations --model_cache .cache --target_lang es --lang_code es --fewshot_lang es --fewshot_style overlap --generation_technique dir --descriptions none --labels 1 --model_type qwen2.5 --model_size 72b --output_filename qwen2.5_72b_webnlg_es
