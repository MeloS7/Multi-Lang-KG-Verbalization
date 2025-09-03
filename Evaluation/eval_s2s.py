import json
import os
import yaml
import copy
import nltk

def load_config(config_path='./config_translation_eval.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def read_translation_pair(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def get_src_tar_list(pair_data):
    src_list = []
    tar_list = []
    for pair in pair_data:
        src_list.append(pair[0])
        tar_list.append(pair[1])
        
    return src_list, tar_list

def compute_nmtscore(src_list, tar_lists, args):
    from nmtscore import NMTScorer
    # Define NMTScorer
    scorer = NMTScorer(args["model_name"], device="cuda:0")
    
    res_direct = []
    res_pivot = []
    res_cross = []
    
    # Truncate sentence
    for i,src_sent in enumerate(src_list):
        if len(src_sent) >= 1024:
            src_list[i] = src_sent[:1024]
    
    for tar_list in tar_lists:
        # Direct translation probability
        scores_direct = scorer.score_direct(
            src_list, 
            tar_list, 
            a_lang=args["input_lang"],
            b_lang=args["target_lang"],
            normalize=True, 
            both_directions=True,
            score_kwargs={"batch_size": args["batch_size"]},
        )

        print("Direct scores finished!")

        # Translation cross-likelihood (default)
        scores_cross = scorer.score_cross_likelihood(
            src_list, 
            tar_list, 
            tgt_lang="en", 
            normalize=True, 
            both_directions=True,
            translate_kwargs={"batch_size": args["batch_size"]},
            score_kwargs={"batch_size": args["batch_size"]}
        )

        print("Cross scores finished!")

        # Pivot translation probability
        scores_pivot = scorer.score_pivot(
            src_list, 
            tar_list, 
            a_lang=args["input_lang"],
            b_lang=args["target_lang"],
            pivot_lang="en",
            normalize=True, 
            both_directions=True,
            translate_kwargs={"batch_size": args["batch_size"]},
            score_kwargs={"batch_size": args["batch_size"]}
        )
        print("Pivot scores finished!")
        
        res_direct.append(scores_direct)
        res_cross.append(scores_cross)
        res_pivot.append(scores_pivot)

    return res_direct, res_pivot, res_cross
        
    def keep_max(embed_res):
        # Keep the maximum score for each sentence, and also keep the index of the maximum score
        best_similarities = []
    
        for i in range(len(embed_res[0])):  # Iterate over each element in the diagonal
            # Get the similarity of all tar_lists at index i and take the max
            max_similarity = max(embed_res[j][i] for j in range(len(embed_res)))
            best_similarities.append(max_similarity)

        return best_similarities
    
    best_direct = keep_max(res_direct)
    best_cross = keep_max(res_cross)
    best_pivot = keep_max(res_pivot)
    
    return best_direct, best_pivot, best_cross

def compute_sbert(src_list, tar_lists, args):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    # Define model
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    
    embed_res = []
    
    for tar_list in tar_lists:
        src_embeddings = model.encode(src_list)
        tar_embeddings = model.encode(tar_list)

        assert len(src_embeddings) == len(tar_embeddings)

        # Compute cosine similarity
        cosine_similarities = cosine_similarity(src_embeddings, tar_embeddings)

        # Take the diagonal
        diagonal_similarities = cosine_similarities.diagonal()

        # Convert all float32 to float
        diagonal_similarities = [float(sim) for sim in diagonal_similarities]
        
        embed_res.append(diagonal_similarities)
        
    best_similarities = []
    
    for i in range(len(embed_res[0])):  # Iterate over each element in the diagonal
        # Get the similarity of all tar_lists at index i and take the max
        max_similarity = max(embed_res[j][i] for j in range(len(embed_res)))
        best_similarities.append(max_similarity)
    
    return best_similarities

def compute_bert_score(src_list, tar_lists, args):
    from bert_score import BERTScorer
    # BERTScore calculation
    scorer = BERTScorer(model_type='FacebookAI/xlm-roberta-large', num_layers=17, batch_size=args["batch_size"])
    
    res = []
    
    for tar_list in tar_lists:
        _, _, F1 = scorer.score(src_list, tar_list, batch_size=args["batch_size"])

        # Convert F1 from tensors to float
        F1 = [float(val) for val in F1]
        
        res.append(F1)
        
    best_F1 = []
    
    for i in range(len(res[0])):  # Iterate over each element in the diagonal
        # Get the similarity of all tar_lists at index i and take the max
        max_F1 = max(res[j][i] for j in range(len(res)))
        best_F1.append(max_F1)
        
    return best_F1

def compute_comet(src_list, tar_lists, args):
    from comet import download_model, load_from_checkpoint
    
    res = []
    
    for tar_list in tar_lists:
        # Create a structure of list of dictionary
        data = [{"src": src, "mt": mt} for src, mt in zip(src_list, tar_list)]

        # Load model
        model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
        model = load_from_checkpoint(model_path)

        # Grade translations
        model_output = model.predict(data, batch_size=args["batch_size"], gpus=1)
        
        res.append(model_output.scores)
        
    best_score = []
    
    for i in range(len(res[0])):  # Iterate over each element in the diagonal
        # Get the similarity of all tar_lists at index i and take the max
        max_score = max(res[j][i] for j in range(len(res)))
        best_score.append(max_score)
    
    return best_score

def sacrebleu_score(src_list, tar_list, target_lang="en"):
    from sacrebleu.metrics import BLEU

    copy_tar_list = tar_list.copy()

    # Mark None if the reference is empty
    for i in range(len(copy_tar_list)):
        for j in range(len(copy_tar_list[0])):
            if copy_tar_list[i][j].strip() == "":
                copy_tar_list[i][j] = None

    # Set tokenizer (default is '13a', for Chinese, we use 'zh')
    if target_lang == "zh":
        bleu = BLEU(effective_order=True, tokenize="zh")
    elif target_lang == "ar":
        bleu = BLEU(effective_order=True, tokenize="flores200")
    else:
        bleu = BLEU(effective_order=True, tokenize="13a")
    
    bleus = [bleu.sentence_score(src_list[i], [r[i] for r in copy_tar_list]).score for i in range(len(src_list))]

    return bleus

def compute_chrf(src_list, tar_list, num_refs=4, nworder=2, ncorder=6, beta=2):
    from webnlg_toolkit.eval.eval import chrF_score

    refs = []
    for i in range(len(tar_list[0])):
        refs_i = []
        for j in range(num_refs):
            refs_i.append(tar_list[j][i])
        refs.append(refs_i)

    chrf_scores, _, _, _, sentence_level_scores = chrF_score(refs, src_list, num_refs, nworder, ncorder, beta, sentence_level_scores=True)

    return sentence_level_scores

def compute_ter(src_list, tar_list, lang="en", num_refs=4):
    from webnlg_toolkit.eval.eval import ter_score
    # For Russian
    from razdel import tokenize
    # For Chinese
    import jieba

    refs = []
    for i in range(len(tar_list[0])):
        refs_i = []
        for j in range(num_refs):
            refs_i.append(tar_list[j][i])
        refs.append(refs_i)

    # Tokenize the references and hypotheses
    refs_tok = copy.copy(refs)
    hyp_tok = copy.copy(src_list)
    lang_map = {
        'en': 'english',
        'es': 'spanish',
        'fr': 'french',
    }   
    # Tokenize the references 
    for i, refs in enumerate(refs_tok):
        if lang == "ar":
            refs_tok[i] = [' '.join(nltk.tokenize.wordpunct_tokenize(ref)) for ref in refs]
        elif lang == "ru":
            refs_tok[i] = [' '.join([_.text for _ in tokenize(ref)]) for ref in refs]
        elif lang == "zh":
            refs_tok[i] = [' '.join(jieba.lcut(ref)) for ref in refs]
        else:
            refs_tok[i] = [' '.join(nltk.word_tokenize(ref, language=lang_map[lang])) for ref in refs]

    # Tokenize the hypotheses
    if lang == "ar":
        hyp_tok = [' '.join(nltk.tokenize.wordpunct_tokenize(hyp)) for hyp in hyp_tok]
    elif lang == "ru":
        hyp_tok = [' '.join([_.text for _ in tokenize(hyp)]) for hyp in hyp_tok]
    elif lang == "zh":
        hyp_tok = [' '.join(jieba.lcut(hyp)) for hyp in hyp_tok]
    else:
        hyp_tok = [' '.join(nltk.word_tokenize(hyp, language=lang_map[lang])) for hyp in hyp_tok]
    
    ter_scores = ter_score(refs_tok, hyp_tok, num_refs)

    return ter_scores

def compute_bleurt(src_list, tar_list, num_refs=4):
    from webnlg_toolkit.eval.eval import bleurt
    
    refs = []
    for i in range(len(tar_list[0])):
        refs_i = []
        for j in range(num_refs):
            refs_i.append(tar_list[j][i])
        refs.append(refs_i)

    return bleurt(refs, src_list, num_refs)

def construct_info_list(src_list, tar_list, scores):
    '''
    This function is used to create a list which includes all source,
    target sentences and the corrsponding scores.
    '''
    assert len(src_list) == len(tar_list) or len(src_list) == len(tar_list[0])
    assert len(src_list) == len(scores["sacrebleu"])
    
    # Here we only consider the first reference
    info_list = []
    for i in range(len(src_list)):
        new_dict = {
            "source": src_list[i],
            "translation": tar_list[0][i],
        }
        for key, value_list in scores.items():
            assert len(src_list) == len(value_list)
            new_dict[key] = value_list[i]
            if key == "sacrebleu":
                new_dict[key] = round(new_dict[key], 2)
            else:
                new_dict[key] = round(new_dict[key], 4) * 100
        info_list.append(new_dict)
    return info_list


def main(args):
    # Read input txt file and store into a list
    with open(args["input_file"], 'r', encoding='utf-8') as f:
        src_list = [line.strip() for line in f.readlines()]
    
    # Read tar_list
    # For example, tar_list is "webnlg_toolkit/data/all_mt_baselines/ar/reference_ar"
    # We look for the "webnlg_toolkit/data/all_mt_baselines/ar/reference_ar0" to "webnlg_toolkit/data/all_mt_baselines/ar/reference_ar3"
    # Read these 4 txt files and store them into a list, where each element is a list containing one txt file
    tar_list = []
    base_path = args["tar_file_base"]  # e.g., "webnlg_toolkit/data/all_mt_baselines/ar/reference_ar"
    
    # We assume that we are reading 4 files: reference_ar0, reference_ar1, reference_ar2, reference_ar3
    for i in range(4):
        tar_file_path = f"{base_path}{i}.txt"  # This will generate file paths like "reference_ar0"
        if os.path.exists(tar_file_path):
            with open(tar_file_path, 'r', encoding='utf-8') as f:
                # Read each file and store its content as a list of strings
                tar_content = [line.strip() for line in f.readlines()]
                tar_list.append(tar_content)
        else:
            print(f"File {tar_file_path} not found.")
            return
    
    # Compute translation scores
    print(f"The input file is: \n{args['input_file']}\n\n")
    scores = {}
    if args["eval_method"] == "nmtscore":
        scores_direct, scores_pivot, scores_cross = compute_nmtscore(src_list, tar_list, args)
        scores["nmt_direct"] = scores_direct
        scores["nmt_pivot"] = scores_pivot
        scores["nmt_cross"] = scores_cross
        print(f"The average score of nmt_direct is {round(sum(scores['nmt_direct']) / len(scores['nmt_direct']), 4)} ")
        print(f"The average score of nmt_cross is {round(sum(scores['nmt_cross']) / len(scores['nmt_cross']), 4)} ")
        print(f"The average score of nmt_pivot is {round(sum(scores['nmt_pivot']) / len(scores['nmt_pivot']), 4)} ")
    elif args["eval_method"] == "sbert":
        scores_sbert = compute_sbert(src_list, tar_list, args)
        scores["sbert"] = scores_sbert
    elif args["eval_method"] == "bertscore":
        scores_bertscore = compute_bert_score(src_list, tar_list, args)
        scores["bertscore"] = scores_bertscore
    elif args["eval_method"] == "comet":
        scores_comet = compute_comet(src_list, tar_list, args)
        scores["comet"] = scores_comet
    elif args["eval_method"] == "sacrebleu":
        scores_sacrebleu = sacrebleu_score(src_list, tar_list, args["target_lang"])
        scores["sacrebleu"] = scores_sacrebleu
    elif args["eval_method"] == "chrf":
        scores_chrf = compute_chrf(src_list, tar_list, num_refs=1)
        scores["chrf"] = scores_chrf
    elif args["eval_method"] == "ter":
        scores_ter = compute_ter(src_list, tar_list, args["target_lang"], num_refs=1)
        scores["ter"] = scores_ter
    elif args["eval_method"] == "bleurt":
        scores_bleurt = compute_bleurt(src_list, tar_list, num_refs=1)
        scores["bleurt"] = scores_bleurt
    else:
        print(f"Error eval method")
    
    # Create new structure to store all info
    info_list = construct_info_list(src_list, tar_list, scores)


    # Analyze the scores by the different evaluation methods
    if args["eval_method"] == "nmtscore":
        for score_type in ["nmt_direct", "nmt_pivot", "nmt_cross"]:
            current_scores = scores[score_type]

            # Get the index of the maximum score
            max_index = current_scores.index(max(current_scores))
            print(f"The index of the maximum score is {max_index}")

            avg_score = sum(current_scores) / len(current_scores)
            print(f"\nOverall average {score_type}: {round(avg_score, 4) * 100}")

            # Save scores to txt file into lines
            # with open(args['output_file'].split(".txt")[0] + f"_{score_type}.txt", 'w', encoding='utf-8') as f:
            #     for score in current_scores:
            #         f.write(f"{score}\n")
    else:
        current_scores = scores[args["eval_method"]]
        
        # =============================== Human Evaluation ===============================
        # # Get the index of the maximum score
        # max_index = current_scores.index(max(current_scores))
        # max_score = max(current_scores)
        # print(f"The index of the maximum score is {max_index + 1}")
        # print(f"The maximum score is {max_score}")
        # print(f"The source sentence is\n{src_list[max_index]}")
        # print(f"The target sentence is\n{tar_list[0][max_index]}")
        # # Get the index of the minimum score
        # min_index = current_scores.index(min(current_scores))
        # min_score = min(current_scores)
        # print(f"The index of the minimum score is {min_index + 1}")
        # print(f"The minimum score is {min_score}")
        # print(f"The source sentence is\n{src_list[min_index]}")
        # print(f"The target sentence is\n{tar_list[0][min_index]}")

        # Print the overall average score
        avg_score = sum(current_scores) / len(current_scores)
        if args["eval_method"] == "sacrebleu":
            print(f"\nOverall average {args['eval_method']}: {round(avg_score, 2)}")
        else:
            print(f"\nOverall average {args['eval_method']}: {round(avg_score, 4) * 100}")

        # Save scores to txt file into lines
        # with open(args['output_file'], 'w', encoding='utf-8') as f:
        #     for score in current_scores:
        #         f.write(f"{score}\n")
    
     
    if args["eval_method"] != "nmtscore":
        print(f"The average score of {args['eval_method']} is {round(sum(scores[args['eval_method']]) / len(scores[args['eval_method']]), 4)}")

    print("============== NEXT FILE ==============")

    # Save scores to json file
    # with open(args['output_file'], 'w', encoding='utf-8') as f:
    #     json.dump(info_list, f, indent=4, ensure_ascii=False)
        
if __name__ == "__main__":
    # args = load_config(config_path = './eval_code/config_trans_eval.yaml')

    # main(args)

    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python eval_s2s.py <config_file_path>")
        sys.exit(1)

    # Get the YAML config file path from the command line
    config_path = sys.argv[1]

    # Load the config
    args = load_config(config_path)

    # Run the main function
    main(args)