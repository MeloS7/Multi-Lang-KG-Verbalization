import json
import yaml
from nmtscore import NMTScorer
# from bert_score import BERTScorer
# from comet import download_model, load_from_checkpoint

def load_config(config_path='./config/config_translation_eval.yaml'):
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

def compute_nmtscore(src_list, tar_list, args):
    # Define NMTScorer
    scorer = NMTScorer(args["model_name"], device="cuda:0")
    
    # Direct translation probability
    scores_direct = scorer.score_direct(
        src_list, 
        tar_list, 
        a_lang="en",
        b_lang=args["target_lang"],
        normalize=True, 
        both_directions=True,
        score_kwargs={"batch_size": args["batch_size"]}
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
        a_lang="en",
        b_lang=args["target_lang"],
        pivot_lang="en",
        normalize=True, 
        both_directions=True,
        translate_kwargs={"batch_size": args["batch_size"]},
        score_kwargs={"batch_size": args["batch_size"]}
    )
    
    print("Pivot scores finished!")
    
    return scores_direct, scores_pivot, scores_cross

def compute_sbert(src_list, tar_list, args):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    # Define model
    # model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    model = SentenceTransformer('sentence-transformers/LaBSE')
    src_embeddings = model.encode(src_list)
    tar_embeddings = model.encode(tar_list)
    
    assert len(src_embeddings) == len(tar_embeddings)
    
    # Compute cosine similarity
    cosine_similarities = cosine_similarity(src_embeddings, tar_embeddings)

    # Take the diagonal
    diagonal_similarities = cosine_similarities.diagonal()
    
    # Convert all float32 to float
    diagonal_similarities = [float(sim) for sim in diagonal_similarities]
    
    return diagonal_similarities

def compute_bert_score(src_list, tar_list, args):
    # BERTScore calculation
    scorer = BERTScorer(model_type='FacebookAI/xlm-roberta-large', num_layers=17, batch_size=args["batch_size"])
    _, _, F1 = scorer.score(src_list, tar_list, batch_size=args["batch_size"])
    
    # Convert F1 from tensors to float
    F1 = [float(val) for val in F1]
    return F1

def compute_comet(src_list, tar_list, args):
    # Create a structure of list of dictionary
    data = [{"src": src, "mt": mt} for src, mt in zip(src_list, tar_list)]
    
    # Load model
    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
    model = load_from_checkpoint(model_path)
    
    # Grade translations
    model_output = model.predict(data, batch_size=args["batch_size"], gpus=1)
    
    return model_output.scores

def construct_info_list(src_list, tar_list, scores):
    '''
    This function is used to create a list which includes all source,
    target sentences and the corrsponding scores.
    '''
    assert len(src_list) == len(tar_list)
    
    info_list = []
    for i in range(len(src_list)):
        new_dict = {
            "source": src_list[i],
            "translation": tar_list[i],
        }
        for key, value_list in scores.items():
            assert len(src_list) == len(value_list)
            new_dict[key] = value_list[i]
        info_list.append(new_dict)
    return info_list

def main(args):
    # Read (src_text, translation) pairs
    pair_data = read_translation_pair(args["input_file"])
    
    # Get source and target sentence lists
    src_list, tar_list = get_src_tar_list(pair_data)
    
    # Compute translation scores
    scores = {}
    if args["eval_method"] == "nmtscore":
        scores_direct, scores_pivot, scores_cross = compute_nmtscore(src_list, tar_list, args)
        scores["nmt_direct"] = scores_direct
        scores["nmt_pivot"] = scores_pivot
        scores["nmt_cross"] = scores_cross
    elif args["eval_method"] == "sbert":
        scores_sbert = compute_sbert(src_list, tar_list, args)
        scores["sbert"] = scores_sbert
    elif args["eval_method"] == "bertscore":
        scores_bertscore = compute_bert_score(src_list, tar_list, args)
        scores["bertscore"] = scores_bertscore
    elif args["eval_method"] == "comet":
        scores_comet = compute_comet(src_list, tar_list, args)
        scores["cometkiwi"] = scores_comet
    else:
        print(f"Error eval method")
    
    # Create new structure to store all info
    info_list = construct_info_list(src_list, tar_list, scores)

    # Save scores to json file
    with open(args['output_file'], 'w') as f:
        json.dump(info_list, f, indent=4)

    print(f"Scores saved to {args['output_file']}")
        
if __name__ == "__main__":
    args = load_config(config_path = './config/config_translation_eval.yaml')

    main(args)