import os
import csv
import kenlm
import evaluate
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from evaluate import load
from sacrebleu.metrics import BLEU
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
#------------------------#

def evaluate_self_bleu(ref, gen, self_sacre):
    """computing self BLEU"""
    print("computing self BLEU ...")
    self_bleu_scores = 0
    tok_ref = [word_tokenize(x.lower().strip()) for x in ref]
    tok_gen = [word_tokenize(x.lower().strip()) for x in gen]
    assert len(tok_ref) == len(tok_gen), "Size of ref and gen doesn't match!"
    
    for r, g in tqdm(zip(tok_ref, tok_gen)):
        self_bleu_scores += (sentence_bleu([r], g, weights=(1, 0, 0, 0))*100)
    self_bleu_score = self_bleu_scores/len(tok_gen)
    
    print("computing self BLEU (sacre)...")
    self_bleu_scores_s = 0
    assert len(ref) == len(gen), "Size of ref and gen doesn't match!"
    for r, g in tqdm(zip(ref, gen)):
        self_bleu_scores_s += (self_sacre.sentence_score(g, [r]).score)
    self_bleu_score_s = self_bleu_scores_s/len(gen)
    
    return (self_bleu_score, self_bleu_score_s)

def evaluate_ref_bleu(ref, gen, ref_sacre):
    """computing reference BLEU"""
    print("computing ref BLEU ...")
    ref_bleu_scores = 0
    for g, r in tqdm(zip(gen, ref)):
        ref_bleu_scores += (ref_sacre.corpus_score([g], [r])).score
    ref_bleu_score = ref_bleu_scores/len(ref)

    return ref_bleu_score
    
def evaluate_ppl(gen, kenlm):
    """computing perplexity"""
    print("computing ppl ...")
    tok = [word_tokenize(i.lower().strip()) for i in gen]

    ppl_scores = 0
    length = 0
    for i, words in tqdm(enumerate(tok)):
        length += len(words)
        log_prob = kenlm.score(' '.join(words))
        ppl_scores += log_prob
    ppl_score = ppl_scores/length
    return (10**(-ppl_score/np.log(10)))

def evaluate_bertscore(bertscore, ref, gen):
    """computing bertscore"""
    print("computing bertscore ...")
    gen_score = bertscore.compute(predictions=gen, references=ref, model_type="distilbert-base-uncased")
    gen_f1 = sum(gen_score["f1"])/len(gen_score["f1"])
    
    return gen_f1

class Evaluator():
    def __init__(self, config):
        super(Evaluator, self).__init__()
        
        self.config = config
        # BLEU #
        self.self_sacre = BLEU(effective_order=True)
        self.ref_sacre = BLEU()
        # BERTSCORE #
        if "bert" in config.metric: self.bertscore = load("bertscore")
        # PERPLEXITY #
        ppl_path = f"data/{config.dataset}/ppl_{config.dataset.lower()}.binary"
        self.kenlm = kenlm.Model(ppl_path)
    
    def compute_eval_process(self, gen, ref1=None, ref2=None):
        evaluation_score = []
        # compute self-BLEU #
        if "self" in self.config.metric: 
            self_out = evaluate_self_bleu(ref1, gen, self.self_sacre)
            evaluation_score.append(self_out[0])
            evaluation_score.append(self_out[1])
        
        # compute ref-BLEU #
        if "ref" in self.config.metric:
            ref_out = evaluate_ref_bleu(ref2, gen, self.ref_sacre)
            evaluation_score.append(ref_out)
        
        # compute bertscore #
        if "bert" in self.config.metric:
            bert_out = evaluate_bertscore(self.bertscore, ref1, gen)
            evaluation_score.append(bert_out)
        
        # compute perplexity (ppl) with 5-gram kenlm #
        if "ppl" in self.config.metric:
            ppl_out = evaluate_ppl(gen, self.kenlm)
            evaluation_score.append(ppl_out)
        
        return evaluation_score

def main(config):
    
    print(f"[EVALUATION START({config.name})] | {config.dataset}")
    print(f"[METRIC] | {' '.join(config.metric)}")
    # get model saved path
    print(f"[SAVED MODEL PATH] | {config.save}")
    
    # make evaluation log directory
    eval_dir = f"{config.save}/eval"
    os.makedirs(eval_dir, exist_ok=True)
    
    data = pd.read_csv(f"{config.data_path}", index_col=False)
    # original code
    assert data.columns[1] == "gen", "Column name is not gen"
    # for new experiemnt
    # assert data.columns[0] == "gen", "Column name is not gen"
    data["gen"] = data["gen"].fillna("' '")
    x_gen = data.gen.astype(str).tolist()
    # assert data.columns[1] == "rev", "Column name is not rev"
    # data["rev"] = data["rev"].fillna("' '")
    # x_gen = data.rev.astype(str).tolist()
    
    if "ref" not in config.metric:
        ref_data = pd.read_csv(f"{config.ref_path}", index_col=False)
        x_ref1 = ref_data.text.astype(str).tolist()
    
    if "ref" in config.metric: 
        ref_data2 = pd.read_csv(f"data/{config.dataset}/reference_{config.ref_cls}.csv", index_col=False)
        x_ref2 = ref_data2.values.tolist()
    
    # compute evalutation
    evaluator = Evaluator(config)
    if "ref" in config.metric: 
        evaluation_score = evaluator.compute_eval_process(x_gen, None, x_ref2)
    else:
        evaluation_score = evaluator.compute_eval_process(x_gen, x_ref1)
    
    with open(f"{eval_dir}/eval_{config.name}_{'+'.join(config.metric)}.csv", mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(config.metric)
        for row in [evaluation_score]:
            writer.writerow(row)
    
    print("[EVALUATION END]")


if __name__ == "__main__":
    """
    [For Excuting]
    python evaluation.py --dataset Amazon --data_path result/amazon_saved_tedi/241030/241030_1735/test_log/rec_gen_1to5_2411111553.csv --ref_path data/Amazon/amazon_test1.csv --metric self s_self ppl
    python evaluation.py --dataset Amazon --data_path result/amazon_saved_tedi/241030/241030_1735/test_log/rec_gen_5to1_2411111719.csv --ref_path data/Amazon/amazon_test5.csv --metric self s_self ppl
    python evaluator.py --dataset Amazon --save result/amazon_saved_tedi/241030/241030_1735 --data_path result/amazon_saved_tedi/241030/241030_1735/test_log/rec_gen_1to5_ref.csv  --ref_cls 1 --metric ref

    python evaluation.py --dataset Yelp --data_path result/yelp_saved_tedi/241209/241209_2301/test_log/rec_gen_1to5_2412091544.csv --ref_path data/Yelp/yelp_test1.csv --metric self s_self ppl
    python evaluation.py --dataset Yelp --data_path result/yelp_saved_tedi/241209/241209_2301/test_log/rec_gen_5to1_2412091544.csv --ref_path data/Yelp/yelp_test5.csv --metric self s_self ppl
    python evaluation.py --dataset Yelp --data_path result/yelp_saved_tedi/241209/241209_2301/test_log/rec_gen_1to5_ref.csv --ref_cls 1 --metric ref
    python evaluation.py --dataset Yelp --data_path result/yelp_saved_tedi/241209/241209_2301/test_log/rec_gen_5to1_ref.csv --ref_cls 5 --metric ref
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None) # amazon, yelp
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--save", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--ref_path", type=str)
    parser.add_argument("--ref_cls", type=str) # 1 or 5 / ref BLEU
    parser.add_argument("--metric", type=str, nargs="+", help="self s_self ref bert ppl")
    # parser.add_argument("--self", type=bool, default=False)
    # parser.add_argument("--ref", type=bool, default=False)
    # parser.add_argument("--bert", type=bool, default=False)
    # parser.add_argument("--ppl", type=bool, default=False)
    args, left_argv = parser.parse_known_args()
    main(args)