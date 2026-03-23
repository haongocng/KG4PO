import numpy as np
import pandas as pd
import re

def normalize_item_name(item_name: str) -> str:
    if not isinstance(item_name, str):
        item_name = str(item_name)
    
    cleaned = item_name.strip().lower()
    cleaned = cleaned.replace('"', '').replace("'", "")
    cleaned = re.sub(r'^\d+[\.\)\-]?\s*', '', cleaned)
    cleaned = re.sub(r'^movie:\s*', '', cleaned)
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace("&amp;", "&").replace("&amp", "&").replace("&reg;", "®")
    
    return cleaned

def get_rank(predictions: list, ground_truth: str) -> int:
    if not predictions:
        return 999
        
    gt_normalized = normalize_item_name(ground_truth)
    
    for idx, pred in enumerate(predictions):
        pred_normalized = normalize_item_name(pred)
        if gt_normalized == pred_normalized:
            return idx + 1
            
    print(f"[DEBUG MISS] GT: '{gt_normalized}' | LLM response: '{predictions}'")
    
    return 999


class Metric():
    def __init__(self, rank_list, conf) -> None:
        self.rank_list = rank_list
        self.conf = conf
    
    def ndcg(self, N):
        res = []
        for rank in self.rank_list:
            if rank > N:
                res.append(0)
            else:
                res.append((1 / np.log2(rank + 1)))
        
        return np.mean(res)
    
    def hit(self, N):
        res = []
        for rank in self.rank_list:
            if rank > N:
                res.append(0)
            else:
                res.append(1)
        return np.mean(res)
    
    def map(self, N):
        res = []
        for rank in self.rank_list:
            if rank > N:
                res.append(0)
            else:
                res.append((1 / rank))
        return np.mean(res)
    
    def run(self):
        res = pd.DataFrame({'KPI@K': ['NDCG', 'HIT', 'MAP']})
        
        if self.conf['candidate_size'] == 10:
            topk_list = [1, 5, 10]
        elif self.conf['candidate_size'] >= 20:
            topk_list = [1, 5, 10, 20]
        else:
            topk_list = [1, 5] 
            
        for topk in topk_list:
            metric_res = []
            metric_res.append(self.ndcg(topk))
            metric_res.append(self.hit(topk))
            metric_res.append(self.map(topk))

            metric_res = np.array(metric_res)
            res[topk] = metric_res
            
        count = 0
        for element in self.rank_list:
            if element <= self.conf['candidate_size']:
                count += 1
                
        res['#valid_data'] = np.array([count, 0, 0])
        
        return res


def evaluate_batch(predictions_list: list, ground_truths_list: list, candidate_size: int = 20) -> dict:
    ranks = []
    for preds, gt in zip(predictions_list, ground_truths_list):
        ranks.append(get_rank(preds, gt))
        
    metric_engine = Metric(rank_list=ranks, conf={'candidate_size': candidate_size})
    df_result = metric_engine.run()
    
    metrics_dict = {}
    for topk in df_result.columns:
        if isinstance(topk, int): 
            metrics_dict[f'NDCG@{topk}'] = df_result[topk].iloc[0]
            metrics_dict[f'HIT@{topk}'] = df_result[topk].iloc[1]
            metrics_dict[f'MAP@{topk}'] = df_result[topk].iloc[2]
            
    return metrics_dict