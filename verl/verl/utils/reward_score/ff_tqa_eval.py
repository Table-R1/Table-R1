'''
reward for FF-TQA task: FeTaQA, Rotowire, ToTTo, QTSumm
'''

import re
import json
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

bleu = BLEU(effective_order=True)
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

ANSWER_BLOCK_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
ANSWER_PATTERN_1 = re.compile(r'```json\s*(\{\s*\"answer\"\s*:\s*.*?\})\s*```', re.DOTALL)
ANSWER_PATTERN_2 = re.compile(r'(\{\s*\"answer\"\s*:\s*.*?\})', re.DOTALL)
ANSWER_PATTERN_3 = re.compile(r'(\"answer\"\s*:\s*.*?)', re.DOTALL)

def parse_json(answer):
    try:
        data = json.loads(answer)
        if not isinstance(data, dict) or "answer" not in data:
            return None
        return data["answer"]
    except json.JSONDecodeError:
        return None

def extract_answer_pattern(predict_str):
    answer_match = ANSWER_PATTERN_1.search(predict_str)
    if answer_match is not None:
        return parse_json(answer_match.group(1).strip())

    answer_match = ANSWER_PATTERN_2.search(predict_str)
    if answer_match is not None:
        return parse_json(answer_match.group(1).strip())

    answer_match = ANSWER_PATTERN_3.search(predict_str)
    if answer_match is not None:
        answer = answer_match.group(1).strip()
        answer = "{" + answer + "}"
        return parse_json(answer)

    return None

def extract_answer(predict_str):
    answer_block_match = ANSWER_BLOCK_PATTERN.search(predict_str)
    if answer_block_match is not None:
        answer = extract_answer_pattern(answer_block_match.group(1))
        if answer is not None:
            return answer

    answer = extract_answer_pattern(predict_str)
    if answer is not None:
        return answer

    return predict_str.strip()

def normalize_answer(ans):
    if isinstance(ans, str):
        return ans.strip()
    elif isinstance(ans, list):
        return " ".join([str(x).strip() for x in ans])
    else:
        return str(ans).strip()

def compute_score(predict_str, ground_truth):
    answer = extract_answer(predict_str)
    if answer is None:
        return {
            "accurate_score": 0.0,
            "bleu_score": 0.0,
            "rouge_score": 0.0,
        }

    answer = normalize_answer(answer)
    if isinstance(ground_truth, str):
        ground_truth = [normalize_answer(ground_truth)]
    else:
        ground_truth = [normalize_answer(gt) for gt in ground_truth]

    bleu_score = bleu.sentence_score(answer, ground_truth).score / 100

    max_rouge_score = 0.0
    for gt in ground_truth:
        rouge_score = rouge.score(answer, gt)['rougeL'].fmeasure
        max_rouge_score = max(max_rouge_score, rouge_score)

    return {
        "accurate_score": (bleu_score + max_rouge_score) / 2,
        "bleu_score": bleu_score,
        "rouge_score": max_rouge_score,
    }
