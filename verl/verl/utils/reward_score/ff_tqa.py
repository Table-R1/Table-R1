'''
reward for FF-TQA task: FeTaQA
'''

import re
import json
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

bleu = BLEU(effective_order=True)
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

PATTERN = re.compile(r'^<think>.*?</think>\s*<answer>.*?</answer>$', re.DOTALL)
ANSWER_BLOCK_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
STRICT_ANSWER_PATTERN = re.compile(r'```json\s*(\{\s*\"answer\"\s*:\s*.*?\})\s*```', re.DOTALL)
ANSWER_PATTERN_1 = re.compile(r'```json\s*(\{\s*\"answer\"\s*:\s*.*?\})\s*```', re.DOTALL)
ANSWER_PATTERN_2 = re.compile(r'(\{\s*\"answer\"\s*:\s*.*?\})', re.DOTALL)

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

# for instruct model
def format_check(predict_str):
    if PATTERN.fullmatch(predict_str):
        for tag in ["<think>", "</think>", "<answer>", "</answer>"]:
            if predict_str.count(tag) != 1:
                return False
        answer_block_match = ANSWER_BLOCK_PATTERN.search(predict_str).group(1)
        answer_match = STRICT_ANSWER_PATTERN.search(answer_block_match)
        if answer_match is not None:
            answer = answer_match.group(1).strip()
            final_answer = parse_json(answer)
            if final_answer is None:
                return False
            return True

    return False

def compute_score(predict_str, ground_truth):
    ground_truth = ground_truth[0]
    answer = extract_answer(predict_str)
    if answer is None:
        return {
            "score": 0.0,
            "format_score": 0.0,
            "accurate_score": 0.0,
            "bleu_score": 0.0,
            "rouge_score": 0.0,
        }

    format_score = 0.0
    if format_check(predict_str):
        format_score = 1.0

    answer = normalize_answer(answer)
    ground_truth = normalize_answer(ground_truth)

    bleu_score = bleu.sentence_score(answer, [ground_truth]).score / 100
    rougel_score = rouge.score(answer, ground_truth)['rougeL'].fmeasure

    return {
        "score": format_score + (bleu_score + rougel_score) / 2,
        "format_score": format_score,
        "accurate_score": (bleu_score + rougel_score) / 2,
        "bleu_score": bleu_score,
        "rouge_score": rougel_score,
    }

# for base model: cumulative format reward
# def format_check(predict_str):
#     format_score = 0.0
#     if PATTERN.fullmatch(predict_str):
#         format_score += 0.3
#         for tag in ["<think>", "</think>", "<answer>", "</answer>"]:
#             if predict_str.count(tag) != 1:
#                 return format_score
#         format_score += 0.2
#         answer_block_match = ANSWER_BLOCK_PATTERN.search(predict_str).group(1)
#         answer_match = STRICT_ANSWER_PATTERN.search(answer_block_match)
#         if answer_match is not None:
#             format_score += 0.3
#             answer = answer_match.group(1).strip()
#             final_answer = parse_json(answer)
#             if final_answer is None:
#                 return format_score
#             format_score += 0.2

#     return format_score

# def compute_score(predict_str, ground_truth):
#     ground_truth = ground_truth[0]
#     answer = extract_answer(predict_str)
#     if answer is None:
#         return {
#             "score": 0.0,
#             "format_score": 0.0,
#             "accurate_score": 0.0,
#             "bleu_score": 0.0,
#             "rouge_score": 0.0,
#         }

#     format_score = format_check(predict_str)

#     answer = normalize_answer(answer)
#     ground_truth = normalize_answer(ground_truth)

#     bleu_score = bleu.sentence_score(answer, [ground_truth]).score / 100
#     rougel_score = rouge.score(answer, ground_truth)['rougeL'].fmeasure

#     return {
#         "score": format_score + (bleu_score + rougel_score) / 2,
#         "format_score": format_score,
#         "accurate_score": (bleu_score + rougel_score) / 2,
#         "bleu_score": bleu_score,
#         "rouge_score": rougel_score,
#     }
