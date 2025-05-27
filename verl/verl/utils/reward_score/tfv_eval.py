'''
reward for TFV task: TabFact, InfoTabs, Feverous, PubHealthTab
'''

import re
import json

ANSWER_BLOCK_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
ANSWER_PATTERN_1 = re.compile(r'```json\s*(\{\s*\"answer\"\s*:\s*\"(?:entailed|refuted)\"\s*\})\s*```')
ANSWER_PATTERN_2 = re.compile(r'(\{\s*\"answer\"\s*:\s*\"(?:entailed|refuted)\"\s*\})')
ANSWER_PATTERN_3 = re.compile(r'(\"answer\"\s*:\s*\"(?:entailed|refuted)\")')

def parse_json(answer):
    try:
        data = json.loads(answer)
        if not isinstance(data, dict) or "answer" not in data:
            return None
        data["answer"] = data["answer"].lower()
        if data["answer"] not in ["entailed", "refuted"]:
            return None
        return data["answer"]
    except json.JSONDecodeError:
        return None

def extract_answer_pattern(predict_str):
    answer_match = ANSWER_PATTERN_1.search(predict_str)
    if answer_match is not None:
        answer = answer_match.group(1).strip()
        return parse_json(answer)
    
    answer_match = ANSWER_PATTERN_2.search(predict_str)
    if answer_match is not None:
        answer = answer_match.group(1).strip()
        return parse_json(answer)

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
    
    return None

def compute_score(predict_str, ground_truth):
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0]
    answer = extract_answer(predict_str)
    if answer is None:
        return {
            "accurate_score": 0.0,
            "bleu_score": 0.0,
            "rouge_score": 0.0,
        }

    accurate_score = 1.0 if answer == ground_truth else 0.0

    return {
        "accurate_score": accurate_score,
        "bleu_score": 0.0,
        "rouge_score": 0.0,
    }
