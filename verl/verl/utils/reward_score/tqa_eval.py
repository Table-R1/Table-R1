'''
reward for TQA task: WTQ, HiTab, TabMCQ, TabMWP, FinQA
'''

import re
import json

ANSWER_BLOCK_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
ANSWER_PATTERN_1 = re.compile(r'```json\s*(\{\s*\"answer\"\s*:\s*(?:\[[\s\S]*?\]|\"[\s\S]*?\"|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\})\s*```')
ANSWER_PATTERN_2 = re.compile(r'(\{\s*\"answer\"\s*:\s*(?:\[[\s\S]*?\]|\"[\s\S]*?\"|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\})')
ANSWER_PATTERN_3 = re.compile(r'(\"answer\"\s*:\s*(?:\[[\s\S]*?\]|\"[\s\S]*?\"|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?))')

def parse_json(answer):
    try:
        data = json.loads(answer)
        if not isinstance(data, dict) or "answer" not in data:
            return None
        if isinstance(data["answer"], list):
            return data["answer"]
        else:
            return [data["answer"]]
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

def str_to_num(text):
    text = text.replace(",", "")
    try:
        num = float(text)
    except ValueError:
        if "%" in text:
            text = text.replace("%", "")
            try:
                num = float(text)
                num = num / 100.0
            except ValueError:
                num = "n/a"
        elif "const" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            try:
                num = float(text)
            except ValueError:
                num = "n/a"
        else:
            num = "n/a"
    return num

def normalize_answer(answer):
    normalized_answer = []
    for x in answer:
        if isinstance(x, int) or isinstance(x, float):
            normalized_answer.append(float(x))
        elif isinstance(x, str):
            num = str_to_num(x)
            if num != "n/a":
                normalized_answer.append(float(num))
            else:
                normalized_answer.append(x.lower())
        else:
            return []
    return normalized_answer

def compute_score(predict_str, ground_truth):
    answer = extract_answer(predict_str)
    if answer is None:
        return {
            "accurate_score": 0.0,
            "bleu_score": 0.0,
            "rouge_score": 0.0,
        }

    normalized_answer = normalize_answer(answer)
    if len(normalized_answer) == 0 or len(normalized_answer) > 100: 
        return {
            "accurate_score": 0.0,
            "bleu_score": 0.0,
            "rouge_score": 0.0,
        }
    normalized_ground_truth = normalize_answer(ground_truth)

    accurate_score = 0.0

    if len(normalized_answer) == len(normalized_ground_truth):
        used = [0] * len(normalized_answer)
        for i in range(len(normalized_answer)):
            for j in range(len(normalized_ground_truth)):
                if used[j] == 0:
                    if isinstance(normalized_answer[i], float) and isinstance(normalized_ground_truth[j], float):
                        if abs(normalized_answer[i] - normalized_ground_truth[j]) < 1e-2:
                            used[j] = 1
                            break
                    else:
                        if normalized_answer[i] == normalized_ground_truth[j]:
                            used[j] = 1
                            break
        if sum(used) == len(normalized_answer):
            accurate_score = 1.0

    return {
        "accurate_score": accurate_score,
        "bleu_score": 0.0,
        "rouge_score": 0.0,
    }
