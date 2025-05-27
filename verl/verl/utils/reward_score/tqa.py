'''
reward for TQA task: WTQ, HiTab
'''

import re
import json

PATTERN = re.compile(r'^<think>.*?</think>\s*<answer>.*?</answer>$', re.DOTALL)
ANSWER_BLOCK_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
STRICT_ANSWER_PATTERN = re.compile(r'```json\s*(\{\s*\"answer\"\s*:\s*(?:\[[\s\S]*?\])\s*\})\s*```')
ANSWER_PATTERN_1 = re.compile(r'```json\s*(\{\s*\"answer\"\s*:\s*(?:\[[\s\S]*?\]|\"[\s\S]*?\"|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\})\s*```')
ANSWER_PATTERN_2 = re.compile(r'(\{\s*\"answer\"\s*:\s*(?:\[[\s\S]*?\]|\"[\s\S]*?\"|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\})')
NUMBER_PATTERN = re.compile(r'^[+-]?(?:(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')

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

def normalize_answer(answer):
    normalized_answer = []
    for x in answer:
        if isinstance(x, int) or isinstance(x, float):
            normalized_answer.append(float(x))
        elif isinstance(x, str):
            if NUMBER_PATTERN.match(x):
                try:
                    normalized_answer.append(float(x.replace(',', '')))
                except ValueError:
                    normalized_answer.append(x)
            else:
                normalized_answer.append(x)
        else:
            return []
    return normalized_answer

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
            for i in final_answer:
                if not isinstance(i, str):
                    return False
            return True

    return False

def compute_score(predict_str, ground_truth):
    answer = extract_answer(predict_str)
    if answer is None:
        return {
            "score": 0.0,
            "format_score": 0.0,
            "accurate_score": 0.0,
            "bleu_score": 0.0,
            "rouge_score": 0.0,
        }

    normalized_answer = normalize_answer(answer)
    if len(normalized_answer) == 0 or len(normalized_answer) > 100: 
        return {
            "score": 0.0,
            "format_score": 0.0,
            "accurate_score": 0.0,
            "bleu_score": 0.0,
            "rouge_score": 0.0,
        }
    normalized_ground_truth = normalize_answer(ground_truth)

    format_score = 0.0
    if format_check(predict_str):
        format_score = 1.0

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
        "score": format_score + accurate_score,
        "format_score": format_score,
        "accurate_score": accurate_score,
        "bleu_score": 0.0,
        "rouge_score": 0.0,
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
#             for i in final_answer:
#                 if not isinstance(i, str):
#                     return format_score
#             return format_score

#     return format_score

# def compute_score(predict_str, ground_truth):
#     answer = extract_answer(predict_str)
#     if answer is None:
#         return {
#             "score": 0.0,
#             "format_score": 0.0,
#             "accurate_score": 0.0,
#             "bleu_score": 0.0,
#             "rouge_score": 0.0,
#         }

#     normalized_answer = normalize_answer(answer)
#     if len(normalized_answer) == 0 or len(normalized_answer) > 100: 
#         return {
#             "score": 0.0,
#             "format_score": 0.0,
#             "accurate_score": 0.0,
#             "bleu_score": 0.0,
#             "rouge_score": 0.0,
#         }
#     normalized_ground_truth = normalize_answer(ground_truth)

#     format_score = format_check(predict_str)

#     accurate_score = 0.0

#     if len(normalized_answer) == len(normalized_ground_truth):
#         used = [0] * len(normalized_answer)
#         for i in range(len(normalized_answer)):
#             for j in range(len(normalized_ground_truth)):
#                 if used[j] == 0:
#                     if isinstance(normalized_answer[i], float) and isinstance(normalized_ground_truth[j], float):
#                         if abs(normalized_answer[i] - normalized_ground_truth[j]) < 1e-2:
#                             used[j] = 1
#                             break
#                     else:
#                         if normalized_answer[i] == normalized_ground_truth[j]:
#                             used[j] = 1
#                             break
#         if sum(used) == len(normalized_answer):
#             accurate_score = 1.0

#     return {
#         "score": format_score + accurate_score,
#         "format_score": format_score,
#         "accurate_score": accurate_score,
#         "bleu_score": 0.0,
#         "rouge_score": 0.0,
#     }
