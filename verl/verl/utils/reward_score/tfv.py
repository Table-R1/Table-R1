'''
reward for TFV task: TabFact
'''

import re
import json

PATTERN = re.compile(r'^<think>.*?</think>\s*<answer>.*?</answer>$', re.DOTALL)
ANSWER_BLOCK_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
STRICT_ANSWER_PATTERN = re.compile(r'```json\s*(\{\s*\"answer\"\s*:\s*\"(?:entailed|refuted)\"\s*\})\s*```')
ANSWER_PATTERN_1 = re.compile(r'```json\s*(\{\s*\"answer\"\s*:\s*\"(?:entailed|refuted)\"\s*\})\s*```')
ANSWER_PATTERN_2 = re.compile(r'(\{\s*\"answer\"\s*:\s*\"(?:entailed|refuted)\"\s*\})')

def parse_json(answer):
    try:
        data = json.loads(answer)
        if not isinstance(data, dict) or "answer" not in data:
            return None
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

    accurate_score = 1.0 if answer == ground_truth else 0.0

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

#     accurate_score = 1.0 if answer == ground_truth else 0.0

#     return {
#         "score": format_score + accurate_score,
#         "format_score": format_score,
#         "accurate_score": accurate_score,
#         "bleu_score": 0.0,
#         "rouge_score": 0.0,
#     }
