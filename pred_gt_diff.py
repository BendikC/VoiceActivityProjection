## this script takes the turn-level csv file extracted with json_to_csv.py
## and compares the predicted turns to manual ground truth turns

import pandas as pd

def read_prediction_csv(pred_csv_path: str) -> dict:
    # reads predicted turns from the csv and returns as a dict
    df = pd.read_csv(pred_csv_path)
    pred_turns = {}
    for _, row in df.iterrows():
        start = row['start_time']
        end = row['end_time']
        speaker = row['predicted_speaker']
        pred_turns[(start, end)] = speaker
    print(f"Read {len(pred_turns)} predicted turns from {pred_csv_path}")
    return pred_turns

def read_ground_truth_csv(gt_csv_path: str, speaker_1: str, speaker_2: str, task: str, level: str) -> dict:
    # reads ground truth turns from the csv and returns as a dict

    # first have to filter the csv for the correct speaker, task, level, pair
    df = pd.read_csv(gt_csv_path)
    df = df[(df['speaker'] == speaker_1) | (df['speaker'] == speaker_2) & (df['task'] == task) & (df['level'] == level)]
    gt_turns = {}
    for _, row in df.iterrows():
        start = row['start_time']
        end = row['end_time']
        speaker = row['speaker']
        gt_turns[(start, end)] = 'A' if speaker == speaker_1 else 'B'
    print(f"Read {len(gt_turns)} ground truth turns from {gt_csv_path}")
    return gt_turns

def compare_turns(pred_turns: dict, gt_turns: dict, threshold: float = 1) -> dict:
    # compares the predicted turns with the ground truth turns, computing:
    # - number of correctly predicted turn shifts 
    # (i.e does a gt turn shift to the corresponding speaker happen within a threshold time window of the predicted turn shift)
    # - number of false positives (predicted turn shifts that do not correspond to a gt turn shift)
    # - number of false negatives (gt turn shifts that are not predicted)
    # - average time difference between predicted and gt turn shifts for correctly predicted shifts
    correct_shifts = 0
    false_positives = 0
    false_negatives = 0
    matched_gt_shifts = set()
    time_differences = []
    # iterate over keys and items in pred_turns, count as correct if
    # 1. there is a gt turn shift within threshold seconds of the predicted turn shift time (so start time only)
    # 2. the predicted turn shift is before the gt turn shift (needs to be predictive)
    # 3. the predicted speaker is the same as the gt speaker after the turn shift
    # 4. has not been matched already
    for (pred_start, pred_end), pred_speaker in pred_turns.items():
        pred_shift_time = pred_start
        matched = False
        for (gt_start, gt_end), gt_speaker in gt_turns.items():
            gt_shift_time = gt_start
            if abs(pred_shift_time - gt_shift_time) <= threshold:
                if pred_shift_time <= gt_shift_time:
                    if pred_speaker == gt_speaker:
                        if (gt_start, gt_end) not in matched_gt_shifts:
                            correct_shifts += 1
                            matched_gt_shifts.add((gt_start, gt_end))
                            time_differences.append(abs(pred_shift_time - gt_shift_time))
                            matched = True
                            break
        if not matched:
            false_positives += 1
    # now count false negatives
    gt_shift_times = [gt_start for (gt_start, gt_end) in gt_turns.keys()]

    false_negatives = len(gt_shift_times) - len(matched_gt_shifts)
    avg_time_difference = sum(time_differences) / len(time_differences) if time_differences else 0
    return {
        'correct_shifts': correct_shifts,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'avg_time_difference': avg_time_difference
    }

def precision_recall_f1(correct: int, false_pos: int, false_neg: int) -> dict:
    precision = correct / (correct + false_pos) if (correct + false_pos) > 0 else 0
    recall = correct / (correct + false_neg) if (correct + false_neg) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    # example usage
    pred_csv_path = "01_S01_desert_d0000_conversation_0.1_adaptive_threshold.csv"
    gt_csv_path = "ch_solo.csv"
    speaker_1 = "S01"
    speaker_2 = "S02"
    task = "desert"
    level = "d0000"
    pred_turns = read_prediction_csv(pred_csv_path)
    gt_turns = read_ground_truth_csv(gt_csv_path, speaker_1, speaker_2, task, level)
    results = compare_turns(pred_turns, gt_turns)
    metrics = precision_recall_f1(results['correct_shifts'], results['false_positives'], results['false_negatives'])
    print("Comparison Results:")
    print(f"Correctly Predicted Turn Shifts: {results['correct_shifts']}")
    print(f"False Positives: {results['false_positives']}")
    print(f"False Negatives: {results['false_negatives']}")
    print(f"Average Time Difference for Correct Shifts: {results['avg_time_difference']:.3f} seconds")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")

if __name__ == "__main__":
    main()