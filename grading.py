import sys

def grade_files(answerFile = "Answer.txt", outputFile = "Output.txt"):
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    results = {
        "correct": 0,
        "total": 0,
        "score": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "error": None
    }

    answer = open(answerFile, "r").readlines()
    output = open(outputFile, "r").readlines()

    try:

        for answerLine, answerOutput in zip(answer, output):
            answerLine = answerLine.replace("\n", "")
            answerOutput = answerOutput.replace("\n", "")

            if not answerOutput or not answerLine:
                continue
            if answerLine == answerOutput:
                # print(answerLine, answerOutput) # for testing
                if answerLine=='TRUE':
                    true_positive += 1
                correct += 1
            else:
                print(f"Line {total + 1}: MISMATCH - Expected: '{answerLine}', Actual: '{answerOutput}'")
                if answerOutput=='TRUE':
                    false_positive += 1
                else:
                    false_negative += 1
            total += 1
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
            recall=true_positive/(true_positive + false_negative) if true_positive + false_negative > 0 else 0
            f1 = 2 *( precision * recall / (precision + recall)) if precision + recall > 0 else 0
            score = correct / total if total > 0 else 0
            results["precision"] = precision
            results["recall"] = recall
            results["f1"] = f1
            results["correct"] = correct
            results["total"] = total
            results["score"] = score
        return results

    except FileNotFoundError as e:
        error_msg = f"File not found: {str(e)}"
        results["error"] = error_msg
        print(f"Error: {error_msg}")
        return results

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        results["error"] = error_msg
        print(f"Error: {error_msg}")
        return results

if __name__ == "__main__":
    answer = sys.argv[1] if len(sys.argv) > 1 else "Answer.txt"
    output = sys.argv[2] if len(sys.argv) > 2 else "Output.txt"
    results = grade_files(answer, output)
    print(f"Final Score: {results['score']:.2%} ({results['correct']}/{results['total']} correct)")
    print(f"Final Precision: {results['precision']:.2%}")
    print(f"Final Recall: {results['recall']:.2%}")
    print(f"Final F1: {results['f1']:.2%}")