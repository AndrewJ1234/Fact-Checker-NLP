import sys

def grade_files(answerFile = "Answer.txt", outputFile = "Output.txt"):
    correct = 0
    total = 0
    results = {
        "correct": 0,
        "total": 0,
        "score": 0,
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
                correct += 1
            else:
                print(f"Line {total + 1}: MISMATCH - Expected: '{answerLine}', Actual: '{answerOutput}'")
            total += 1

            score = correct / total if total > 0 else 0

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