Answer = open("Answer.txt", "r").readlines()
Output = open("Output.txt", "r").readlines()

correct = 0
total = 0
answer = 0
# print(Answer) # testing
# print(Output) # testing
for answerLine, answerOutput in zip(Answer, Output):
    answerLine = answerLine.replace("\n", "")
    answerOutput = answerOutput.replace("\n", "")

    if not answerOutput or not answerLine:
        continue
    if answerLine == answerOutput:
        # print(answerLine, answerOutput) # for testing
        correct += 1
    total += 1

answer = correct / total
print(answer)


