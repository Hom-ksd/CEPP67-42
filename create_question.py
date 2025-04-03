import csv

# Read questions and answers
with open("questions.txt", "r", encoding="utf-8") as q_file:
    questions = [line.strip() for line in q_file.readlines()]

with open("answers.txt", "r", encoding="utf-8") as a_file:
    answers = [line.strip() for line in a_file.readlines()]

# Ensure both files have the same number of lines
if len(questions) != len(answers):
    raise ValueError("Mismatch between number of questions and answers")

# Save to CSV file
with open("Question.csv", "w", encoding="utf-8", newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Question", "Answer"])
    writer.writerows(zip(questions, answers))

print("Data successfully saved to Question.csv")
