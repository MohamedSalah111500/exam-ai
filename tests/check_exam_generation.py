import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exam_generation import MAX_QUESTIONS, build_messages, copy_mismatches, generate_questions, join_pages

STEM = "Which choice completes the text so that it conforms to the conventions of Standard English?"
WORDS = ["writers", "photosynthesis", "rare", "century", "bay", "cinema", "quilts", "desert", "lagos", "rings", "congress", "length"]
QUESTIONS = [
    (f"Passage number {n} talks about the {word} and ends with a ______ blank.", [f"{word},", f"{word} that", f"{word}", f"{word}, but"])
    for n, word in enumerate(WORDS, 1)
]
COPY_NOTE = "rewite these questions exactly without changing the questions wording or the choices and include the whole question passage and the question under it exactly as given"


def page(number, body):
    return f"THE REVOLUTION IN SAT MR. TEACHER 00966500000000\n{body}\nTogether, we can {number}"


PAGES = [page(1, "Remember, to join two Independent Clauses, Use ; , FANBOYS . :")]
for index in range(0, len(QUESTIONS), 2):
    body = ""
    for passage, choices in QUESTIONS[index:index + 2]:
        body += f"{passage}\n{STEM}\n" + "\n".join(f"{'ABCD'[i]}. {c}" for i, c in enumerate(choices)) + "\n"
    PAGES.append(page(len(PAGES) + 1, body))
SOURCE = join_pages(PAGES)


def copied(n):
    passage, choices = QUESTIONS[n - 1]
    return {"id": str(n), "questionHead": f"{passage}\n\n{STEM}", "answers": list(choices), "correctAnswer": 1}


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = json.dumps(self.replies.pop(0))
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": content})()})()]})()


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    return condition


results = []
results.append(check("page headers and footers are removed", "REVOLUTION" not in SOURCE and "Together, we can" not in SOURCE and "Passage number 12" in SOURCE))

perfect = [copied(n) for n in range(1, 11)]
results.append(check("an exact copy of the first 10 questions has no mismatch", copy_mismatches(perfect, SOURCE) == []))

drifted = [copied(n) for n in range(1, 4)]
drifted.append({**copied(4), "answers": ["century, but", "century but", "century; but", "century. But"]})
drifted.append({"id": "5", "questionHead": "According to the passage, what does the author think?", "answers": ["a", "b", "c", "d"], "correctAnswer": 0})
drifted.append({"id": "6", "questionHead": "Remember, to join two Independent Clauses, Use", "answers": [";", ","], "correctAnswer": 0})
problems = copy_mismatches(drifted, SOURCE)
results.append(check("rewritten choices, invented questions and header text are all flagged",
                     any("Question 4" in p for p in problems) and any("Question 5" in p for p in problems)
                     and any("Question 6" in p for p in problems)
                     and not any(p.startswith(("Question 1:", "Question 2:", "Question 3:")) for p in problems)))

reordered = [{**copied(1), "answers": list(reversed(QUESTIONS[0][1]))}]
swapped = [copied(2), copied(1)]
results.append(check("choices or questions out of their original order are flagged", copy_mismatches(reordered, SOURCE) != [] and copy_mismatches(swapped, SOURCE) != []))

client = FakeClient([{"mode": "copy", "questions": drifted}, {"mode": "copy", "questions": perfect}])
result = generate_questions(client, "model", SOURCE, COPY_NOTE, 10, "English", "medium")
heads = [q["questionHead"] for q in result["questions"]]
results.append(check("copy mode retries once and keeps the exact copy: 10 questions in source order",
                     len(client.calls) == 2 and client.calls[1]["temperature"] == 0.0 and result["unverified"] == []
                     and len(result["questions"]) == 10
                     and all(f"Passage number {n} " in h for n, h in zip(range(1, 11), heads))
                     and all(q["answers"] == list(QUESTIONS[i][1]) for i, q in enumerate(result["questions"]))
                     and heads[0].startswith("<p>") and heads[0].count("<p>") == 2))

messages = build_messages(SOURCE, COPY_NOTE, 4, "Arabic", "easy")
results.append(check("the teacher note sits in its own system message before the source",
                     messages[1]["role"] == "system" and COPY_NOTE in messages[1]["content"]
                     and messages[2]["role"] == "user" and COPY_NOTE not in messages[2]["content"]
                     and "Count: 4." in messages[0]["content"]))

generated = [{"id": str(n), "questionHead": f"New question {n}?", "answers": ["A. one", "B. two", "C. three", "D. four"], "correctAnswer": 2} for n in range(1, 6)]
client = FakeClient([{"mode": "generate", "questions": generated}])
result = generate_questions(client, "model", SOURCE, "generate 5 questions", 3, "English", "medium")
results.append(check("generate mode follows the note's count, skips copy checks and strips letter labels",
                     len(client.calls) == 1 and len(result["questions"]) == 5 and result["questions"][0]["answers"] == ["one", "two", "three", "four"]
                     and result["questions"][0]["correctAnswer"] == 2))

too_many = [{"id": str(n), "questionHead": f"Q{n}?", "answers": ["x", "y"], "correctAnswer": 0} for n in range(1, 16)]
client = FakeClient([{"mode": "generate", "questions": too_many}])
result = generate_questions(client, "model", SOURCE, "give me 15 questions", 5, "English", "medium")
results.append(check("never more than 10 questions", len(result["questions"]) == MAX_QUESTIONS))

print(f"TOTAL {sum(results)} passed, {len(results) - sum(results)} failed")
sys.exit(0 if all(results) else 1)
