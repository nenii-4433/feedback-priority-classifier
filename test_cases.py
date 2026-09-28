"""
test_cases.py - honest accuracy check.

These are hand-written examples with wording that generate_data.py never
uses, so the score here shows how the model handles unfamiliar phrasing
(the train/test split in train_model.py only measures template-style text).

Usage:  python test_cases.py
Add your own real examples to CASES to make this check more meaningful.
"""

from predict import predict_priority

CASES = [
    ("cant log in at all", "high"),
    ("payment page not working", "high"),
    ("the site keeps crashing", "high"),
    ("search is slow sometimes", "medium"),
    ("upload glitches now and then", "medium"),
    ("thanks, nice design", "low"),
    ("typo on the pricing page", "low"),
    ("I was billed twice and support is not replying", "high"),
    ("can't sign in, password reset email never arrives", "high"),
    ("entire site returns 500 error for all our users", "high"),
    ("my account got hacked and money is missing", "high"),
    ("lost three days of work, the save button did nothing", "high"),
    ("payment failed but money was deducted", "high"),
    ("nothing is working everything crashed", "high"),
    ("URGENT!!! nothing works!!! fix now!!!", "high"),
    ("our whole team is locked out since this morning", "high"),
    ("orders are not going through at checkout", "high"),
    ("reports take about a minute to generate, would be nice if faster", "medium"),
    ("the mobile view cuts off the last column of the table", "medium"),
    ("sometimes the page needs a refresh to show new data", "medium"),
    ("email alerts arrive an hour late", "medium"),
    ("search misses some results when I use filters", "medium"),
    ("upload fails on large files but small ones work", "medium"),
    ("the export button is hard to find", "medium"),
    ("password reset link expires too quickly", "medium"),
    ("would love a dark mode someday", "low"),
    ("the logo on the login page looks slightly blurry", "low"),
    ("thanks, the new layout is really nice", "low"),
    ("wow this app is amazing, zero issues at all", "low"),
    ("everything is fine except the dark mode toggle is missing", "low"),
    ("haven't had a single problem, works great", "low"),
    ("maybe add more emoji options in comments", "low"),
    ("small typo in the footer, it says 'copywrite'", "low"),
    ("great support team, very friendly", "low"),
    ("could the welcome text be a bit shorter", "low"),
]

wrong = []
for text, expected in CASES:
    got, conf = predict_priority(text)
    if got != expected:
        wrong.append((text, expected, got, conf[got]))

correct = len(CASES) - len(wrong)
print(f"Correct: {correct}/{len(CASES)} ({correct / len(CASES):.0%})\n")
if wrong:
    print("Misses:")
    for text, expected, got, c in wrong:
        print(f"  expected {expected.upper():6} got {got.upper():6} ({c:.0%})  {text}")