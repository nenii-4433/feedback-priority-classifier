"""
test_model.py - automated tests for the feedback priority classifier.

Run:   python -m pytest test_model.py -v
Needs: pip install pytest   (and a trained model: python train_model.py)

Works with either backend (embedding or TF-IDF). What is tested:
  1. Output format is always valid
  2. Odd inputs never crash it (empty, emoji, huge, non-English)
  3. Accuracy on 22 NEW hand-written cases (separate from test_cases.py)
  4. Recall on HIGH priority (missing an urgent item is the costly mistake)
  5. Behavior rules: harmless changes must not flip the label,
     adding severity must not lower priority, negation must not stay HIGH
  6. Training data sanity
  7. Batch CSV mode
"""

import os
import pandas as pd
import pytest

from predict import predict_priority, run_batch

LABELS = {"high", "medium", "low"}
RANK = {"low": 0, "medium": 1, "high": 2}

# Fresh cases: typos, slang, short text, negation. Not used in training.
CASES = [
    ("cant login since yesterday pls help", "high"),
    ("money got deducted twice from my card", "high"),
    ("whole website is showing 404 for our customers", "high"),
    ("app deleted all my saved projects!!", "high"),
    ("getting charged every day after i cancelled", "high"),
    ("server is down, nobody in the office can work", "high"),
    ("the payment button does nothing, we cannot sell anything", "high"),
    ("my account was hacked, someone changed my email", "high"),
    ("images take too long to load on mobile data", "medium"),
    ("sometimes the dashboard shows yesterdays numbers", "medium"),
    ("notifications dont always show up", "medium"),
    ("the app logs me out every couple of hours", "medium"),
    ("csv download has the columns in the wrong order", "medium"),
    ("search feels slow when there are many results", "medium"),
    ("the calendar view is off by one day for some users", "medium"),
    ("love the new update, thanks team", "low"),
    ("it would be cool to have a widget for the home screen", "low"),
    ("the about page has a small grammar mistake", "low"),
    ("can you make the buttons slightly rounder", "low"),
    ("no problems so far, everything works fine", "low"),
    ("great app", "low"),
    ("maybe add a few more color themes", "low"),
]


# ---------- 1. output format ----------
@pytest.mark.parametrize("text", [t for t, _ in CASES[:6]])
def test_output_format(text):
    label, probs = predict_priority(text)
    assert label in LABELS
    assert set(probs) == LABELS
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert probs[label] == max(probs.values())


# ---------- 2. odd inputs must not crash ----------
@pytest.mark.parametrize("text", [
    "", "   ", "!!!", "????", "a", "😡😡😡", "12345",
    "x" * 5000, "crash " * 500,
    "کام نہیں کر رہا",            # Urdu
    "app kaam nahi kar raha",      # Roman Urdu
    "<script>alert(1)</script>", "SELECT * FROM users;",
])
def test_odd_inputs_do_not_crash(text):
    label, _ = predict_priority(text)
    assert label in LABELS


# ---------- 3 + 4. accuracy on unseen cases ----------
def _misses():
    out = []
    for text, expected in CASES:
        got, probs = predict_priority(text)
        if got != expected:
            out.append(f"expected {expected.upper()} got {got.upper()} "
                       f"({probs[got]:.0%}): {text}")
    return out


def test_overall_accuracy():
    misses = _misses()
    acc = 1 - len(misses) / len(CASES)
    assert acc >= 0.80, f"accuracy {acc:.0%} < 80%\n" + "\n".join(misses)


def test_high_priority_recall():
    high = [t for t, y in CASES if y == "high"]
    found = sum(predict_priority(t)[0] == "high" for t in high)
    assert found / len(high) >= 0.80, f"only {found}/{len(high)} HIGH cases found"


# ---------- 5. behavior rules ----------
INVARIANCE_TEXTS = [t for t, _ in CASES]


@pytest.mark.parametrize("text", INVARIANCE_TEXTS)
def test_trailing_space_and_period_do_not_change_label(text):
    base = predict_priority(text)[0]
    assert predict_priority(text + "  ")[0] == base
    assert predict_priority(text.rstrip(".") + ".")[0] == base


ESCALATIONS = [
    "the export button is hard to find",
    "the report page is a bit slow",
    "the logo looks slightly blurry",
    "search misses some results",
]
SEVERITY_SUFFIX = " and now customers cannot pay at all, we are losing money"


@pytest.mark.parametrize("base", ESCALATIONS)
def test_adding_severity_never_lowers_priority(base):
    before = RANK[predict_priority(base)[0]]
    after = RANK[predict_priority(base + SEVERITY_SUFFIX)[0]]
    assert after >= before


NEGATED = [
    "the app never crashes on login, works great",
    "checkout hasn't failed once since the update, very reliable",   # near the training pattern
    "nothing has broken this month, super stable",                    # different wording
    "no more freezing on my phone, thanks",                           # different wording
]


def test_problem_words_alone_mean_high():
    assert predict_priority("the app crashes on login")[0] == "high"


@pytest.mark.parametrize("text", NEGATED)
def test_negation_is_not_treated_as_a_problem(text):
    assert predict_priority(text)[0] != "high"


# ---------- 6. training data sanity ----------
def test_dataset_is_clean_and_balanced():
    df = pd.read_csv("feedback_dataset.csv")
    assert list(df.columns[:2]) == ["text", "priority"]
    assert df["text"].notna().all() and df["priority"].notna().all()
    assert set(df["priority"].str.lower()) <= LABELS
    shares = df["priority"].str.lower().value_counts(normalize=True)
    assert shares.min() >= 0.15, f"a class is under-represented:\n{shares}"
    assert df["text"].duplicated().mean() < 0.05


# ---------- 7. batch mode ----------
def test_batch_csv_mode(tmp_path):
    src = tmp_path / "in.csv"
    pd.DataFrame({"text": ["site is down", "love the design", "app is slow"]}).to_csv(src, index=False)
    run_batch(str(src))
    out = pd.read_csv(tmp_path / "in_predicted.csv")
    assert len(out) == 3
    assert {"predicted_priority", "confidence"} <= set(out.columns)
    assert set(out["predicted_priority"]) <= LABELS
    assert out["confidence"].between(0, 1).all()