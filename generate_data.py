"""
generate_data.py

Generates a large, balanced synthetic dataset of feedback text labeled
high/medium/low priority. Unlike the Kaggle dataset, priority here is
derived directly from the severity of the language used, so there is
real signal for the model to learn from.

Usage:
    python generate_data.py

Outputs:
    feedback_dataset.csv (overwrites the file in the current folder)
"""

import random
import csv

random.seed(42)

PRODUCTS = [
    "app", "website", "dashboard", "mobile app", "platform", "software",
    "account", "system", "tool", "portal", "checkout page", "billing system",
]

# --- HIGH priority building blocks ---------------------------------------
HIGH_PROBLEMS = [
    "crashes every time I try to open it",
    "won't let me log in at all",
    "charged me twice for the same order",
    "lost all my saved data after the update",
    "is completely down and I can't access anything",
    "shows other users' private information on my screen",
    "keeps freezing during checkout so I can't complete my purchase",
    "deleted my entire project with no way to recover it",
    "hasn't sent my order confirmation and I was already charged",
    "keeps logging me out every few seconds, I can't get any work done",
    "is stuck on a blank screen no matter what I try",
    "rejected my payment but still took the money from my account",
]
HIGH_URGENCY = [
    "URGENT: ", "This is a serious problem. ", "Please fix this immediately, ",
    "I need help right now, ", "This is blocking my entire team, ",
    "I've already lost money because of this. ", "", "", "",
]
HIGH_IMPACT = [
    " This is affecting my whole team.",
    " I've tried everything and nothing works.",
    " This has happened three times this week.",
    " I can't do my job until this is fixed.",
    " This is a critical issue for our business.",
    "", "", "",
]

# --- MEDIUM priority building blocks ---------------------------------------
MEDIUM_PROBLEMS = [
    "search results are sometimes inaccurate",
    "loads slowly on my phone but works fine on desktop",
    "notifications arrive a few hours late",
    "export feature is missing the last page of the report",
    "login occasionally needs two attempts to work",
    "layout looks a bit off on tablets",
    "sync between devices isn't always reliable",
    "filter options don't cover everything I need",
    "confirmation emails take longer than they should to arrive",
    "sometimes shows an outdated version of my data",
    "could really use better error messages when something fails",
    "has a workaround but the main feature is still clunky",
]
MEDIUM_FRAMING = [
    "It's not blocking me, but ", "It still works, though ",
    "Not urgent, but ", "Would appreciate a fix soon: ",
    "Minor annoyance: ", "", "", "",
]
MEDIUM_IMPACT = [
    " Would be great if this got fixed in the next update.",
    " I can work around it for now.",
    " It happens fairly often though.",
    " Just flagging it so it's on your radar.",
    "", "", "",
]

# --- LOW priority building blocks ---------------------------------------
LOW_PROBLEMS = [
    "there's a small typo on the settings page",
    "the icon spacing looks slightly uneven",
    "the font size is a little inconsistent on the about page",
    "it would be nice to have a dark mode option",
    "a keyboard shortcut for saving would be convenient",
    "the loading animation could be a bit smoother",
    "more color theme options would be a nice touch",
    "the button placement could be a little more intuitive",
    "it might be cool to support more languages someday",
    "the changelog page could use a small design refresh",
]
LOW_FRAMING = [
    "Just a small suggestion: ", "Minor thing, ", "Not a big deal, but ",
    "Small idea: ", "Tiny nitpick: ", "", "", "",
]
LOW_IMPACT = [
    " Love the product overall though!",
    " Just wanted to mention it.",
    " No rush at all on this.",
    " Really enjoying using it otherwise.",
    "", "", "",
]


# ---- Wider vocabulary (added to improve generalization) -----------------
HIGH_PROBLEMS.extend([
    "rejects my login credentials even right after I reset them",
    "fails every time a customer tries to place an order",
    "declines every card and nobody can pay",
    "never sends the verification code so nobody can register",
    "locked my account and support hasn't answered",
    "shows a server error on every single page",
    "erased the changes I made when I hit save",
    "double billed several customers this morning",
    "won't process any refunds and people are angry",
    "was hacked and customer data may have leaked",
    "is down for everyone in our office",
    "stopped accepting orders and we are losing sales",
])
MEDIUM_PROBLEMS.extend([
    "takes several minutes to build large reports",
    "fails on big attachments although small ones go through",
    "asks me to log in again far too often",
    "hides the billing settings in a confusing menu",
    "cuts off columns on narrow screens",
    "needs a manual refresh before the newest data appears",
    "sends alerts noticeably late",
    "skips a few results when filters are on",
    "has a reset link that stops working too soon",
    "is missing a way to undo bulk changes",
])
LOW_PROBLEMS.extend([
    "the welcome text is a bit long",
    "the header logo looks a little soft",
    "more reaction icons would be fun",
    "there's a misspelled word in the footer",
    "some tooltip wording could be friendlier",
    "the sidebar could use a bit more padding",
    "a short tutorial video might be nice someday",
])
HIGH_SUBJECTS = ["The {p} ", "Whenever I use it, the {p} ", "Users report that the {p} ", "Today the {p} "]
MEDIUM_SUBJECTS = ["the {p} ", "lately the {p} ", "I noticed the {p} ", "on my side the {p} "]

COMPLIMENTS = [
    "Just wanted to say the new design looks great.",
    "Really happy with how smooth everything has been running lately.",
    "The support team was super helpful, thanks for that.",
    "Great job on the last update, it feels much faster.",
    "I appreciate how easy the onboarding process was.",
]

# Negated-compliment patterns: contain words that look like problem-signals
# ("issue", "bug", "problem", "crash") but are explicitly negated. These
# teach the model that word presence alone isn't enough - negation flips
# the meaning entirely, and these should be LOW priority.
# Built compositionally (not a fixed list) so there are enough unique
# combinations for the model to actually learn the negation pattern,
# rather than just memorizing a handful of exact sentences.
NEGATION_PREFIXES = [
    "No ", "Zero ", "Not a single ", "Haven't had any ",
    "Never experienced any ", "No sign of any ", "Not one ",
]
PROBLEM_WORDS = [
    "issues", "problems", "bugs", "crashes", "glitches",
    "complaints", "errors", "outages",
]
POSITIVE_SUFFIXES = [
    " - everything works perfectly.",
    " so far, really impressed.",
    " at all, smooth experience overall.",
    ", very happy with it.",
    ", this has been rock solid.",
    " here, everything runs great.",
    ", works flawlessly every time.",
]


POSITIVE_OPENERS = ["", "", "Wow, ", "Love it, ", "Great job, ", "Honestly amazing, ", "Really happy, "]


def gen_negated_compliment():
    return (
        random.choice(POSITIVE_OPENERS)
        + random.choice(NEGATION_PREFIXES).lower()
        + random.choice(PROBLEM_WORDS)
        + random.choice(POSITIVE_SUFFIXES)
    )

# --- HIGH priority: shouting / heavy urgency markers ----------------------
# Explicit ALL-CAPS and repeated punctuation urgency patterns, so the model
# learns these markers correlate with HIGH regardless of the exact words.
HIGH_SHOUTING = [
    "URGENT!!! Nothing works!!! Please fix this now!!!",
    "THIS IS BROKEN AND I NEED HELP IMMEDIATELY!!!",
    "EMERGENCY - the whole system is down right now!!!",
    "PLEASE HELP, I CANNOT ACCESS ANYTHING!!!",
    "This is CRITICAL, we are losing money every minute this stays down!!!",
    "STOP EVERYTHING - production is completely broken!!!",
    "I NEED THIS FIXED TODAY, it is completely unusable!!!",
    "SERIOUS ISSUE!!! All our customers are affected right now!!!",
    "This cannot wait, it's an emergency, please respond ASAP!!!",
    "DOWN DOWN DOWN - nothing is loading for anyone on the team!!!",
]


def gen_high(n):
    rows = []
    # Reserve ~15% of high examples for the shouting/urgency-marker pattern
    shouting_count = max(1, int(n * 0.15))
    for _ in range(shouting_count):
        rows.append((random.choice(HIGH_SHOUTING), "high"))

    for _ in range(n - shouting_count):
        product = random.choice(PRODUCTS)
        text = (
            random.choice(HIGH_URGENCY)
            + random.choice(HIGH_SUBJECTS).format(p=product)
            + random.choice(HIGH_PROBLEMS)
            + "."
            + random.choice(HIGH_IMPACT)
        )
        rows.append((text.strip(), "high"))
    return rows


def gen_medium(n):
    rows = []
    for _ in range(n):
        product = random.choice(PRODUCTS)
        text = (
            random.choice(MEDIUM_FRAMING)
            + random.choice(MEDIUM_SUBJECTS).format(p=product)
            + random.choice(MEDIUM_PROBLEMS)
            + "."
            + random.choice(MEDIUM_IMPACT)
        )
        text = text[0].upper() + text[1:]
        rows.append((text.strip(), "medium"))
    return rows


def gen_low(n):
    rows = []
    for _ in range(n):
        roll = random.random()
        if roll < 0.20:
            text = random.choice(COMPLIMENTS)
        elif roll < 0.40:
            # negated-compliment: contains problem-words but negated
            text = gen_negated_compliment()
        else:
            text = (
                random.choice(LOW_FRAMING)
                + random.choice(LOW_PROBLEMS)
                + "."
                + random.choice(LOW_IMPACT)
            )
            text = text[0].upper() + text[1:]
        rows.append((text.strip(), "low"))
    return rows



# ---- Terse, real-world style messages (what people actually type) --------
CRITICAL_FEATURES = ["login", "checkout", "payment page", "signup", "the app", "the site", "my account", "ordering", "the payment system"]
OTHER_FEATURES = ["search", "upload", "the dashboard", "export", "the report page", "notifications", "the mobile view", "filters", "the settings page"]
COSMETIC = ["colors", "layout", "logo", "design", "icons", "font", "theme", "homepage"]

def gen_terse(n):
    rows = []
    for _ in range(n):
        f = random.choice(CRITICAL_FEATURES)
        rows.append((random.choice([
            f"{f} is broken", f"{f} not working", f"{f} keeps crashing",
            f"{f} is down", f"{f} fails every time", f"{f} won't load at all",
            f"{f} crashes on start", f"{f} gives an error for everyone",
            f"cannot use {f}", f"{f} stopped working today",
            "cannot log in", "cant log in at all", "cannot pay", "can't sign up",
            "cannot access my account", "cannot place orders",
            "the app crashes on login", "app crashes when I open it",
        ]).replace("the the", "the"), "high"))
        f = random.choice(OTHER_FEATURES)
        rows.append((random.choice([
            f"{f} is slow", f"{f} is slow sometimes", f"{f} sometimes fails",
            f"{f} loads slowly", f"{f} glitches occasionally",
            f"{f} shows wrong data sometimes", f"{f} is a bit laggy",
            f"{f} not working properly for some files",
            f"{f} takes a while to respond", f"{f} is confusing to use",
        ]), "medium"))
        c = random.choice(COSMETIC)
        rows.append((random.choice([
            f"nice {c}", f"love the {c}", f"the {c} looks good",
            f"could use a darker {c}", f"small typo in the {c}",
            "thanks, great work", "thanks!", "great app overall",
            f"maybe tweak the {c} a little", f"the {c} is fine, just an idea",
        ]), "low"))
    return rows


# ---- Short, terse feedback (how people really type) ----------------------
SHORT_SUBJECTS = ["the app ", "app ", "the site ", "our website ", "the system ", "my account "]
SHORT_HIGH = ["crashes on login", "crashes on startup", "won't open", "is down",
              "is not loading at all", "keeps crashing", "is completely broken",
              "stopped working", "is not working at all"]
SHORT_HIGH_STANDALONE = ["cannot login", "can't sign in", "login is broken",
                         "payment not working", "checkout not working", "cannot pay",
                         "lost all my data", "payment failed", "site is down",
                         "everything crashed", "nothing works"]
SHORT_MEDIUM = ["is slow", "loads slowly", "lags sometimes", "has a small glitch",
                "sometimes fails to sync", "shows notifications late",
                "looks off on mobile", "search is inaccurate"]
SHORT_LOW = ["nice colors", "looks great", "love the design", "small typo",
             "font is a bit small", "could use a dark mode", "thanks, works well"]


def gen_short(n):
    rows = []
    for _ in range(n):
        if random.random() < 0.4:
            rows.append((random.choice(SHORT_HIGH_STANDALONE), "high"))
        else:
            rows.append((random.choice(SHORT_SUBJECTS) + random.choice(SHORT_HIGH), "high"))
        rows.append((random.choice(SHORT_SUBJECTS) + random.choice(SHORT_MEDIUM), "medium"))
        low = random.choice(SHORT_LOW)
        rows.append((low if random.random() < 0.6 else "Small note: " + low, "low"))
    return rows



# ---- Negated behavior: "never crashes", "hasn't failed once" -> LOW --------
NEG_VERBS = [
    {"base": "crash", "s": "crashes", "part": "crashed", "ing": "crashing"},
    {"base": "freeze", "s": "freezes", "part": "frozen", "ing": "freezing"},
    {"base": "fail", "s": "fails", "part": "failed", "ing": "failing"},
    {"base": "lag", "s": "lags", "part": "lagged", "ing": "lagging"},
    {"base": "break", "s": "breaks", "part": "broken", "ing": "breaking"},
]
NEG_FORMS = ["never {s}", "hasn't {part} once", "doesn't {base} anymore",
             "no longer {s}", "stopped {ing}", "hasn't {part} since the update"]
NEG_CONTEXTS = ["", " on login", " at checkout", " when saving",
                " during uploads", " on startup", " on my phone"]
NEG_SUBJECTS = ["the app ", "the site ", "it ", "our dashboard ", "the checkout "]
NEG_SUFFIXES = [", works great", ", really stable now", ", very happy with it",
                " - smooth experience", ", thanks for the fixes", ""]


def gen_negated_behavior(n):
    rows = []
    for _ in range(n):
        text = (random.choice(NEG_SUBJECTS)
                + random.choice(NEG_FORMS).format(**random.choice(NEG_VERBS))
                + random.choice(NEG_CONTEXTS)
                + random.choice(NEG_SUFFIXES) + ".")
        rows.append((text[0].upper() + text[1:], "low"))
    return rows


def main(per_class=500):
    rows = gen_high(per_class) + gen_medium(per_class) + gen_low(per_class) + gen_terse(per_class // 4)
    rows += gen_short(80)
    rows += gen_negated_behavior(120)
    random.shuffle(rows)

    # de-duplicate exact repeats
    seen = set()
    unique_rows = []
    for text, label in rows:
        if text not in seen:
            seen.add(text)
            unique_rows.append((text, label))

    with open("feedback_dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "priority"])
        writer.writerows(unique_rows)

    print(f"Generated {len(unique_rows)} unique rows (requested {per_class * 3})")
    from collections import Counter
    print(Counter(label for _, label in unique_rows))


if __name__ == "__main__":
    main(per_class=500)