# src/data/label_rules.py
"""
Simple keyword lists for weak labelling.
English and Persian (Farsi) tokens are provided.
Matching is *case‑insensitive* and diacritic‑insensitive.
"""


# Helper: collapse both EN and FA lists into lowercase once
def _l(words):
    return [w.lower() for w in words]


SUPPORT = _l(
    [
        # English
        "refund",
        "issue",
        "bug",
        "help",
        "complaint",
        "problem",
        # Farsi
        "بازپرداخت",
        "مشکل",
        "خطا",
        "کمک",
        "شکایت",
        "مسئله",
    ]
)

SALES = _l(
    [
        "price",
        "quotation",
        "buy",
        "cost",
        "discount",
        "قیمت",
        "پیش‌فاکتور",
        "خرید",
        "هزینه",
        "تخفیف",
    ]
)

PARTNERSHIP = _l(
    [
        "collaboration",
        "partner",
        "joint venture",
        "synergy",
        "cooperate",
        "همکاری",
        "شریک",
        "سرمایه‌گذاری مشترک",
        "هم‌افزایی",
        "مشارکت",
    ]
)

SPAM = _l(
    [
        "casino",
        "lottery",
        "viagra",
        "bitcoins",
        "winner",
        "free money",
        "کازینو",
        "لاتاری",
        "ویارا",
        "بیتکوین",
        "برنده",
        "کسب درآمد رایگان",
    ]
)
