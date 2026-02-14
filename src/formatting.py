def format_example(context: str, question: str, answer: str) -> str:
    """Format a complete QA example with context, question, and answer."""
    return f"""Bağlam:
{context}

Soru:
{question}

Cevap:
{answer}"""


def format_prompt(context: str, question: str) -> str:
    """Format a QA prompt without answer for inference."""
    return f"""Bağlam:
{context}

Soru:
{question}

Cevap:
"""
