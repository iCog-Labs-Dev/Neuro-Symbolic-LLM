"""Shared prompt contract for student training and inference."""

STUDENT_PROMPT = "Extract triples from this sentence as JSON:\nSentence: {text}\nJSON:"


def build_student_prompt(sentence: str, context: str = "") -> str:
    """Keep sentence-only prompts compatible with existing training data."""
    prompt = STUDENT_PROMPT.format(text=sentence.strip())
    if context.strip():
        return f"Context: {context.strip()}\n{prompt}"
    return prompt
