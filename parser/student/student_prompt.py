"""Shared prompt contract for student training and inference."""

STUDENT_PROMPT = """Convert the sentence into the structured semantic JSON contract.

Return JSON only.

The output must have exactly these top-level fields:
- "assertions": a list of standalone semantic assertions
- "rules": a list of conditional semantic rules

Use the supported semantic predicates and canonical argument roles.

Important:
- Use PropertyOf for adjectival properties such as "X is red".
- Use Evaluation for actions and relations such as "X sees Y".
- Use rules for conditional or universal implications.
- Preserve negation with polarity="negative".
- Preserve factuality.
- Use canonical relation lemmas such as "see", not "sees".
- Do not invent extra assertions.
- Do not return generic triples, dependency parses, tokens, or explanations.

Sentence: {text}

JSON:"""


def build_student_prompt(sentence: str, context: str = "") -> str:
    """Build the shared student training/inference prompt."""
    prompt = STUDENT_PROMPT.format(text=sentence.strip())

    if context.strip():
        return f"Context: {context.strip()}\n{prompt}"

    return prompt
