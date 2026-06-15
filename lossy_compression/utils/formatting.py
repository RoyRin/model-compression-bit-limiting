import re, textwrap, difflib


def _extract_python_code(answer: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)\s*```",
                  answer,
                  flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    answer = answer.strip()
    answer = re.sub(r"^```+|```+$", "", answer, flags=re.MULTILINE).strip()
    # If there's non-code prose, you could add heuristics here; for now return all.
    return answer


def _dedent_and_normalize(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = code.expandtabs(4)
    code = textwrap.dedent(code)
    code = "\n".join(line.rstrip() for line in code.split("\n"))
    if not code.endswith("\n"):
        code += "\n"
    return code


def _try_black(code: str) -> str:
    try:
        import black
        mode = black.Mode()
        return black.format_str(code, mode=mode)
    except Exception:
        return code


def _try_autopep8(code: str) -> str:
    try:
        import autopep8
        return autopep8.fix_code(
            code,
            options={
                "aggressive": 2,
                "max_line_length": 88,
                "experimental": True
            },
        )
    except Exception:
        return code


def _only_whitespace_changes(original: str, rewritten: str) -> bool:
    ow = re.sub(r"\s+", " ", original).strip()
    rw = re.sub(r"\s+", " ", rewritten).strip()
    return ow == rw


def rewrite_answer_to_be_syntactically_correct(
        answer: str,
        original_prompt: str,
        *,
        allow_non_whitespace_changes: bool = False) -> str:
    """
    Normalize/format Python code without executing or compiling it.
    By default we only allow whitespace/indent changes to avoid content loss.
    Set allow_non_whitespace_changes=True to permit minimal fixer edits.
    """
    original_code = _extract_python_code(answer)
    if not original_code:
        return answer  # nothing to do

    # 1) normalize basics
    code = _dedent_and_normalize(original_code)

    # 2) formatters (black first, then autopep8 as an optional fixer)
    formatted = _try_black(code)
    formatted = _try_autopep8(formatted)

    # 3) guard: avoid accidental token deletions unless explicitly allowed
    if not allow_non_whitespace_changes and not _only_whitespace_changes(
            original_code, formatted):
        formatted = _dedent_and_normalize(
            original_code)  # safest normalized original

    return f"```python\n{formatted}```"


EXAMPLES_BLOCK = """### EXAMPLES OF EXPECTED BEHAVIOR

BAD (adds extra text) ❌:
Here is the completed function:

def truncate_number(number: float) -> float:
    return number - int(number)

GOOD (only code, no extra text) ✅:
def truncate_number(number: float) -> float:
    \"\"\"Return the fractional part of a number.
    >>> truncate_number(3.14)
    0.14
    >>> truncate_number(-2.75)
    0.25
    \"\"\"
    frac = number - int(number)
    return frac if frac >= 0 else 1.0 + frac


BAD (drops other functions) ❌:
def make_palindrome(string: str) -> str:
    ...

GOOD (preserves all functions, just fills in) ✅ :
def is_palindrome(string: str) -> bool:
    \"\"\" Test if given string is a palindrome \"\"\"
    return string == string[::-1]


def make_palindrome(string: str) -> str:
    \"\"\" Find the shortest palindrome that begins with a supplied string. \"\"\"
    for i in range(len(string) + 1):
        if is_palindrome(string[i:]):
            return string + string[:i][::-1]
"""
