"""Grader tests from the spec, not from the implementation: the reward is 1 only if the LAST \\boxed{} holds exactly
the gold number. Hostile model outputs included (fences, special tokens, think spans, empty, cut-off, over-long)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import grade  # noqa: E402

CASES = [  # (response, gold, expected)
    ("so the answer is \\boxed{18}.", "18", True),
    ("\\boxed{18}<|im_end|>", "18", True),
    ("\\boxed{18}<|endoftext|>", "18", True),
    ("<think>\n\\boxed{5}\n</think>\n\nThe answer is \\boxed{18}.", "18", True),   # last box counts
    ("The answer is \\boxed{18}. <think>maybe \\boxed{5}</think>", "18", False),  # last box is 5
    ("\\boxed{17} wait, \\boxed{18}", "18", True),
    ("\\boxed{18} wait, \\boxed{17}", "18", False),
    ("```python\nprint(18)\n```\nanswer 18", "18", False),                         # no box
    ("```\n\\boxed{18}\n```", "18", True),                                           # box inside a fence still a box
    ("", "18", False),
    ("\\boxed{}", "18", False),
    ("\\boxed{18", "18", False),                                                     # unclosed (cut off)
    ("The answer is \\boxed{1", "18", False),                                        # cut mid-number
    ("\\boxed{18.}", "18", True),
    ("\\boxed{18.0}", "18", True),
    ("\\boxed{018}", "18", True),
    ("\\boxed{-18}", "-18", True),
    ("\\boxed{-18}", "18", False),
    ("\\boxed{\\frac{1}{2}}", "0.5", True),
    ("\\boxed{\\dfrac{1}{2}}", "0.5", True),
    ("\\boxed{1/2}", "0.5", True),
    ("\\boxed{18,000}", "18000", True),
    ("\\boxed{18\\,000}", "18000", True),
    ("\\boxed{3, 4}", "34", False),                                                  # a list, not 34
    ("\\boxed{3,4}", "34", False),
    ("\\boxed{1,2345}", "12345", False),                                            # not a thousands group
    ("\\boxed{18 \\text{ dollars}}", "18", True),
    ("\\boxed{\\$18}", "18", True),
    ("\\boxed{18\\%}", "18", True),
    ("\\boxed{x = 18}", "18", True),
    ("\\boxed{18 and 20}", "18", False),
    ("\\boxed{\\boxed{18}}", "18", True),
    ("a" * 200000 + "\\boxed{18}", "18", True),                                     # over-long
    ("\\boxed{18}" + "x" * 200000, "18", True),
    ("\\boxed{" + "9" * 5000 + "}", "18", False),
    ("\\boxed{.5}", "0.5", True),
    ("\\boxed{\u221218}", "-18", True),
    ("\\boxed{18{,}000}", "18000", True),
]


def main():
    bad = []
    for resp, gold, exp in CASES:
        got = grade(resp, gold)
        if got != exp:
            bad.append((resp[:60], gold, exp, got))
    for b in bad:
        print("FAIL", b)
    print(f"{len(CASES) - len(bad)}/{len(CASES)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
