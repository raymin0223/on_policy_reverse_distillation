"""Explicit local judging policies. Unknown policies fail closed; training is unaffected."""

from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction


def check_output(actual, expected, input_text, policy):
    try:
        a, b = str(actual).split(), str(expected).split()
        kind = policy["kind"]
        if kind == "tokens":
            if policy.get("case_insensitive"):
                a, b = [x.casefold() for x in a], [x.casefold() for x in b]
            return a == b
        if kind == "numeric":
            if len(a) != len(b) or not a:
                return False
            for x, y in zip(a, b, strict=True):
                x, y = Decimal(x), Decimal(y)
                if not x.is_finite() or not y.is_finite():
                    return False
                tol = max(Decimal(policy["atol"]), Decimal(policy["rtol"]) * abs(y))
                if abs(x - y) > tol:
                    return False
            return True
        inp, out = iter(str(input_text).split()), iter(a)
        t = int(next(inp))
        for _ in range(t):
            if kind == "cf1971b":
                s, verdict = next(inp), next(out).upper()
                if verdict == "NO":
                    if len(set(s)) != 1:
                        return False
                elif verdict == "YES":
                    r = next(out)
                    if r == s or Counter(r) != Counter(s):
                        return False
                else:
                    return False
            elif kind == "cf1977b":
                x, n = int(next(inp)), int(next(out))
                if not 1 <= n <= 32:
                    return False
                digits = [int(next(out)) for _ in range(n)]
                if any(d not in (-1, 0, 1) for d in digits):
                    return False
                if any(x * y for x, y in zip(digits, digits[1:])):
                    return False
                if sum(d * (1 << i) for i, d in enumerate(digits)) != x:
                    return False
            elif kind == "cf1978c":
                n, k, verdict = int(next(inp)), int(next(inp)), next(out).upper()
                possible = k % 2 == 0 and 0 <= k <= n * n // 2
                if verdict == "NO":
                    if possible:
                        return False
                elif verdict == "YES":
                    p = [int(next(out)) for _ in range(n)]
                    if sorted(p) != list(range(1, n + 1)) or sum(abs(v - i) for i, v in enumerate(p, 1)) != k:
                        return False
                else:
                    return False
            elif kind == "cf1979c":
                n = int(next(inp))
                multipliers = [int(next(inp)) for _ in range(n)]
                first = int(next(out))
                if first == -1:
                    if sum((Fraction(1, k) for k in multipliers), Fraction()) < 1:
                        return False
                else:
                    bets = [first] + [int(next(out)) for _ in range(n - 1)]
                    if any(not 1 <= x <= 10**9 for x in bets):
                        return False
                    total = sum(bets)
                    if any(total >= x * k for x, k in zip(bets, multipliers, strict=True)):
                        return False
            else:
                return False
        return next(inp, None) is None and next(out, None) is None
    except (ValueError, TypeError, KeyError, StopIteration, InvalidOperation, OverflowError):
        return False
