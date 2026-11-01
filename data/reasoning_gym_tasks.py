from __future__ import annotations
import itertools
import json
import random
from collections import deque
from pathlib import Path
from typing import Any

COLOR_CUBE_ROTATION_DATASETS = ("color_cube_rotation",)
COMMON_NAMES = [
    "Emma",
    "Liam",
    "Olivia",
    "Noah",
    "Ava",
    "Ethan",
    "Sophia",
    "Mason",
    "Isabella",
    "William",
    "Mia",
    "James",
    "Charlotte",
    "Benjamin",
    "Amelia",
    "Lucas",
    "Harper",
    "Henry",
    "Evelyn",
    "Alexander",
    "Abigail",
    "Michael",
    "Emily",
    "Daniel",
    "Elizabeth",
    "Jacob",
    "Sofia",
    "Logan",
    "Avery",
    "Jackson",
    "Ella",
    "Sebastian",
    "Scarlett",
    "Jack",
    "Grace",
    "Aiden",
    "Chloe",
    "Owen",
    "Victoria",
    "Samuel",
    "Riley",
    "Matthew",
    "Aria",
    "Joseph",
    "Lily",
    "Luke",
    "Aurora",
    "David",
    "Zoey",
    "Oliver",
    "Penelope",
]
KNIGHT_KNAVE_PAIRS = [
    ("a knight", "a knave"),
    ("a pioneer", "a laggard"),
    ("a saint", "a sinner"),
    ("a hero", "a villain"),
    ("an angel", "a devil"),
    ("an altruist", "an egoist"),
    ("a sage", "a fool"),
]
PREFIX = "A very special island is inhabited only by {knight}s and {knave}s. {Knight}s always tell the truth, and {knave}s always lie. "
POSTFIX = "So who is {a_knight} and who is {a_knave}?"
TEMPLATES = [
    "{name} said that {content}.",
    "{name} told you that {content}.",
    '{name} said, "{content}."',
    '{name} stated, "{content}".',
    'According to {name}, "{content}".',
    'In {name}\'s words: "{content}".',
    '{name} remarked, "{content}".',
    '"{content}," {name} declared.',
    '{name} was heard saying, "{content}".',
    "{name} expressed that {content}.",
    '"{content}" - {name}.',
    'As {name} put it, "{content}".',
    '{name} asserted: "{content}".',
    '"{content}," {name} mentioned.',
    '{name} commented, "{content}".',
    'In a statement by {name}: "{content}".',
    '{name} noted, "{content}".',
    '"{content}," {name} claimed.',
]
BINARY_MATRIX_QUESTION_TEMPLATE = "Given a square matrix, your job is to find the taxicab (Manhattan) distance of the nearest 0 for each cell.\n\nThe output should be a matrix of the same size as the input matrix, where each cell contains the distance to the nearest 0.\n\nFind the distance to the nearest 0 for each cell in the matrix below:\n{matrix}"
COUNTDOWN_PROMPT_TEMPLATES = (
    "Using all the numbers {numbers}, create an expression that equals {target}.\nYou can only use each number once.",
    "Find a way to make {target} using all of these numbers: {numbers}.\nEach number can only be used once.",
    "Calculate {target} using all of these numbers: {numbers}.\nEach number may be used at most once.",
)


def _join_natural(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _final_instruction() -> str:
    return "Please reason step by step. Put your final answer in \\boxed{} format."


class KKProblemSampler:
    """Sample Knights and Knaves logical statements with a unique solution."""

    def __init__(self, rand_seed: int, n_people: int, depth_constraint: int = 2, width_constraint: int = 2):
        self.rng = random.Random(rand_seed)
        self.n_people = n_people
        self.depth_constraint = depth_constraint
        self.width_constraint = width_constraint

    def sample(self) -> tuple:
        statements = tuple(
            (self._sample_statement(person_id, self.depth_constraint) for person_id in range(self.n_people))
        )
        return statements

    def sample_valid_problems(
        self,
        n_problems: int,
        max_retry: int = 1000,
        skip_no_solution: bool = True,
        skip_multiple_solutions: bool = True,
    ) -> list[dict[str, Any]]:
        problems: list[dict[str, Any]] = []
        unique_statements = set()
        for _ in range(n_problems):
            for _ in range(max_retry):
                statements = self.sample()
                if statements in unique_statements:
                    continue
                solutions = find_solution(statements)
                if len(solutions) == 0 and skip_no_solution:
                    continue
                if len(solutions) > 1 and skip_multiple_solutions:
                    continue
                problems.append(
                    {
                        "statements": statements,
                        "solution": solutions[0] if solutions else None,
                        "all_solutions": solutions,
                    }
                )
                unique_statements.add(statements)
                break
        return problems

    def _sample_statement(self, person_id: int, depth_constraint: int) -> tuple:
        dice = self.rng.randrange(6)
        if depth_constraint == 1 or dice == 0:
            while True:
                knight_or_knave = self.rng.choice(["telling-truth", "lying"])
                person = self.rng.randrange(self.n_people)
                if not (knight_or_knave == "lying" and person == person_id):
                    return (knight_or_knave, person)
        if dice == 1:
            return ("not", self._sample_statement(person_id, depth_constraint - 1))
        if dice in (2, 3):
            operator = "and" if dice == 2 else "or"
            n_substatements = self.rng.randint(2, self.width_constraint)
            return (operator,) + self._sample_substatements(person_id, depth_constraint, n_substatements)
        operator = "->" if dice == 4 else "<=>"
        return (operator,) + self._sample_substatements(person_id, depth_constraint, 2)

    def _sample_substatements(
        self, person_id: int, depth_constraint: int, count: int, dedup: bool = True
    ) -> tuple:
        sub_statements = []
        dedup_set = set()
        while len(sub_statements) < count:
            stmt = self._sample_statement(person_id, depth_constraint - 1)
            if dedup and stmt in dedup_set:
                continue
            dedup_set.add(stmt)
            sub_statements.append(stmt)
        return tuple(sub_statements)


class KKProblemFormatter:
    """Format sampled Knights and Knaves statements as natural language."""

    def __init__(self, rand_seed: int, problem: dict[str, Any]):
        self.rng = random.Random(rand_seed)
        self.problem = problem

    def format_problem(self) -> dict[str, Any]:
        statements = self.problem["statements"]
        n_people = len(statements)
        names = self.rng.sample(COMMON_NAMES, n_people)
        pair = self.rng.choice(KNIGHT_KNAVE_PAIRS)
        knight_knave = {
            "knight": pair[0].split()[1],
            "knave": pair[1].split()[1],
            "a_knight": pair[0],
            "a_knave": pair[1],
        }
        knight_knave["Knight"] = knight_knave["knight"].capitalize()
        knight_knave["Knave"] = knight_knave["knave"].capitalize()
        text = PREFIX.format(**knight_knave)
        text += f"You meet {n_people} inhabitants: {_join_natural(names)}."
        for i, stmt in enumerate(statements):
            tmpl = self.rng.choice(TEMPLATES)
            content = self._format_statement(names, knight_knave, stmt)
            text += " " + tmpl.format(name=names[i], content=content)
        text += " " + POSTFIX.format(**knight_knave)
        format_items = [f"{name} is {knight_knave['a_knight']}/{knight_knave['a_knave']}" for name in names]
        text += f' (Format your answer like: "{_join_natural(format_items)}")'
        if self.problem["solution"] is None:
            solution_text = "No valid solution exists."
        else:
            solution_items = []
            for name, indicator in zip(names, self.problem["solution"], strict=True):
                role = knight_knave["a_knight"] if indicator else knight_knave["a_knave"]
                solution_items.append(f"{name} is {role}")
            solution_text = _join_natural(solution_items) + "."
        return {
            "quiz": text,
            "names": names,
            "knight_knave": knight_knave,
            "solution": self.problem["solution"],
            "solution_text": solution_text,
        }

    def _format_statement(self, names: list[str], knight_knave: dict[str, str], statement: tuple) -> str:
        if statement[0] in ("telling-truth", "lying"):
            return self._format_knight_knave(names, knight_knave, statement)
        if statement[0] == "not":
            if statement[1][0] in ("telling-truth", "lying"):
                complementary = (
                    "lying" if statement[1][0] == "telling-truth" else "telling-truth",
                    statement[1][1],
                )
                return self._format_knight_knave(names, knight_knave, complementary)
            inner_content = self._format_statement(names, knight_knave, statement[1])
            if statement[1][0] not in ("telling-truth", "lying"):
                inner_content = f"({inner_content})"
            return f"it is not the case that {inner_content}"
        if statement[0] in ("and", "or"):
            formatted = []
            for sub_stmt in statement[1:]:
                sub_content = self._format_statement(names, knight_knave, sub_stmt)
                if sub_stmt[0] not in ("telling-truth", "lying"):
                    sub_content = f"({sub_content})"
                formatted.append(sub_content)
            return f" {statement[0]} ".join(formatted)
        if statement[0] == "->":
            antecedent = self._format_statement(names, knight_knave, statement[1])
            consequent = self._format_statement(names, knight_knave, statement[2])
            if statement[1][0] not in ("telling-truth", "lying"):
                antecedent = f"({antecedent})"
            if statement[2][0] not in ("telling-truth", "lying"):
                consequent = f"({consequent})"
            return f"if {antecedent} then {consequent}"
        if statement[0] == "<=>":
            left = self._format_statement(names, knight_knave, statement[1])
            right = self._format_statement(names, knight_knave, statement[2])
            if statement[1][0] not in ("telling-truth", "lying"):
                left = f"({left})"
            if statement[2][0] not in ("telling-truth", "lying"):
                right = f"({right})"
            return f"{left} if and only if {right}"
        raise ValueError(f"Unknown statement type: {statement[0]}")

    @staticmethod
    def _format_knight_knave(names: list[str], knight_knave: dict[str, str], statement: tuple) -> str:
        role = {"telling-truth": knight_knave["a_knight"], "lying": knight_knave["a_knave"]}[statement[0]]
        return f"{names[statement[1]]} is {role}"


def find_solution(statements: tuple) -> list[tuple[bool, ...]]:
    n_people = len(statements)
    single_statement = ("and",) + tuple(
        (("<=>", ("telling-truth", i), statements[i]) for i in range(n_people))
    )
    return [
        assignments
        for assignments in itertools.product([True, False], repeat=n_people)
        if test_satisfiability(single_statement, assignments)
    ]


def test_satisfiability(statement: tuple, assignments: tuple[bool, ...]) -> bool:
    if statement[0] == "telling-truth":
        return assignments[statement[1]]
    if statement[0] == "lying":
        return not assignments[statement[1]]
    if statement[0] == "not":
        return not test_satisfiability(statement[1], assignments)
    if statement[0] == "and":
        return all((test_satisfiability(statement[i], assignments) for i in range(1, len(statement))))
    if statement[0] == "or":
        return any((test_satisfiability(statement[i], assignments) for i in range(1, len(statement))))
    if statement[0] == "->":
        return not test_satisfiability(statement[1], assignments) or test_satisfiability(
            statement[2], assignments
        )
    if statement[0] == "<=>":
        left = test_satisfiability(statement[1], assignments)
        right = test_satisfiability(statement[2], assignments)
        return left == right
    raise KeyError(f"Unknown statement: {statement}")


def generate_knights_knaves_problems(
    num_problems: int, seed: int, n_people: int = 5, depth_constraint: int = 3, width_constraint: int = 3
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    formatted_problems = []
    unique_statements = set()
    attempts = 0
    max_attempts = max(num_problems * 100, 1000)
    consecutive_failures = 0
    max_consecutive_failures = 1000
    while len(formatted_problems) < num_problems and attempts < max_attempts:
        attempts += 1
        sampler = KKProblemSampler(
            rand_seed=rng.randint(0, 2**31 - 1),
            n_people=n_people,
            depth_constraint=depth_constraint,
            width_constraint=width_constraint,
        )
        problems = sampler.sample_valid_problems(1, skip_no_solution=True, skip_multiple_solutions=True)
        if not problems:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                break
            continue
        problem = problems[0]
        statements = problem["statements"]
        if statements in unique_statements:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                break
            continue
        consecutive_failures = 0
        unique_statements.add(statements)
        formatter = KKProblemFormatter(rand_seed=rng.randint(0, 2**31 - 1), problem=problem)
        formatted_problems.append(formatter.format_problem())
    if len(formatted_problems) < num_problems:
        raise RuntimeError(
            f"Generated only {len(formatted_problems)} of {num_problems} Knights and Knaves problems. Try reducing --num-problems or increasing the Knights and Knaves depth/width/people settings."
        )
    return formatted_problems


def generate_binary_matrix(rng: random.Random, n: int, p_zero: float = 0.25) -> list[list[int]]:
    numbers = [0] + [0 if rng.random() < p_zero else 1 for _ in range(n * n - 1)]
    rng.shuffle(numbers)
    return [numbers[i * n : (i + 1) * n] for i in range(n)]


def compute_binary_matrix_distances(matrix: list[list[int]]) -> list[list[int]]:
    n = len(matrix)
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    visited = set()
    queue = deque()
    output = [[10**9] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            if matrix[r][c] == 0:
                output[r][c] = 0
                visited.add((r, c))
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = (r + dr, c + dc)
            if 0 <= nr < n and 0 <= nc < n and ((nr, nc) not in visited):
                output[nr][nc] = output[r][c] + 1
                visited.add((nr, nc))
                queue.append((nr, nc))
    return output


def matrix_to_str(matrix: list[list[int]]) -> str:
    return "\n".join((" ".join((str(x) for x in row)) for row in matrix))


def generate_binary_matrix_problems(
    num_problems: int, seed: int, min_n: int = 3, max_n: int = 5, p_zero: float = 0.25
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    problems = []
    for _ in range(num_problems):
        n = rng.randint(min_n, max_n)
        matrix = generate_binary_matrix(rng, n, p_zero)
        problems.append({"matrix": matrix, "solution": compute_binary_matrix_distances(matrix), "n": n})
    return problems


def binary_matrix_problem_key(problem: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    return tuple((tuple(row) for row in problem["matrix"]))


def dedup_binary_matrix_train_against_test(
    problems: list[dict[str, Any]],
    test_size: int,
    seed: int,
    min_n: int = 3,
    max_n: int = 5,
    p_zero: float = 0.25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Keep the held-out test split fixed and refill train without test overlap."""
    train_problems, test_problems = split_train_test(problems, test_size=test_size)
    target_train_size = len(train_problems)
    test_keys = {binary_matrix_problem_key(problem) for problem in test_problems}
    deduped_train = [
        problem for problem in train_problems if binary_matrix_problem_key(problem) not in test_keys
    ]
    removed_train_rows = target_train_size - len(deduped_train)
    rng = random.Random(seed)
    attempts = 0
    accepted_replacements = 0
    max_attempts = max(removed_train_rows * 1000, 10000)
    while len(deduped_train) < target_train_size and attempts < max_attempts:
        attempts += 1
        n = rng.randint(min_n, max_n)
        matrix = generate_binary_matrix(rng, n, p_zero)
        problem = {"matrix": matrix, "solution": compute_binary_matrix_distances(matrix), "n": n}
        if binary_matrix_problem_key(problem) in test_keys:
            continue
        deduped_train.append(problem)
        accepted_replacements += 1
    if len(deduped_train) < target_train_size:
        raise RuntimeError(
            f"Could only refill Binary Matrix train split to {len(deduped_train)} of {target_train_size} rows after {attempts} attempts. Try changing the seed or size bounds."
        )
    stats = {
        "dedup_train_against_test": True,
        "dedup_removed_train_rows": removed_train_rows,
        "dedup_refill_attempts": attempts,
        "dedup_refill_accepted": accepted_replacements,
    }
    return (deduped_train, test_problems, stats)


def _generate_countdown_expression(
    rng: random.Random,
    min_numbers: int,
    max_numbers: int,
    min_value: int,
    max_value: int,
    operators: tuple[str, ...],
) -> tuple[str, list[int], int]:
    """Fold a random sequence of operators over random integers (left-to-right).

    Mirrors reasoning-gym's countdown generator: each operator is applied to the
    accumulated value so far. Division only picks a remaining number that divides
    the accumulator evenly (otherwise it falls back to subtraction), so the target
    is always an integer.
    """
    from fractions import Fraction

    num_terms = rng.randint(min_numbers, max_numbers)
    numbers = [rng.randint(min_value, max_value) for _ in range(num_terms)]
    acc = Fraction(numbers[0])
    expr_str = str(numbers[0])
    for i in range(1, num_terms):
        op = rng.choice(operators)
        if op == "/":
            current = int(acc)
            divisor_idx = None
            order = list(range(i, num_terms))
            rng.shuffle(order)
            for j in order:
                if numbers[j] != 0 and current % numbers[j] == 0:
                    divisor_idx = j
                    break
            if divisor_idx is None:
                op = "-"
            else:
                numbers[i], numbers[divisor_idx] = (numbers[divisor_idx], numbers[i])
        value = numbers[i]
        if op == "+":
            acc = acc + value
        elif op == "-":
            acc = acc - value
        elif op == "*":
            acc = acc * value
        else:
            acc = acc / value
        expr_str = f"({expr_str} {op} {value})"
    return (expr_str, numbers, int(acc))


def generate_countdown_problems(
    num_problems: int,
    seed: int,
    min_numbers: int = 4,
    max_numbers: int = 6,
    min_value: int = 1,
    max_value: int = 100,
    min_target: int = 100,
    max_target: int = 999,
    operators: tuple[str, ...] = ("+", "-", "*", "/"),
    shuffle: bool = True,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    problems: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    attempts = 0
    max_attempts = max(num_problems * 200, 10000)
    while len(problems) < num_problems and attempts < max_attempts:
        attempts += 1
        expr_str, numbers, target = _generate_countdown_expression(
            rng, min_numbers, max_numbers, min_value, max_value, operators
        )
        if not min_target <= target <= max_target:
            continue
        if shuffle:
            rng.shuffle(numbers)
        key = (tuple(sorted(numbers)), target)
        if key in seen:
            continue
        seen.add(key)
        problems.append({"numbers": numbers, "target": target, "expression": expr_str})
    if len(problems) < num_problems:
        raise RuntimeError(
            f"Generated only {len(problems)} of {num_problems} Countdown problems after {attempts} attempts. Try widening the target/value bounds or reducing --num-problems."
        )
    return problems


ZEBRA_ATTRIBUTES: dict[str, list[str]] = {
    "color": ["red", "green", "blue", "yellow", "white", "ivory", "orange", "purple"],
    "nationality": ["English", "Spanish", "Japanese", "Italian", "Norwegian", "German", "Swedish", "Danish"],
    "pet": ["dog", "cat", "bird", "fish", "horse", "snake", "rabbit", "hamster"],
    "drink": ["tea", "coffee", "milk", "water", "juice", "cola", "wine", "cocoa"],
    "hobby": ["chess", "reading", "painting", "gardening", "cooking", "dancing", "hiking", "fishing"],
}


def _zebra_house_of(houses: dict[str, tuple[str, ...]], cat: str, val: str) -> int:
    return houses[cat].index(val)


def _zebra_eval_clue(clue: dict[str, Any], houses: dict[str, tuple[str, ...]]) -> bool:
    kind = clue["type"]
    if kind == "position":
        return _zebra_house_of(houses, clue["cat"], clue["val"]) == clue["pos"]
    h1 = _zebra_house_of(houses, clue["c1"], clue["v1"])
    h2 = _zebra_house_of(houses, clue["c2"], clue["v2"])
    if kind == "same":
        return h1 == h2
    if kind == "not_same":
        return h1 != h2
    if kind == "immediate_left":
        return h1 + 1 == h2
    if kind == "left":
        return h1 < h2
    if kind == "next_to":
        return abs(h1 - h2) == 1
    raise ValueError(f"Unknown zebra clue type: {kind}")


def _zebra_clue_text(clue: dict[str, Any]) -> str:
    if clue["type"] == "position":
        return f"The person whose {clue['cat']} is {clue['val']} lives in house {clue['pos'] + 1}."
    p1 = f"the person whose {clue['c1']} is {clue['v1']}"
    p2 = f"the person whose {clue['c2']} is {clue['v2']}"
    p1_cap = p1[0].upper() + p1[1:]
    if clue["type"] == "same":
        return f"The person whose {clue['c1']} is {clue['v1']} is the same person whose {clue['c2']} is {clue['v2']}."
    if clue["type"] == "not_same":
        return f"The person whose {clue['c1']} is {clue['v1']} is not the same person whose {clue['c2']} is {clue['v2']}."
    if clue["type"] == "immediate_left":
        return f"{p1_cap} lives immediately to the left of {p2}."
    if clue["type"] == "left":
        return f"{p1_cap} lives somewhere to the left of {p2}."
    if clue["type"] == "next_to":
        return f"{p1_cap} lives next to {p2}."
    raise ValueError(f"Unknown zebra clue type: {clue['type']}")


def _zebra_candidate_clues(
    cats: list[str], solution: dict[str, tuple[str, ...]], n_houses: int
) -> list[dict[str, Any]]:
    """All clues that are TRUE for ``solution`` across the supported templates."""
    items = [(cat, val) for cat in cats for val in solution[cat]]
    pos = {(cat, val): solution[cat].index(val) for cat, val in items}
    clues: list[dict[str, Any]] = []
    for (cat, val), p in pos.items():
        clues.append({"type": "position", "cat": cat, "val": val, "pos": p})
    for c1, v1 in items:
        for c2, v2 in items:
            if (c1, v1) == (c2, v2):
                continue
            h1, h2 = (pos[c1, v1], pos[c2, v2])
            if c1 != c2 and h1 == h2:
                clues.append({"type": "same", "c1": c1, "v1": v1, "c2": c2, "v2": v2})
            if c1 != c2 and h1 != h2:
                clues.append({"type": "not_same", "c1": c1, "v1": v1, "c2": c2, "v2": v2})
            if h1 + 1 == h2:
                clues.append({"type": "immediate_left", "c1": c1, "v1": v1, "c2": c2, "v2": v2})
            if h1 < h2:
                clues.append({"type": "left", "c1": c1, "v1": v1, "c2": c2, "v2": v2})
            if abs(h1 - h2) == 1:
                clues.append({"type": "next_to", "c1": c1, "v1": v1, "c2": c2, "v2": v2})
    return clues


def _zebra_build_question(
    rng: random.Random, cats: list[str], solution: dict[str, tuple[str, ...]], n_houses: int
) -> tuple[str, str, str]:
    if rng.random() < 0.5:
        cat = rng.choice(cats)
        val = rng.choice(solution[cat])
        pos = solution[cat].index(val) + 1
        question = (
            f"In which house (1 to {n_houses}, counting from the left) is the person whose {cat} is {val}?"
        )
        return (question, str(pos), "position")
    c1 = rng.choice(cats)
    c2 = rng.choice([c for c in cats if c != c1])
    house = rng.randrange(n_houses)
    v1 = solution[c1][house]
    v2 = solution[c2][house]
    question = f"What {c2} does the person whose {c1} is {v1} have?"
    return (question, v2, "attribute")


def _zebra_render_prompt(
    cats: list[str],
    value_lists: dict[str, list[str]],
    clues: list[dict[str, Any]],
    question: str,
    n_houses: int,
) -> str:
    lines = [
        f"There are {n_houses} houses in a row, numbered 1 to {n_houses} from left to right.",
        f"Each house is occupied by one person, and every person has a unique value for each of the following attributes (no two people share a value in the same attribute): {', '.join(cats)}.",
        "",
        "The possible values are:",
    ]
    for cat in cats:
        lines.append(f"- {cat}: {', '.join(value_lists[cat])}")
    lines.append("")
    lines.append("Clues:")
    for i, clue in enumerate(clues, start=1):
        lines.append(f"{i}. {_zebra_clue_text(clue)}")
    lines.append("")
    lines.append(f"Question: {question}")
    return "\n".join(lines)


def _generate_one_zebra(
    rng: random.Random, n_houses: int, n_attributes: int, max_clues: int
) -> dict[str, Any] | None:
    cats = rng.sample(list(ZEBRA_ATTRIBUTES.keys()), n_attributes)
    value_lists: dict[str, list[str]] = {}
    solution: dict[str, tuple[str, ...]] = {}
    for cat in cats:
        pool = ZEBRA_ATTRIBUTES[cat][:]
        rng.shuffle(pool)
        chosen = pool[:n_houses]
        value_lists[cat] = sorted(chosen)
        arrangement = chosen[:]
        rng.shuffle(arrangement)
        solution[cat] = tuple(arrangement)
    all_perms = {cat: list(itertools.permutations(value_lists[cat])) for cat in cats}
    candidates = [
        {cat: combo[i] for i, cat in enumerate(cats)}
        for combo in itertools.product(*(all_perms[cat] for cat in cats))
    ]
    pool = _zebra_candidate_clues(cats, solution, n_houses)
    rng.shuffle(pool)
    chosen_clues: list[dict[str, Any]] = []
    surviving = candidates
    for clue in pool:
        if len(surviving) <= 1 or len(chosen_clues) >= max_clues:
            break
        filtered = [cand for cand in surviving if _zebra_eval_clue(clue, cand)]
        if len(filtered) < len(surviving):
            chosen_clues.append(clue)
            surviving = filtered
    if len(surviving) != 1:
        return None
    i = 0
    while i < len(chosen_clues):
        trial = chosen_clues[:i] + chosen_clues[i + 1 :]
        survivors = candidates
        for clue in trial:
            survivors = [cand for cand in survivors if _zebra_eval_clue(clue, cand)]
            if len(survivors) == 1:
                break
        if len(survivors) == 1:
            chosen_clues.pop(i)
        else:
            i += 1
    rng.shuffle(chosen_clues)
    question, answer, answer_type = _zebra_build_question(rng, cats, solution, n_houses)
    prompt = _zebra_render_prompt(cats, value_lists, chosen_clues, question, n_houses)
    return {"question": prompt, "answer": answer, "answer_type": answer_type, "num_clues": len(chosen_clues)}


def generate_zebra_puzzles_problems(
    num_problems: int, seed: int, n_houses: int = 4, n_attributes: int = 3, max_clues: int = 20
) -> list[dict[str, Any]]:
    if n_attributes < 2:
        raise ValueError("Zebra puzzles need at least 2 attributes.")
    if n_attributes > len(ZEBRA_ATTRIBUTES):
        raise ValueError(f"Zebra puzzles support at most {len(ZEBRA_ATTRIBUTES)} attributes.")
    if any((n_houses > len(values) for values in ZEBRA_ATTRIBUTES.values())):
        raise ValueError("n_houses exceeds the available values for some attribute.")
    rng = random.Random(seed)
    problems: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(num_problems * 200, 10000)
    while len(problems) < num_problems and attempts < max_attempts:
        attempts += 1
        problem = _generate_one_zebra(rng, n_houses, n_attributes, max_clues)
        if problem is None:
            continue
        if problem["question"] in seen:
            continue
        seen.add(problem["question"])
        problems.append(problem)
    if len(problems) < num_problems:
        raise RuntimeError(
            f"Generated only {len(problems)} of {num_problems} Zebra puzzles after {attempts} attempts. Try reducing --num-problems or adjusting --zebra-n-houses/--zebra-n-attributes."
        )
    return problems


CUBE_COLORS = [
    "red",
    "green",
    "blue",
    "yellow",
    "white",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
    "cyan",
    "magenta",
    "gold",
    "silver",
    "indigo",
    "violet",
]
CUBE_SIDES = ["top", "right", "front", "left", "back", "bottom"]
CUBE_ROTATION_SOURCES = ["right", "front", "left", "back", "bottom"]
CUBE_ROTATION_TEMPLATES = [
    "The cube is rotated so that the side which was before at the {side} is now at the top.",
    "Then the cube is rotated to bring the {side} side to the top.",
    "After that the cube is turned to make the {side} face the top.",
    "Now the cube is rotated to place its {side} side at the top.",
    "Next, the {side} side is rotated to become the top face.",
]
CUBE_ROTATION_MAPS: dict[str, dict[str, str]] = {
    "front": {"top": "front", "front": "bottom", "bottom": "back", "back": "top"},
    "right": {"top": "right", "right": "bottom", "bottom": "left", "left": "top"},
    "back": {"top": "back", "back": "bottom", "bottom": "front", "front": "top"},
    "left": {"top": "left", "left": "bottom", "bottom": "right", "right": "top"},
    "bottom": {"top": "bottom", "bottom": "top", "front": "back", "back": "front"},
}


def rotate_cube_to_top(colors: dict[str, str], from_side: str) -> dict[str, str]:
    """Return the cube face colors after rotating ``from_side`` into the top position."""
    rotation = CUBE_ROTATION_MAPS.get(from_side)
    if rotation is None:
        raise ValueError(f"Cannot rotate the {from_side} side to the top.")
    rotated = dict(colors)
    for destination, source in rotation.items():
        rotated[destination] = colors[source]
    return rotated


def _build_color_cube_question(
    initial_state: dict[str, str], rotations: list[str], target_side: str, rng: random.Random
) -> str:
    story_parts = ["A cube has:"]
    for side in CUBE_SIDES:
        story_parts.append(f"- a {initial_state[side]} {side} side")
    for i, from_side in enumerate(rotations):
        template = CUBE_ROTATION_TEMPLATES[0] if i == 0 else rng.choice(CUBE_ROTATION_TEMPLATES[1:])
        story_parts.append(f"\n{template.format(side=from_side)}")
    story_parts.append(f"\nWhat is now the color of the {target_side} side of the cube?")
    story_parts.append("Provide only the color as your final answer.")
    return "\n".join(story_parts)


def generate_color_cube_rotation_problems(
    num_problems: int, seed: int, min_rotations: int = 1, max_rotations: int = 3
) -> list[dict[str, Any]]:
    if min_rotations < 1:
        raise ValueError("min_rotations must be positive.")
    if max_rotations < min_rotations:
        raise ValueError("max_rotations must be >= min_rotations.")
    rng = random.Random(seed)
    problems: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(num_problems * 200, 10000)
    while len(problems) < num_problems and attempts < max_attempts:
        attempts += 1
        colors = list(CUBE_COLORS)
        rng.shuffle(colors)
        initial_state = {side: color for side, color in zip(CUBE_SIDES, colors)}
        num_rotations = rng.randint(min_rotations, max_rotations)
        rotations = [rng.choice(CUBE_ROTATION_SOURCES) for _ in range(num_rotations)]
        state = dict(initial_state)
        for from_side in rotations:
            state = rotate_cube_to_top(state, from_side)
        target_side = rng.choice(CUBE_SIDES)
        question = _build_color_cube_question(initial_state, rotations, target_side, rng)
        if question in seen:
            continue
        seen.add(question)
        problems.append(
            {
                "question": question,
                "answer": state[target_side],
                "initial_state": initial_state,
                "rotations": rotations,
                "target_side": target_side,
                "num_rotations": num_rotations,
            }
        )
    if len(problems) < num_problems:
        raise RuntimeError(
            f"Generated only {len(problems)} of {num_problems} Color Cube Rotation problems after {attempts} attempts. Try reducing --num-problems or widening --cube-min-rotations/--cube-max-rotations."
        )
    return problems


def make_reasoning_row(dataset_name: str, idx: int, problem: dict[str, Any], split: str) -> dict[str, Any]:
    if dataset_name == "knights_knaves":
        prompt = f"{problem['quiz']}\n\n{_final_instruction()}"
        answer = problem["solution_text"]
        description = problem["quiz"]
    elif dataset_name == "binary_matrix":
        matrix_str = matrix_to_str(problem["matrix"])
        prompt = f"{BINARY_MATRIX_QUESTION_TEMPLATE.format(matrix=matrix_str)}\n\n{_final_instruction()}"
        answer = matrix_to_str(problem["solution"])
        description = BINARY_MATRIX_QUESTION_TEMPLATE.format(matrix=matrix_str)
    elif dataset_name == "countdown":
        numbers_str = ", ".join((str(x) for x in problem["numbers"]))
        question = COUNTDOWN_PROMPT_TEMPLATES[idx % len(COUNTDOWN_PROMPT_TEMPLATES)].format(
            numbers=numbers_str, target=problem["target"]
        )
        prompt = f"{question}\n\n{_final_instruction()}"
        answer = json.dumps(
            {"numbers": problem["numbers"], "target": problem["target"], "expression": problem["expression"]},
            ensure_ascii=True,
            sort_keys=True,
        )
        description = question
    elif dataset_name == "zebra_puzzles":
        prompt = f"{problem['question']}\n\n{_final_instruction()}"
        answer = json.dumps(
            {"answer": problem["answer"], "answer_type": problem["answer_type"]},
            ensure_ascii=True,
            sort_keys=True,
        )
        description = problem["question"]
    elif dataset_name == "quantum_lock":
        prompt = f"{problem['question']}\n\n{_final_instruction()}"
        answer = json.dumps(
            {
                "answer": problem["answer"],
                "solution_path": problem["solution_path"],
                "target_value": problem["target_value"],
                "buttons": problem["buttons"],
                "initial_state": problem["initial_state"],
                "initial_value": problem["initial_value"],
                "difficulty": problem["difficulty"],
                "source_index": problem.get("source_index"),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        description = problem["question"]
    elif dataset_name in COLOR_CUBE_ROTATION_DATASETS:
        prompt = f"{problem['question']}\n\n{_final_instruction()}"
        answer = problem["answer"]
        description = problem["question"]
    elif dataset_name == "string_manipulation":
        prompt = f"{problem['question']}\n\n{_final_instruction()}"
        answer = problem["answer"]
        description = problem["question"]
    else:
        raise ValueError(f"Unsupported reasoning-gym dataset: {dataset_name}")
    return {
        "idx": idx,
        "kind": "reasoning_gym",
        "dataset": dataset_name,
        "answer": answer,
        "elo": 1500,
        "prompt": prompt,
        "description": f"{description} The solution will be evaluated in a reasoning_gym environment.",
        "tests": "-",
        "embedding": [],
        "source_dataset": "reasoning-gym",
        "source_split": split,
    }


def split_train_test(
    problems: list[dict[str, Any]], test_size: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(problems) == 0:
        raise ValueError("Cannot split an empty problem list.")
    if len(problems) == 1:
        return (problems, problems)
    if test_size <= 0:
        return (problems, problems[-min(200, len(problems)) :])
    actual_test_size = min(test_size, len(problems) - 1)
    return (problems[:-actual_test_size], problems[-actual_test_size:])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_dataset_dir(
    output_dir: Path,
    dataset_name: str,
    problems: list[dict[str, Any]],
    test_size: int,
    config: dict[str, Any],
    train_problems: list[dict[str, Any]] | None = None,
    test_problems: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if train_problems is None or test_problems is None:
        train_problems, test_problems = split_train_test(problems, test_size=test_size)
    train_rows = [
        make_reasoning_row(dataset_name, idx, problem, "train") for idx, problem in enumerate(train_problems)
    ]
    test_rows = [
        make_reasoning_row(dataset_name, idx, problem, "test") for idx, problem in enumerate(test_problems)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.json", train_rows)
    write_jsonl(output_dir / "test.json", test_rows)
    sample_config = {
        "dataset": dataset_name,
        "source_dataset": "reasoning-gym",
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        **config,
    }
    (output_dir / "sample_config.json").write_text(json.dumps(sample_config, indent=2, sort_keys=True) + "\n")
    return sample_config
