import re

from airfoil_ml.acquisition import safe_solver_name


def test_sanitizer_removes_spaces_and_punctuation() -> None:
    name = safe_solver_name("74130 WP2")
    assert " " not in name
    assert re.fullmatch(r"[A-Za-z0-9_]+", name)


def test_sanitizer_is_deterministic_and_collision_resistant() -> None:
    assert safe_solver_name("AG03 flat aft bottom") == safe_solver_name("AG03 flat aft bottom")
    assert safe_solver_name("a b") != safe_solver_name("ab")


def test_sanitizer_stays_short_and_nonempty() -> None:
    for name in ("", "   ", "!!!!", "e387", "x" * 200):
        safe = safe_solver_name(name)
        assert 0 < len(safe) <= 33  # 24-char slug + underscore + 8-char hash
        assert re.fullmatch(r"[A-Za-z0-9_]+", safe)
