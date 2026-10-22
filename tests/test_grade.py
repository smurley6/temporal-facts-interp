from tfi.model import grade, normalize


def test_normalize_strips_accents():
    assert normalize("Sébastien Lecornu") == "sebastien lecornu"


def test_grade_correct_alias():
    assert grade(" Tim Cook.\nThe chief", ["Timothy D. Cook", "Tim Cook"], []) == "correct"


def test_grade_outdated_beats_stale():
    assert grade(" Bob Iger.", ["Josh D'Amaro"], ["Bob Iger"], ["Michael Eisner"]) == "outdated"


def test_grade_stale():
    assert grade(" Rupert Stadler.", ["Gernot Döllner"], [], ["Rupert Stadler"]) == "stale"


def test_grade_only_first_clause_counts():
    assert grade(" a man. Tim Cook", ["Tim Cook"], []) == "other"
