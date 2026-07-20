from src.g1.token_category_metrics import extract_expressions, extract_numbers, template_without_numbers


def test_token_category_helpers_find_numbers_and_expressions():
    text = "First add 6 and 29: 6 + 29 = 35."
    assert extract_numbers(text) == ["6", "29", "6", "29", "35"]
    assert extract_expressions(text) == ["6+29=35"]
    assert template_without_numbers(text) == "First add <NUM> and <NUM>: <NUM> + <NUM> = <NUM>."

