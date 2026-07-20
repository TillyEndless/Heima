from src.g1.synthetic_data import generate_synthetic_split, validate_record


def test_synthetic_split_is_deterministic_and_correct():
    train_a, val_a = generate_synthetic_split(train_size=64, validation_size=32, seed=42)
    train_b, val_b = generate_synthetic_split(train_size=64, validation_size=32, seed=42)

    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) == 64
    assert len(val_a) == 32

    combos = set()
    for record in train_a + val_a:
        assert validate_record(record)
        combo = tuple(record["metadata"][k] for k in ("a", "b", "c"))
        assert combo not in combos
        combos.add(combo)
        assert len(record["steps"]) == 2
        assert record["cot"] == " ".join(record["steps"])

