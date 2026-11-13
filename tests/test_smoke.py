import cardd


def test_class_names():
    assert len(cardd.CLASS_NAMES) == 5
    assert "tire_flat" not in cardd.CLASS_NAMES
