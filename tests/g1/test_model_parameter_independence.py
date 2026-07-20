from src.g1.latent_reasoner import assert_parameter_independence


class Tiny:
    def __init__(self, params):
        self._params = params

    def parameters(self):
        return iter(self._params)


def test_parameter_independence_rejects_shared_objects():
    shared = object()
    assert_parameter_independence(Tiny([object()]), Tiny([object()]))

    try:
        assert_parameter_independence(Tiny([shared]), Tiny([shared]))
    except ValueError as exc:
        assert "share" in str(exc)
    else:
        raise AssertionError("shared parameter object should fail")

