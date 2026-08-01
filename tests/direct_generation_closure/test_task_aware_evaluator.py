import importlib.util
from pathlib import Path
SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'experiments' / 'direct_generation_closure.py'
spec = importlib.util.spec_from_file_location('closure', SCRIPT)
closure = importlib.util.module_from_spec(spec); spec.loader.exec_module(closure)

def test_numeric_normalization():
    assert closure.parse_num('Answer: 1,234.0') == '1234'
    assert closure.task_correct('1234', 'The answer is 1,234.0', 'numeric')

def test_code_exact_only():
    gold='```python\nreturn x\n```'
    assert closure.task_correct(gold, gold, 'code')
    assert not closure.task_correct(gold, 'return x', 'code')

def test_natural_language_normalized():
    assert closure.task_correct('hello   world', ' hello world ', 'natural_language')

def test_type_inference():
    assert closure.infer_answer_type({'answer':'```python\ndef f(): pass\n```','question':''}) == 'code'
    assert closure.infer_answer_type({'answer':'42','question':''}) == 'numeric'
