import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocess import clean_text


def test_lowercase():
    assert clean_text("Hello World", remove_stopwords=False) == "hello world"

def test_removes_urls():
    assert "http" not in clean_text("Check https://example.com out", remove_stopwords=False)

def test_removes_punctuation():
    assert "!" not in clean_text("Breaking!!!", remove_stopwords=False)

def test_removes_digits():
    assert "100" not in clean_text("There are 100 cases", remove_stopwords=False)

def test_empty_string():
    assert clean_text("") == ""

def test_none_input():
    assert clean_text(None) == ""

def test_returns_string():
    assert isinstance(clean_text("any text"), str)

def test_removes_hashtags():
    assert "#fakenews" not in clean_text("this is #fakenews", remove_stopwords=False)


if __name__ == "__main__":
    tests = [
        test_lowercase, test_removes_urls, test_removes_punctuation,
        test_removes_digits, test_empty_string, test_none_input,
        test_returns_string, test_removes_hashtags,
    ]
    passed = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
