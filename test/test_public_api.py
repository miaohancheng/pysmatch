from pysmatch import Matcher, __version__, load_demo
from pysmatch.matcher import Matcher as LowercaseMatcher


def test_public_imports_and_demo_loader():
    assert Matcher is LowercaseMatcher
    assert isinstance(__version__, str)
    assert load_demo().shape == (500, 9)
