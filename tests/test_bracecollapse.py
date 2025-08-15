from bracecollapse import bracecollapse, PrefixTreeNode


def test_prefix_tree():
    root = PrefixTreeNode()
    root.insert("foo.bar")
    root.insert("foo.baz")
    root.insert("qux")
    root.insert("001")

    assert sorted(root.children.keys()) == ["001", "foo", "qux"]

    assert root.children["foo"].type == 0  # Type 0 for alpha
    assert not root.children["foo"].is_terminal

    assert root.children["001"].type == 1  # Type 1 for numeric
    assert root.children["001"].is_terminal

    assert not root.children["qux"].children
    assert root.children["qux"].is_terminal

    assert "." in root.children["foo"].children
    assert root.children["foo"].children["."].type == 2
    assert root.children["foo"].children["."].children["bar"].is_terminal
    assert root.children["foo"].children["."].children["baz"].is_terminal


def test_alphasingle():
    assert bracecollapse(["a", "b", "c"]) == ["{a,b,c}"]
    assert bracecollapse(["foo.a", "foo.b", "foo.c"]) == ["foo.{a,b,c}"]
    assert bracecollapse(["a.b.c", "a.b.d", "a.e.f"]) == ["a.b.{c,d}", "a.e.f"]


def test_alphamultiple():
    assert bracecollapse(["a.b.c", "a.b.d", "a.e.c", "a.e.d"]) == ["a.{b,e}.{c,d}"]


def test_numeric_rangeset():
    # Consecutive range
    assert bracecollapse(["001", "002", "003"], numeric="rangeset") == ["{001..003}"]
    assert bracecollapse(["foo.001", "foo.002", "foo.003"], numeric="rangeset") == [
        "foo.{001..003}"
    ]
    assert bracecollapse(["a.001.b", "a.002.b", "a.003.b"], numeric="rangeset") == [
        "a.{001..003}.b"
    ]

    # Non-consecutive range
    assert bracecollapse(["001", "002", "003", "005"], numeric="rangeset") == [
        "{{001..003},005}"
    ]
    assert bracecollapse(
        ["foo.001", "foo.002", "foo.003", "foo.005"], numeric="rangeset"
    ) == ["foo.{{001..003},005}"]
    assert bracecollapse(
        ["a.001.b", "a.002.b", "a.003.b", "a.005.b"], numeric="rangeset"
    ) == ["a.{{001..003},005}.b"]


def test_numeric_range():
    assert bracecollapse(["001", "002", "003", "005"], numeric="range") == [
        "{001..005}"
    ]
    assert bracecollapse(
        ["foo.001", "foo.002", "foo.003", "foo.005"], numeric="range"
    ) == ["foo.{001..005}"]
    assert bracecollapse(
        ["a.001.b", "a.002.b", "a.003.b", "a.005.b"], numeric="range"
    ) == ["a.{001..005}.b"]


def test_glob():
    assert bracecollapse(["foo", "bar", "baz"], alpha="glob") == ["*"]
