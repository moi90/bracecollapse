from bracecollapse import TYPE_OTHER, PrefixTreeNode, bracecollapse


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
    assert root.children["foo"].children["."].type == TYPE_OTHER
    assert root.children["foo"].children["."].children["bar"].is_terminal
    assert root.children["foo"].children["."].children["baz"].is_terminal


def test_alphasingle():
    assert bracecollapse(["a", "b", "c"]) == {"{a,b,c}"}
    assert bracecollapse(["foo.a", "foo.b", "foo.c"]) == {"foo.{a,b,c}"}
    assert bracecollapse(["a.b.c", "a.b.d", "a.e.f"]) == {"a.b.{c,d}", "a.e.f"}


def test_alphamultiple():
    assert bracecollapse(["a.b.c", "a.b.d", "a.e.c", "a.e.d"]) == {"a.{b,e}.{c,d}"}


def test_numeric_rangeset():
    # Consecutive range
    assert bracecollapse(["001", "002", "003"], numeric="rangeset") == {"{001..003}"}
    assert bracecollapse(["foo.001", "foo.002", "foo.003"], numeric="rangeset") == {
        "foo.{001..003}"
    }
    assert bracecollapse(["a.001.b", "a.002.b", "a.003.b"], numeric="rangeset") == {
        "a.{001..003}.b"
    }

    # Non-consecutive range
    assert bracecollapse(["001", "002", "003", "005"], numeric="rangeset") == {
        "{{001..003},005}"
    }
    assert bracecollapse(
        ["foo.001", "foo.002", "foo.003", "foo.005"], numeric="rangeset"
    ) == {"foo.{{001..003},005}"}
    assert bracecollapse(
        ["a.001.b", "a.002.b", "a.003.b", "a.005.b"], numeric="rangeset"
    ) == {"a.{{001..003},005}.b"}


def test_numeric_range():
    assert bracecollapse(["001", "002", "003", "005"], numeric="range") == {
        "{001..005}"
    }
    assert bracecollapse(
        ["foo.001", "foo.002", "foo.003", "foo.005"], numeric="range"
    ) == {"foo.{001..005}"}
    assert bracecollapse(
        ["a.001.b", "a.002.b", "a.003.b", "a.005.b"], numeric="range"
    ) == {"a.{001..005}.b"}


def test_mixed_padded_numeric():
    assert bracecollapse(["001", "002", "003", "01", "02", "03"], numeric="range") == {
        "{001..003}",
        "{01..03}",
    }

    assert bracecollapse(["09", "10", "11", "12"], numeric="range") == {"{09..12}"}


def test_glob():
    assert bracecollapse(["foo", "bar", "baz"], alpha="glob") == {"*"}


def test_collapse_format():
    assert bracecollapse(["001", "002", "003"], numeric="format") == {"{:03d}"}
    assert bracecollapse(["1", "2", "3"], numeric="format") == {"{:d}"}
    assert bracecollapse(["foo.001", "foo.002", "foo.003"], numeric="format") == {
        "foo.{:03d}"
    }
    assert bracecollapse(["a.001.b", "a.002.b", "a.003.b"], numeric="format") == {
        "a.{:03d}.b"
    }


def test_collapse_type():
    assert bracecollapse(["001", "002", "003"], numeric="type") == {"<int03>"}
    assert bracecollapse(["1", "2", "3"], numeric="type") == {"<int>"}
    assert bracecollapse(["foo.001", "foo.002", "foo.003"], numeric="type") == {
        "foo.<int03>"
    }
    assert bracecollapse(["a.001.b", "a.002.b", "a.003.b"], numeric="type") == {
        "a.<int03>.b"
    }
    assert bracecollapse(["foo", "bar", "baz"], alpha="type") == {"<str>"}
    assert bracecollapse(["foo.a", "foo.b", "foo.c"], alpha="type") == {"foo.<str>"}
