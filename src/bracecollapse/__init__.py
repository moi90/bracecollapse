import re
from abc import ABCMeta, abstractmethod
from collections.abc import Collection, Hashable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, Sequence

TOKEN_RE = re.compile(
    r"([A-Za-z\u00C0-\u00FF]+)|(0+[0-9]+)|([0-9]+)|([^A-Za-z\u00C0-\u00FF0-9])"
)
TYPE_ALPHA = 0
TYPE_PADDED_NUMERIC = 1
TYPE_NUMERIC = 2
TYPE_OTHER = 3


class Expression(metaclass=ABCMeta):
    __slots__ = ()

    def __init__(self, type: int) -> None:
        self.type = type

    @abstractmethod
    def __hash__(self) -> int:
        return 0

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        return NotImplemented

    @abstractmethod
    def __str__(self) -> str:
        return ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type}, expression={str(self)})"


class SetExpression(Expression):
    """
    Expression that expands to a comma-separated set.

    Examples:
        SetExpression(TYPE_ALPHA, ['a', 'b', 'c']) => "{a,b,c}"
        SetExpression(TYPE_ALPHA, ['x']) => "x" (single element)
    """

    def __init__(self, type: int, tokens: Sequence[str]):
        super().__init__(type)
        self.tokens = tuple(sorted(tokens))

    def __hash__(self) -> int:
        return hash((self.type, self.tokens))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SetExpression):
            return NotImplemented
        return self.type == other.type and self.tokens == other.tokens

    def __str__(self) -> str:
        if len(self.tokens) == 1:
            return next(iter(self.tokens))

        return "{" + ",".join(self.tokens) + "}"


ExpressionOrLiteral = str | Expression
Pattern = tuple[ExpressionOrLiteral, ...]


class RangeSetExpression(Expression):
    """
    Expression that expands to a set with consecutive numeric ranges collapsed.

    Consecutive integers are represented as ranges, separated by commas.

    Examples:
        RangeSetExpression(TYPE_NUMERIC, ['1', '2', '3', '5', '6']) => "{1..3,5..6}"
        RangeSetExpression(TYPE_NUMERIC, ['1']) => "1" (single element)
    """

    def __init__(self, type: int, tokens: Sequence[str]):
        super().__init__(type)
        self.ranges = tuple(self._parse_ranges(tokens))

    @staticmethod
    def _parse_ranges(tokens: Sequence[str]) -> Iterable[str | tuple[str, str]]:
        token_values = [(int(token), token) for token in tokens]
        token_values.sort(key=lambda x: x[0])  # Sort by integer value

        start = None
        prev = None
        for value, token in token_values:
            if start is None:
                start = prev = (value, token)
                continue

            assert prev is not None

            if value == prev[0] + 1:
                # If the current value is consecutive, continue the range
                prev = (value, token)
                continue

            # If the current value is not consecutive, finalize the previous range
            if start[0] == prev[0]:
                yield start[1]  # Single value
            else:
                yield (start[1], prev[1])  # Range

            start = prev = (value, token)

        # Finalize the last range
        if start is not None:
            assert prev is not None
            if start[0] == prev[0]:
                yield start[1]  # Single value
            else:
                yield (start[1], prev[1])  # Range

    def __hash__(self) -> int:
        return hash((self.type, self.ranges))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RangeSetExpression):
            return NotImplemented
        return self.type == other.type and self.ranges == other.ranges

    def __str__(self) -> str:
        parts = [
            f"{{{part[0]}..{part[1]}}}" if isinstance(part, tuple) else part
            for part in self.ranges
        ]

        if len(parts) == 1:
            return parts[0]

        return "{" + ",".join(parts) + "}"


class RangeExpression(Expression):
    """
    Expression that expands to a range from minimum to maximum value.

    Examples:
        RangeExpression(TYPE_NUMERIC, ['1', '5', '9']) => "{1..9}"
        RangeExpression(TYPE_NUMERIC, ['5']) => "5" (single element)
    """

    def __init__(self, type: int, tokens: Sequence[str]):
        super().__init__(type)
        self.range = self._parse_range(tokens)

    @staticmethod
    def _parse_range(tokens: Sequence[str]) -> str | tuple[str, str]:
        if not tokens:
            raise ValueError("No tokens provided for range expression")

        token_values = [(int(token), token) for token in tokens]

        start = min(token_values, key=lambda x: x[0])
        end = max(token_values, key=lambda x: x[0])

        if start[0] == end[0]:
            return start[1]

        return (start[1], end[1])

    def __hash__(self) -> int:
        return hash((self.type, self.range))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RangeExpression):
            return NotImplemented
        return self.type == other.type and self.range == other.range

    def __str__(self) -> str:
        if isinstance(self.range, str):
            return self.range

        return f"{{{self.range[0]}..{self.range[1]}}}"


class GlobExpression(Expression):
    """
    Expression that matches any token of a given type.

    Examples:
        GlobExpression(TYPE_ALPHA) => "*"
    """

    def __init__(self, type: int):
        super().__init__(type)

    def __hash__(self) -> int:
        return hash((self.type, "*"))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GlobExpression):
            return NotImplemented
        return self.type == other.type

    def __str__(self) -> str:
        return "*"


class TypeExpression(Expression):
    """
    Expression using angle bracket notation to specify types precisely.

    In contrast to FormatStrExpression, this is intended to co-exist
    with brace expansion syntax. Uses angle brackets
    to denote type with optional padding width.

    Examples:
        TypeExpression(TYPE_ALPHA, None) => "<str>"
        TypeExpression(TYPE_NUMERIC, None) => "<int>"
        TypeExpression(TYPE_PADDED_NUMERIC, 3) => "<int03>"
    """

    def __init__(self, type: int, width: int | None = None):
        super().__init__(type)
        self.width = width

    def __hash__(self) -> int:
        return hash((self.type, self.width, "{}"))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FormatStrExpression):
            return NotImplemented
        return self.type == other.type and self.width == other.width

    def __str__(self) -> str:
        if self.type == TYPE_ALPHA:
            return "<str>"

        if self.type == TYPE_PADDED_NUMERIC:
            assert self.width is not None
            return f"<int0{self.width}>"

        if self.type == TYPE_NUMERIC:
            return "<int>"

        raise ValueError(f"Unknown type for TypeExpression: {self.type}")


class FormatStrExpression(TypeExpression):
    """Expression using Python format string syntax to specify types precisely.

    Note: This format can collide with brace expansion syntax.

    Examples:
        FormatStrExpression(TYPE_ALPHA, None) => "{:s}"
        FormatStrExpression(TYPE_NUMERIC, None) => "{:d}"
        FormatStrExpression(TYPE_PADDED_NUMERIC, 3) => "{:03d}"
    """

    def __str__(self) -> str:
        if self.type == TYPE_ALPHA:
            return "{:s}"

        if self.type == TYPE_PADDED_NUMERIC:
            assert self.width is not None
            return f"{{:0{self.width}d}}"

        if self.type == TYPE_NUMERIC:
            return "{:d}"

        raise ValueError(f"Unknown type for FormatStrExpression: {self.type}")


def make_expression(
    type, width: int | None, tokens: Sequence[str], alpha: str, numeric: str
) -> Expression:
    if type == TYPE_ALPHA:
        if alpha == "set":
            return SetExpression(type, tokens)
        elif alpha == "glob":
            return GlobExpression(type)
        elif alpha == "format":
            return FormatStrExpression(type, width)
        elif alpha == "type":
            return TypeExpression(type, width)
        raise ValueError(f"Unknown alpha type: {alpha}")
    elif type in (TYPE_PADDED_NUMERIC, TYPE_NUMERIC):
        if numeric == "set":
            return SetExpression(type, tokens)
        elif numeric == "rangeset":
            return RangeSetExpression(type, tokens)
        elif numeric == "range":
            return RangeExpression(type, tokens)
        elif numeric == "glob":
            return GlobExpression(type)
        elif numeric == "format":
            return FormatStrExpression(type, width)
        elif numeric == "type":
            return TypeExpression(type, width)
        raise ValueError(f"Unknown numeric type: {numeric}")

    raise ValueError(f"Unknown type: {type}")


class PrefixTreeNode:
    def __init__(self, type=-1):
        self.children: dict[str, "PrefixTreeNode"] = {}
        self.is_terminal = False
        self.type = type  # Type of the node, e.g., 0, 1, or 2 for alpha, numeric, or other tokens

    def insert(self, string: str):
        """
        Insert a string into the prefix tree.

        The string is tokenized into alpha, numeric, and other tokens
        which are treated as atoms in the tree.

        Args:
            string: A string to insert.
        """

        # Tokenize the string.
        # (Each match is a tuple of (alpha, numeric, other), only one of which is non-empty.)
        matches: list[tuple[str, str, str]] = TOKEN_RE.findall(string)

        if not matches:
            return

        node = self
        for match in matches:
            # Find the first non-empty group in the match
            result = next(((i, m) for i, m in enumerate(match) if m), None)
            if result is None:
                raise RuntimeError(f"Invalid match in string '{string}': {match}")

            type, token = result
            node = node.children.setdefault(token, PrefixTreeNode(type))
        node.is_terminal = True

    def to_patterns(
        self,
        alpha: Literal["set", "glob", "format", "type"] = "set",
        numeric: Literal[
            "set", "rangeset", "range", "glob", "format", "type"
        ] = "rangeset",
    ) -> Iterable[Pattern]:
        if self.is_terminal:
            yield ()

        if not self.children:
            return

        # Group children by type and patterns
        # (type, width, patterns) => [token, ...]
        children_by_type_and_patterns: dict[
            tuple[int, int | None, frozenset[Pattern]], list[str]
        ] = {}
        for token, child in self.children.items():
            child_patterns = frozenset(child.to_patterns(alpha, numeric))
            width = (
                len(token)
                if child.type in (TYPE_PADDED_NUMERIC, TYPE_NUMERIC)
                else None
            )
            children_by_type_and_patterns.setdefault(
                (self.children[token].type, width, child_patterns), []
            ).append(token)

        # Padded numeric tokens "steal" from unpadded numeric tokens if they have the same width
        for type, width, child_patterns in list(children_by_type_and_patterns.keys()):
            if type != TYPE_PADDED_NUMERIC:
                continue
            unpadded_key = (TYPE_NUMERIC, width, child_patterns)
            if unpadded_key in children_by_type_and_patterns:
                children_by_type_and_patterns[unpadded_key].extend(
                    children_by_type_and_patterns[(type, width, child_patterns)]
                )
                del children_by_type_and_patterns[(type, width, child_patterns)]

        for (
            type,
            width,
            child_patterns,
        ), tokens in children_by_type_and_patterns.items():
            if len(tokens) == 1:
                # If there's only one token, yield a literal pattern
                for child_pattern in child_patterns:
                    yield (tokens[0], *child_pattern)
                continue

            if type == TYPE_OTHER:
                # If the type is 'other', yield literal patterns
                for token in tokens:
                    for child_pattern in child_patterns:
                        yield (token, *child_pattern)
                continue

            # If there are multiple tokens (that are not type==TYPE_OTHER), yield an expression pattern
            for child_pattern in child_patterns:
                yield (
                    make_expression(type, width, tokens, alpha, numeric),
                    *child_pattern,
                )


def bracecollapse(
    strings: Collection[str],
    alpha: Literal["set", "glob", "format", "type"] = "set",
    numeric: Literal["set", "rangeset", "range", "glob", "format", "type"] = "rangeset",
) -> set[str]:
    """
    Collapse a collection of raw strings into a list of pattern strings with brace expansion.

    Args:
        strings: A collection of strings to collapse.

    Returns:
        A list of strings with brace expressions.
    """

    # Build a prefix tree for tokens
    root = PrefixTreeNode()
    for s in strings:
        root.insert(s)

    # Convert the prefix tree to patterns
    patterns = root.to_patterns(alpha=alpha, numeric=numeric)

    return {"".join(str(expr) for expr in pattern) for pattern in patterns}
