# %%
from collections import defaultdict
from collections.abc import Iterable, Mapping
import re

# %%

SPLIT_PATTERN = re.compile(r"([A-Za-z]+|[0-9]+|[_/.])")


def tokenize_str(s):
    """
    Tokenizes a string into a list of alphanumeric and non-alphanumeric tokens.
    """
    return [t for t in SPLIT_PATTERN.split(s) if t]


tokenize_str("IHLS2014_2019_IHLS012015/tridens_jan_2015.1_hol_6_tot_1_dat1.txt")


# %%
with open("files.txt") as f:
    names = [ls for line in f if (ls := line.strip())]

# %%
tokenized_names = [tuple(tokenize_str(name)) for name in names]
print("\n".join(" ".join(tname) for tname in tokenized_names))

# %%

class BraceExpression:
    def __str__(self):
        raise NotImplementedError()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)})"
    
    def __contains__(self, item: str) -> bool:
        """
        Checks if the given item is contained in the brace expression.
        """
        raise NotImplementedError()
    
class BraceExpressionNumeric(BraceExpression):
    def __init__(self, min_value: int, max_value: int, length: int | None = None):
        self.min_value = min_value
        self.max_value = max_value
        self.length = length

    def __str__(self):
        if self.length is not None:
            return "{" + f"{self.min_value:0{self.length}}..{self.max_value:0{self.length}}" + "}"
        return "{" + f"{self.min_value}..{self.max_value}" + "}"
    
    def __contains__(self, item: str) -> bool:
        """
        Checks if the given item is contained in the numeric brace expression.
        """

        item = item.strip()

        if not item.isnumeric():
            return False
        
        value = int(item)
        if value < self.min_value or value > self.max_value:
            return False
        
        return True
    
class BraceExpressionSet(BraceExpression):
    def __init__(self, items: Iterable[str]):
        self.items = frozenset(items)

    def __str__(self):
        return "{" + ",".join(sorted(self.items)) + "}"
    
    def __contains__(self, item: str) -> bool:
        return item in self.items
    
    def __eq__(self, other):
        if isinstance(other, BraceExpressionSet):
            return self.items == other.items
        return False
    
    def __hash__(self) -> int:
        return hash(self.items)

PartType = str | BraceExpression

class TrieNode:
    def __init__(self, children=None, is_root=False):
        if children is None:
            children = {}
        self.children = defaultdict[PartType, TrieNode](TrieNode, children)
        self.is_root = is_root

    def insert(self, tokens: Iterable[str], subtree_token=None):
        node = self
        for token in tokens:
            node = node.children[token]
            if subtree_token is not None and token == subtree_token:
                node.is_root = True

    def insert_many(self, tokens_list: Iterable[Iterable[str]], subtree_token=None):
        for tokens in tokens_list:
            self.insert(tokens, subtree_token=subtree_token)

        return self
    
    def find(self, tokens: Iterable[str]) -> "TrieNode":
        """
        Finds the node corresponding to the given tokens.
        """
        node = self
        for token in tokens:
            if token not in node.children:
                raise KeyError(f"Token '{token}' not found among {sorted(node.children.keys())}.")
            node = node.children[token]
        return node

    def format(self, max_depth=None, _depth=0, _prefix="") -> str:
        if not self.children or (max_depth is not None and _depth >= max_depth):
            if _prefix:
                return _depth * "  " + _prefix
            return ""

        if len(self.children) == 1:
            # If there's only one child, we can merge it with the parent
            (token, child) = next(iter(self.children.items()))
            return child.format(max_depth, _depth, _prefix + str(token))

        result: list[str] = []
        for token, child in self.children.items():
            if isinstance(token, str) and token.endswith("/"):
                result.append(_depth * "  " + _prefix + str(token))
                result.append(child.format(max_depth, _depth + 1))
            else:
                result.append(child.format(max_depth, _depth, _prefix + str(token)))

        return "\n".join(line for line in result if line)

    def print(self, max_depth=None):
        """
        Prints the trie in a human-readable format.
        """
        print(self.format(max_depth))

    def __repr__(self):
        return repr(self.children)

    def simplify(self) -> "TrieNode":
        """
        Simplifies the trie by constructing a common brace expansion for all children.
        """

        # TODO: A path separator should be treated as a special case so that
        # parts between separators can be efficiently merged.
        # However, 

        # First, simplify all children recursively
        children = {token: child.simplify() for token, child in self.children.items()}

        # We don't have to simplify nodes with zero or one children
        if len(children) < 2:
            return TrieNode(children)

        print("Simplifying node with children:", children.keys())

        # TODO: We should only merge groups into brace expressions if they are equivalent,
        # i.e., same placement of literal tokens and expressions (expressions don't have to be equivalent themselves).
        
        # Group children by their sequence of tokens and expressions
        multiple_expansions = []
        single_expansions = defaultdict[tuple, dict[PartType, TrieNode]](dict)

        for token, child in children.items():
            try:
                expansion, = child.enumerate_raw()
            except ValueError:
                multiple_expansions.append((token, child))
                continue

            # Replace expressions in the expansion with their type as a placeholder
            expansion = tuple(type(part) if isinstance(part, BraceExpression) else part for part in expansion)
            single_expansions[expansion][token] = child

        result = TrieNode()
        for expansion, children in single_expansions.items():
            # We have to classify the children into different groups:
            # Unpadded numeric tokens, padded numeric tokens, alphabetic tokens, and others

            token_groups = defaultdict[str, list[tuple[PartType, TrieNode]]](list)

            for token, child in children.items():
                if isinstance(token, str):
                    if token.isnumeric():
                        if token[0] == "0":
                            token_groups[f"numeric{len(token)}"].append((token, child))
                        else:
                            token_groups["numeric"].append((token, child))
                        continue
                    elif token.isalpha():
                        token_groups["alpha"].append((token, child))
                        continue

                token_groups["other"].append((token, child))

            # If we have unpadded numeric tokens where we also have padded numeric tokens of the same length,
            # we can add the unpadded tokens to the padded group
            new_numeric = []
            for token, child in token_groups.pop("numeric", []):
                target_key = f"numeric{len(token)}"
                if target_key in token_groups:
                    # We have padded numeric tokens of the same length, so we can add the unpadded token to this group
                    print("  Moving {token} to {target_key}")
                    token_groups[target_key].append((token, child))
                else:
                    # We don't have padded numeric tokens of the same length, so we keep it in the unpadded group
                    new_numeric.append((token, child))
            if new_numeric:
                token_groups["numeric"] = new_numeric

            print("  Token groups after classification: ", token_groups.keys())

            # Now, we can simplify each group separately
            for group, items in token_groups.items():
                if len(items) < 2:
                    brace_expression = None
                else:
                    if group.startswith("numeric"):
                        # Numeric tokens
                        # Convert tokens to integers and find the min and max
                        tokens = [int(item[0]) for item in items]
                        min_token = min(tokens)
                        max_token = max(tokens)
                        if group == "numeric":
                            # Unpadded numeric tokens
                            brace_expression = BraceExpressionNumeric(min_token, max_token)
                            print(f"  Unpadded numeric brace expression: {brace_expression}")
                        else:
                            # Padded numeric tokens
                            length = len(
                                items[0][0]
                            )  # All tokens in this group have the same length
                            brace_expression = BraceExpressionNumeric(min_token, max_token, length)
                            print(f"  Padded numeric brace expression: {brace_expression}")
                    elif group == "alpha":
                        # Alphabetic tokens
                        # Sort tokens and create a comma-separated list
                        brace_expression = BraceExpressionSet(
                            [item[0] for item in items]
                        )
                        print(f"  Alphabetic brace expression: {brace_expression}")
                    else:
                        # Other tokens (e.g., special characters)
                        brace_expression = None

                if brace_expression is not None:
                    print(f"  Brace expression: {brace_expression}")
                    result.children[brace_expression].merge_children({
                                    gc_token: gc
                                    for _, child in items
                                    for gc_token, gc in child.children.items()
                                })

                else:
                    result.children.update(
                        {token: child for token, child in items}
                    )

        for token, child in multiple_expansions:
            # If we have multiple expansions, we cannot simplify them, so we keep them as is
            print(f"  Keeping multiple expansion: {token}")
            result.children[token].merge_children(child.children)

        return result
    
    def merge_children(self, children: Mapping[PartType, "TrieNode"]):
        """
        Merges the given children into this node's children.
        """
        for token, child in children.items():
            self.children[token].merge_children(child.children)

    def enumerate_raw(self) -> Iterable[tuple[PartType, ...]]:
        """
        Enumerates all possible raw token combinations represented by this trie node.
        """
        if not self.children:
            yield ()
            return

        for token, child in self.children.items():
            for suffix in child.enumerate_raw():
                yield (token,) + suffix

    def enumerate(self) -> Iterable[str]:
        """
        Enumerates all possible strings represented by this trie node.
        """
        for raw_tokens in self.enumerate_raw():
            yield "".join(str(token) for token in raw_tokens)


actual = list(
    TrieNode({"foo": TrieNode(), "bar": TrieNode(), "baz": TrieNode({})})
    .simplify()
    .children.keys()
)
expected = [BraceExpressionSet({"bar", "baz", "foo"})]
assert actual == expected, f"Expected {expected}, got {actual}"

t1 = TrieNode(
    {"qux": TrieNode({"foo": TrieNode(), "bar": TrieNode(), "baz": TrieNode({})})}
).simplify()

assert list(t1.children.keys()) == ["qux"]
t2 = t1.children["qux"]
assert list(t2.children.keys()) == [BraceExpressionSet({"bar", "baz", "foo"})]

print("Building trie...")
root = TrieNode().insert_many(tokenized_names, subtree_token="/")

# print(root.children)
# print(root.format())
# print(root.format())

actual = sorted(root.enumerate())
expected = sorted(names)
assert actual == expected, f"Expected {expected}, got {actual}"

print("Trie built successfully.")

# %%

print("Simplifying trie...")
simplified = root.simplify()

# %%
print(simplified.format())

# %%
tokenize_str(names[0])
# %%

names
# %%

subtree = root.find(tokenize_str("IHLS2014_2019_IHLS012014/organisms/validated/copepoda/"))
subtree.print()

# %%
subtree.simplify().print()
# %%
list(subtree.enumerate_raw())
# %%
