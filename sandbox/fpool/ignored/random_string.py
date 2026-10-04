"""Generate random lowercase text for examples or throwaway labels.

This helper uses Python's pseudorandom generator. Results are not guaranteed
to be unique and are not suitable for passwords or security tokens.
"""

import random
import string


def random_string(length: int = 20) -> str:
    """Return `length` random ASCII lowercase letters (a-z).

    A length of zero returns an empty string; negative lengths are invalid.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


if __name__ == '__main__':
    print(random_string(30))
