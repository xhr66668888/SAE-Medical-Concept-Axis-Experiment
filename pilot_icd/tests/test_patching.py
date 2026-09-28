from __future__ import annotations

import sys
import unittest

from scripts.run_patching import DEFAULT_POSITIONS, parse_args, parse_positions


class PatchingArgsTest(unittest.TestCase):
    def test_bare_positions_flag_uses_default_positions(self) -> None:
        old_argv = sys.argv[:]
        sys.argv = ["run_patching.py", "--positions"]
        try:
            args = parse_args()
        finally:
            sys.argv = old_argv
        self.assertEqual(args.positions, DEFAULT_POSITIONS)
        self.assertEqual(parse_positions(args.positions), [-1, -2, -3, -4])

    def test_negative_positions_after_flag_are_consumed_as_value(self) -> None:
        args = parse_args(["--positions", "-1,-2,-3,-4", "--layers", "window:2"])
        self.assertEqual(args.positions, "-1,-2,-3,-4")
        self.assertEqual(parse_positions(args.positions), [-1, -2, -3, -4])

    def test_negative_positions_with_equals_are_supported(self) -> None:
        args = parse_args(["--positions=-1,-2"])
        self.assertEqual(args.positions, "-1,-2")
        self.assertEqual(parse_positions(args.positions), [-1, -2])


if __name__ == "__main__":
    unittest.main()
