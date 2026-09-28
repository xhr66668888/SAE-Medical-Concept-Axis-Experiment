from __future__ import annotations

import sys
import unittest

from scripts.run_steering import DEFAULT_ALPHAS, parse_alphas, parse_args


class SteeringArgsTest(unittest.TestCase):
    def test_parse_alphas_uses_default_for_empty_values(self) -> None:
        self.assertEqual(parse_alphas(None), [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0])
        self.assertEqual(parse_alphas(""), [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0])

    def test_bare_alphas_flag_uses_default_grid(self) -> None:
        old_argv = sys.argv[:]
        sys.argv = ["run_steering.py", "--alphas"]
        try:
            args = parse_args()
        finally:
            sys.argv = old_argv
        self.assertEqual(args.alphas, DEFAULT_ALPHAS)

    def test_negative_alpha_grid_after_flag_is_consumed_as_value(self) -> None:
        args = parse_args(["--alphas", "-6.0,-4.0,-2.0,0.0,2.0", "--positions", "prompt_last"])
        self.assertEqual(args.alphas, "-6.0,-4.0,-2.0,0.0,2.0")
        self.assertEqual(args.positions, "prompt_last")

    def test_negative_alpha_grid_with_equals_is_supported(self) -> None:
        args = parse_args(["--alphas=-6.0,-4.0,-2.0,0.0,2.0"])
        self.assertEqual(args.alphas, "-6.0,-4.0,-2.0,0.0,2.0")


if __name__ == "__main__":
    unittest.main()
