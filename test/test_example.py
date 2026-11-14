import unittest

from src.agents.example import add


class TestAdd(unittest.TestCase):
    def test_add(self):
        for left, right, expected in [(2, 3, 5), (-2, 3, 1), (0, 0, 0)]:
            with self.subTest(left=left, right=right):
                self.assertEqual(add(left, right), expected)
