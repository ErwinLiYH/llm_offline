import os
import tempfile
import unittest

from utils.model_path_glob import resolve_trailing_wildcard_path


class ModelPathGlobTests(unittest.TestCase):
    def test_without_wildcard_returns_path_unchanged(self):
        self.assertEqual(
            resolve_trailing_wildcard_path("checkpoints/model/final"),
            "checkpoints/model/final",
        )

    def test_trailing_wildcard_resolves_single_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            match = os.path.join(tmpdir, "ep7(step123)")
            other = os.path.join(tmpdir, "ep8(step456)")
            os.mkdir(match)
            os.mkdir(other)

            self.assertEqual(
                resolve_trailing_wildcard_path(os.path.join(tmpdir, "ep7*")),
                match,
            )

    def test_rejects_non_trailing_wildcard(self):
        with self.assertRaisesRegex(ValueError, "single trailing"):
            resolve_trailing_wildcard_path("checkpoints/ep*(step123)")

    def test_rejects_multiple_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.mkdir(os.path.join(tmpdir, "ep7(step123)"))
            os.mkdir(os.path.join(tmpdir, "ep7(step456)"))

            with self.assertRaisesRegex(ValueError, "multiple paths"):
                resolve_trailing_wildcard_path(os.path.join(tmpdir, "ep7*"))

    def test_rejects_no_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "matched no paths"):
                resolve_trailing_wildcard_path(os.path.join(tmpdir, "ep7*"))


if __name__ == "__main__":
    unittest.main()
