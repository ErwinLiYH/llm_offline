import os
import subprocess
import tempfile
import unittest

import yaml

from utils.experiment_config import save_experiment_config_snapshot


class ExperimentConfigSnapshotTest(unittest.TestCase):
    def _git(self, repo, *args):
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _write_text(self, path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def _init_repo(self, tmpdir):
        repo = os.path.join(tmpdir, "repo")
        os.makedirs(repo)
        self._git(repo, "init")
        self._git(repo, "config", "user.email", "tester@example.com")
        self._git(repo, "config", "user.name", "Test User")
        self._write_text(os.path.join(repo, "tracked.txt"), "original\n")
        self._write_text(os.path.join(repo, "delete_me.txt"), "delete me\n")
        self._write_text(os.path.join(repo, "rename_me.txt"), "rename me\n")
        with open(os.path.join(repo, "tracked_binary.bin"), "wb") as f:
            f.write(b"\x00original")
        self._git(repo, "add", ".")
        self._git(repo, "commit", "-m", "initial")
        return repo

    def test_saves_config_under_experiment_id(self):
        config = {
            "experiment_id": "abc123ef",
            "env_family": "pointmaze",
            "resolved_train_variants": ["large"],
            "global_effective_batch_size": 16,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = self._init_repo(tmpdir)
            paths = save_experiment_config_snapshot(config, root=tmpdir, repo_dir=repo)

            self.assertEqual(paths["config"], os.path.join(tmpdir, "abc123ef", "config.yaml"))
            self.assertEqual(paths["git"], os.path.join(tmpdir, "abc123ef", "git.yaml"))
            self.assertEqual(paths["patch"], os.path.join(tmpdir, "abc123ef", "dirty.patch"))
            with open(paths["config"], encoding="utf-8") as f:
                saved = yaml.safe_load(f)
            self.assertEqual(saved, config)

    def test_clean_repo_writes_empty_patch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = self._init_repo(tmpdir)
            paths = save_experiment_config_snapshot(
                {"experiment_id": "clean"},
                root=os.path.join(tmpdir, "snapshots"),
                repo_dir=repo,
            )

            with open(paths["git"], encoding="utf-8") as f:
                metadata = yaml.safe_load(f)

            self.assertTrue(metadata["available"])
            self.assertEqual(metadata["repo_root"], repo)
            self.assertFalse(metadata["dirty"])
            self.assertEqual(metadata["status_porcelain"], [])
            self.assertEqual(metadata["patch_file"], "dirty.patch")
            self.assertEqual(metadata["patch_bytes"], 0)
            self.assertEqual(os.path.getsize(paths["patch"]), 0)
            self.assertEqual(metadata["skipped_files"], [])

    def test_dirty_patch_restores_text_changes_and_skips_binary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = self._init_repo(tmpdir)
            self._write_text(os.path.join(repo, "tracked.txt"), "changed\n")
            os.remove(os.path.join(repo, "delete_me.txt"))
            os.rename(os.path.join(repo, "rename_me.txt"), os.path.join(repo, "renamed.txt"))
            self._write_text(os.path.join(repo, "new_file.txt"), "new file\n")
            with open(os.path.join(repo, "tracked_binary.bin"), "wb") as f:
                f.write(b"\x00changed")
            with open(os.path.join(repo, "binary.bin"), "wb") as f:
                f.write(b"\x00\x01\x02")

            paths = save_experiment_config_snapshot(
                {"experiment_id": "dirty"},
                root=os.path.join(tmpdir, "snapshots"),
                repo_dir=repo,
            )
            with open(paths["git"], encoding="utf-8") as f:
                metadata = yaml.safe_load(f)
            with open(paths["patch"], "rb") as f:
                patch = f.read()

            self.assertTrue(metadata["available"])
            self.assertTrue(metadata["dirty"])
            self.assertEqual(metadata["patch_bytes"], len(patch))
            self.assertIn(
                {"path": "tracked_binary.bin", "reason": "binary"},
                metadata["skipped_files"],
            )
            self.assertIn({"path": "binary.bin", "reason": "binary"}, metadata["skipped_files"])
            self.assertIn(b"tracked.txt", patch)
            self.assertIn(b"delete_me.txt", patch)
            self.assertIn(b"rename_me.txt", patch)
            self.assertIn(b"renamed.txt", patch)
            self.assertIn(b"new_file.txt", patch)
            self.assertNotIn(b"tracked_binary.bin", patch)
            self.assertNotIn(b"binary.bin", patch)

            self._git(repo, "reset", "--hard", "HEAD")
            for filename in ("new_file.txt", "renamed.txt", "binary.bin"):
                path = os.path.join(repo, filename)
                if os.path.exists(path):
                    os.remove(path)
            self._git(repo, "apply", paths["patch"])

            with open(os.path.join(repo, "tracked.txt"), encoding="utf-8") as f:
                self.assertEqual(f.read(), "changed\n")
            self.assertFalse(os.path.exists(os.path.join(repo, "delete_me.txt")))
            self.assertFalse(os.path.exists(os.path.join(repo, "rename_me.txt")))
            with open(os.path.join(repo, "renamed.txt"), encoding="utf-8") as f:
                self.assertEqual(f.read(), "rename me\n")
            with open(os.path.join(repo, "new_file.txt"), encoding="utf-8") as f:
                self.assertEqual(f.read(), "new file\n")
            self.assertFalse(os.path.exists(os.path.join(repo, "binary.bin")))

    def test_non_git_directory_writes_unavailable_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_experiment_config_snapshot(
                {"experiment_id": "nogit"},
                root=os.path.join(tmpdir, "snapshots"),
                repo_dir=tmpdir,
            )

            with open(paths["git"], encoding="utf-8") as f:
                metadata = yaml.safe_load(f)

            self.assertFalse(metadata["available"])
            self.assertIn("error", metadata)
            self.assertFalse(metadata["dirty"])
            self.assertEqual(metadata["patch_bytes"], 0)
            self.assertEqual(os.path.getsize(paths["patch"]), 0)

    def test_requires_experiment_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                save_experiment_config_snapshot({}, root=tmpdir)

    def test_rejects_nested_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                save_experiment_config_snapshot(
                    {"experiment_id": "abc123ef"},
                    root=tmpdir,
                    filename="nested/config.yaml",
                )


if __name__ == "__main__":
    unittest.main()
