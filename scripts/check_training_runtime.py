"""CPU-only checks: python scripts/check_training_runtime.py"""
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from checkpoint_server import CheckpointHandler, get_runs
from training_runtime import iteration_limit, training_progress


class TrainingLimitTests(unittest.TestCase):
    def test_unlimited_and_finite_budgets(self):
        for budget in (-1, None):
            self.assertIsNone(iteration_limit(budget, 1, 131072))
        self.assertEqual(iteration_limit(500_000_000, 1, 131072), 3814)
        self.assertEqual(iteration_limit(500_000_000, 2, 131072), 1907)
        for budget in (0, -2, 1):
            with self.assertRaises(ValueError):
                iteration_limit(budget, 1, 131072)

    def test_infinite_schedule_reaches_and_holds_final_progress(self):
        self.assertEqual(training_progress(0, None, 100), 0)
        self.assertEqual(training_progress(99, None, 100), 1)
        self.assertEqual(training_progress(10000, None, 100), 1)
        self.assertAlmostEqual(training_progress(25, 101, 10), 0.25)


class CheckpointServerTests(unittest.TestCase):
    def test_discovery_download_and_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run-20200101_000000-example"
            files = run / "files"
            files.mkdir(parents=True)
            ckpt = files / "checkpoint_temp.pt"
            ckpt.write_bytes(b"completed checkpoint")
            (files / "checkpoint_latest.pt").symlink_to(ckpt.name)
            (files / "cfg.yaml").write_text("total_frames: -1\n")
            legacy_run = root / "2020-01-01" / "legacy"
            legacy_files = legacy_run / "wandb/latest-run/files"
            legacy_files.mkdir(parents=True)
            (legacy_files / "checkpoint_latest.pt").write_bytes(b"legacy")
            self.assertEqual({p for p, _ in get_runs(root)}, {run, legacy_run})

            server = HTTPServer(("127.0.0.1", 0), CheckpointHandler)
            server.root = root
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base) as response:
                    self.assertIn(run.name.encode(), response.read())
                url = f"{base}/download/{run.name}"
                with urlopen(Request(url, method="HEAD")) as response:
                    self.assertEqual(int(response.headers["Content-Length"]), ckpt.stat().st_size)
                    self.assertIsNotNone(response.headers["Last-Modified"])
                with urlopen(url) as response:
                    self.assertEqual(response.read(), b"completed checkpoint")
                # Writer publishes a new file while retaining the same latest URL.
                temporary = files / "checkpoint_temp.pt.tmp"
                temporary.write_bytes(b"next completed checkpoint")
                temporary.replace(ckpt)
                with urlopen(url) as response:
                    self.assertEqual(response.read(), b"next completed checkpoint")
                with urlopen(f"{base}/config/{run.name}") as response:
                    self.assertIn(b"total_frames: -1", response.read())
                with urlopen(f"{base}/download/2020-01-01/legacy") as response:
                    self.assertEqual(response.read(), b"legacy")
                for path in ("/download/%2E%2E", "/download/missing"):
                    with self.assertRaises(HTTPError) as caught:
                        urlopen(base + path)
                    caught.exception.close()
            finally:
                server.shutdown()
                thread.join()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
