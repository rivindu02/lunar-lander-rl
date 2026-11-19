import os
from pathlib import Path
import tempfile
import unittest

from common import checksum, summarize, write_csv, write_json
from verify_results import verify


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.previous_cwd = Path.cwd()
        os.chdir(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(os.chdir, self.previous_cwd)
        self.out = Path("results/fixture")
        self.protocol = dict(name="fixture", steps=2048, training_seeds=[42],
                             test_seeds=[20000, 20001], validation_seeds=[10000], demo_seeds=[30000],
                             eval_frequency=2048, alpha=1.0, gamma=0.99,
                             n_steps=2048, batch_size=64, source_sha256={"fixture.py": "source_hash"})
        write_json(self.out / "protocol.json", self.protocol)
        self.metadata = {}
        for condition in ("random", "baseline", "shaped"):
            run = "random" if condition == "random" else f"fixture_{condition}_seed42"
            model = Path("models") / run / "best_model.zip"
            if condition != "random":
                model.parent.mkdir(parents=True)
                model.write_bytes(b"fixture checkpoint identity only")
            rows = [dict(run_id=run, condition=condition, training_seed=42 if condition != "random" else None,
                         checkpoint=str(model) if condition != "random" else "none", episode_seed=seed,
                         **{"return": score}, length=10, terminated=True, truncated=False)
                    for seed, score in [(20000, 100), (20001, 300)]]
            metadata = dict(status="complete", actual_timesteps=2048, seed=42, reward_mode=condition,
                            validation_seeds=[10000], eval_freq=2048, alpha=1.0, gamma=0.99,
                            ppo=dict(n_steps=2048, batch_size=64, gamma=0.99),
                            source={"source_sha256": self.protocol["source_sha256"]})
            self.metadata[condition] = metadata
            path = self.out / f"{run}_episodes.csv"
            write_csv(path, rows)
            write_json(path.with_suffix(".json"), {**summarize(rows), "episode_seeds": [20000, 20001],
                       "run_metadata": metadata,
                       "checkpoint_sha256": checksum(model) if condition != "random" else None})

    def test_matched_fixture_accepted(self):
        self.assertEqual(verify(self.out), 3)

    def test_summary_tampering_rejected(self):
        import json
        path = self.out / "random_episodes.json"
        summary = json.loads(path.read_text())
        summary["mean_return"] = 999
        write_json(path, summary)
        with self.assertRaisesRegex(ValueError, "summary mismatch"):
            verify(self.out)

    def test_unequal_budget_rejected(self):
        import json
        path = self.out / "fixture_shaped_seed42_episodes.json"
        summary = json.loads(path.read_text())
        summary["run_metadata"]["actual_timesteps"] = 4096
        write_json(path, summary)
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            verify(self.out)

    def test_changed_checkpoint_rejected(self):
        (Path("models") / "fixture_baseline_seed42" / "best_model.zip").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
            verify(self.out)

    def test_test_validation_overlap_rejected(self):
        self.protocol["validation_seeds"] = [20000]
        write_json(self.out / "protocol.json", self.protocol)
        with self.assertRaisesRegex(ValueError, "seed banks overlap"):
            verify(self.out)


if __name__ == "__main__":
    unittest.main()
