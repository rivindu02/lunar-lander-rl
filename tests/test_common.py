import csv
import math
from pathlib import Path
import tempfile
import unittest

from common import (DEMO_SEEDS, TEST_SEEDS, VALIDATION_SEEDS, checksum,
                    episodes, summarize, write_csv, write_json)


class ActionSpace:
    def seed(self, seed):
        self.value = seed % 4

    def sample(self):
        return self.value


class FixtureEnv:
    def __init__(self):
        self.action_space = ActionSpace()
        self.resets, self.actions = [], []

    def reset(self, *, seed):
        self.resets.append(seed)
        self.step_count = 0
        return [seed], {}

    def step(self, action):
        self.actions.append(action)
        self.step_count += 1
        done = self.step_count == 2
        return [0], 100.0, done, False, {}


class MeasurementTests(unittest.TestCase):
    def test_statistics_known_fixture(self):
        summary = summarize([{"return": 100, "truncated": False},
                             {"return": 300, "truncated": "True"}])
        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["mean_return"], 200)
        self.assertAlmostEqual(summary["std_return"], math.sqrt(20_000))
        self.assertEqual(summary["fraction_ge_200"], 0.5)
        self.assertEqual(summary["truncation_fraction"], 0.5)

    def test_empty_summary_rejected(self):
        with self.assertRaises(ValueError):
            summarize([])

    def test_episode_protocol_and_random_seeding(self):
        env = FixtureEnv()
        rows = episodes(None, [12, 13], env=env)
        self.assertEqual(env.resets, [12, 13])
        self.assertEqual(env.actions, [0, 0, 1, 1])
        self.assertEqual([r["return"] for r in rows], [200, 200])
        self.assertEqual([r["length"] for r in rows], [2, 2])
        self.assertTrue(all(r["terminated"] and not r["truncated"] for r in rows))

    def test_deterministic_actions_and_truncation(self):
        class Policy:
            def predict(self, obs, deterministic):
                self.deterministic = deterministic
                return 3, None

        class TruncatedEnv(FixtureEnv):
            def step(self, action):
                obs, reward, terminated, _, info = super().step(action)
                return obs, reward, False, terminated, info

        policy, env = Policy(), TruncatedEnv()
        rows = episodes(policy, [9], env=env, condition="baseline", run_id="fixture",
                        training_seed=42, checkpoint="fixture.zip")
        self.assertTrue(policy.deterministic)
        self.assertEqual(env.actions, [3, 3])
        self.assertFalse(rows[0]["terminated"])
        self.assertTrue(rows[0]["truncated"])
        self.assertEqual(rows[0]["training_seed"], 42)

    def test_seed_banks_disjoint(self):
        banks = [set(VALIDATION_SEEDS), set(TEST_SEEDS), set(DEMO_SEEDS)]
        self.assertEqual([len(b) for b in banks], [10, 100, 3])
        for i, bank in enumerate(banks):
            for other in banks[i + 1:]:
                self.assertFalse(bank & other)

    def test_artifact_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "nested" / "episodes.csv"
            rows = [{"return": 200.5, "truncated": False}]
            write_csv(csv_path, rows)
            with csv_path.open() as handle:
                loaded = list(csv.DictReader(handle))
            self.assertEqual(summarize(loaded)["mean_return"], 200.5)
            path = csv_path.with_suffix(".json")
            write_json(path, summarize(loaded))
            self.assertEqual(len(checksum(path)), 64)
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
