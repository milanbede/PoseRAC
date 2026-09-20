import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_dataset import load_own
from phase_alignment import align_examples
from train_rtmpose import build_split, sampling_weights


def example(exercise, source, phase, target, part="own_train"):
    points = np.zeros((17, 2), np.float32)
    if exercise == "jump_jack":
        points[[5, 6], 1] = -1
        points[[9, 10], 1] = -2 if phase else 0
        cls = 4
    else:
        points[[5, 6]] = [0, 0]
        points[[7, 8]] = [1, 0]
        points[[9, 10]] = [1, 1] if phase else [2, 0]
        cls = 6
    labels = np.zeros(9, np.float32)
    labels[cls] = target
    return points.reshape(-1), labels, cls, exercise, source, 42, part


class PhaseAlignmentTest(unittest.TestCase):
    def test_reversed_datasets_align_without_mutating_inputs(self):
        for exercise in ("jump_jack", "push_up"):
            data = [example(exercise, "hash", p, p) for p in (0, 1)]
            data += [example(exercise, "video", p, 1-p, "repcount_train") for p in (0, 1)]
            before = [e[1].copy() for e in data]
            aligned, report = align_examples(data)
            self.assertEqual([e[1][e[2]] for e in aligned], [0, 1, 0, 1])
            self.assertEqual({r["decision"] for r in report}, {"kept", "flipped"})
            for old, saved, new in zip(data, before, aligned):
                np.testing.assert_array_equal(old[1], saved)
                np.testing.assert_array_equal(old[0], new[0])
                self.assertEqual(old[2:], new[2:])
                self.assertEqual(np.count_nonzero(np.delete(new[1], new[2])), 0)

    def test_ambiguous_missing_phase_and_nonfinite_excluded(self):
        for data in ([example("jump_jack", "x", 0, t) for t in (0, 1)],
                     [example("jump_jack", "x", 0, 0)]):
            aligned, report = align_examples(data)
            self.assertEqual(aligned, [])
            self.assertEqual(report[0]["decision"], "excluded")
        data = [example("push_up", "x", p, p) for p in (0, 1)]
        data[0][0][10] = np.nan
        self.assertEqual(align_examples(data)[0], [])

    def test_other_exercises_unchanged(self):
        e = list(example("jump_jack", "hash", 0, 0))
        e[3] = "jab"
        aligned, report = align_examples([tuple(e)])
        self.assertIs(aligned[0][1], e[1])
        self.assertEqual(report, [])

    def test_own_weight_uses_partition_not_hashed_source(self):
        data = {"source": np.array(["a"*64, "video.mp4", "b"*64, "held.mp4"]),
                "part": np.array(["own_train", "repcount_train", "own_test", "repcount_val"])}
        train, validation = build_split(data)
        self.assertEqual(train.tolist(), [True, True, False, False])
        self.assertEqual(validation.tolist(), [False, False, False, True])
        self.assertEqual(sampling_weights(data["part"], train, 4).tolist(), [4., 1.])
        train, _ = build_split(data, exclude="a"*64, include_holdout=True)
        self.assertEqual(train.tolist(), [False, True, False, True])

    def test_test_sources_do_not_load_landmarks(self):
        import json
        sources = {"test-source": {"exercise": "jumping_jack", "split": "test"}}
        pairs = {"test-source": [{"opposite_index": 0, "return_index": 1}]}
        with patch.object(Path, "read_text", side_effect=[json.dumps(sources), json.dumps(pairs)]), \
                patch("build_dataset.np.load") as load:
            self.assertEqual(load_own(Path("unused")), ([], {}))
            load.assert_not_called()

    def test_inconsistent_recording_is_not_forced_to_one_polarity(self):
        data = [example("jump_jack", "x", phase, target)
                for phase, target in [(0, 0), (1, 1), (1, 0), (0, 1)]]
        self.assertEqual(align_examples(data)[0], [])

    def test_export_refuses_existing_candidate(self):
        import tempfile

        from export_candidate import main
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "candidates" / "existing").mkdir(parents=True)
            with patch.object(sys, "argv", ["export", "--home", directory, "--id", "existing"]), \
                    patch("torch.load", return_value={"config": {}}), patch("torch.save") as save:
                with self.assertRaisesRegex(ValueError, "already exists"):
                    main()
                save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
