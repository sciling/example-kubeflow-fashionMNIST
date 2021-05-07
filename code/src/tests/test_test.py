import sys
import unittest
from unittest import TestCase

sys.path.append("../..")
import os
import pathlib
import tempfile

import src.models.predict_model as test

DATA_DIR = f"{pathlib.Path(__file__).parent.absolute()}/../../data/test"


class TestTrain(TestCase):
    def test_test(self):
        # Results file
        results_file = tempfile.NamedTemporaryFile()

        # Labels directory
        labels_directory = tempfile.mkdtemp()

        # Test call
        test.test(
            data_path=DATA_DIR,
            results_path=results_file.name,
            labels_dir=labels_directory,
        )

        # Check the labels are correctly created in the directory
        self.assertIn("true_labels.txt", os.listdir(labels_directory))
        self.assertIn("pred_labels.txt", os.listdir(labels_directory))
        self.assertIn("class_names.txt", os.listdir(labels_directory))

        # Check results file is not empty
        self.assertNotEqual(os.path.getsize(results_file.name), 0)


if __name__ == "__main__":
    unittest.main()
