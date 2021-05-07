import sys
import unittest
from unittest import TestCase

sys.path.append("..")
import json
import os
import tempfile

import models.train_model as train


class TestTrain(TestCase):
    def test_train(self):
        # Directory of the model
        model_directory = tempfile.mkdtemp()

        # Loss plot file
        loss_plot_file = tempfile.NamedTemporaryFile()

        # Train
        lossplot_web_app = train.train(
            "adam",
            10,
            0.1,
            data_path=model_directory,
            lossplot_path=loss_plot_file.name,
        )

        # Check the model is created in the specified path
        self.assertIn("mnist_model.h5", os.listdir(model_directory))

        # Check the lossplot file content
        self.assertIn("outputs", list(json.loads(lossplot_web_app[0]).keys()))
        self.assertIsInstance(json.loads(lossplot_web_app[0])["outputs"], list)
        self.assertEqual(len(json.loads(lossplot_web_app[0])["outputs"]), 1)

        # Closing all open files
        loss_plot_file.close()


if __name__ == "__main__":
    unittest.main()
