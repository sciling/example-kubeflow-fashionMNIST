from unittest import TestCase
import unittest
import sys
sys.path.append('..')
import src.confusion_matrix as confusion_matrix
import json
import pathlib


DATA_DIR = f"{pathlib.Path(__file__).parent.absolute()}/../data/test"


class TestTrain(TestCase):
    def test_confusion_matrix(self):
        # Confusion matrix call
        confusion_matrix_web_app = confusion_matrix.confusion_matrix(labels_dir=DATA_DIR)

        # Check output
        self.assertIn('outputs', list(json.loads(confusion_matrix_web_app[0]).keys()))
        self.assertIsInstance(json.loads(confusion_matrix_web_app[0])['outputs'], list)
        self.assertEqual(len(json.loads(confusion_matrix_web_app[0])['outputs']), 1)
        matrix_data = json.loads(confusion_matrix_web_app[0])['outputs'][0]
        self.assertEqual(matrix_data['type'], 'confusion_matrix')
        self.assertEqual(matrix_data['storage'], 'inline')


if __name__ == '__main__':
    unittest.main()
