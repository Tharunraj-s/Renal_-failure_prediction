import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / 'project'


class Routes(unittest.TestCase):
    def setUp(self):
        utils = types.ModuleType('utils')
        utils.biomarker_questions = {'age': 'Age'}
        utils.column_mappings = {}
        utils.save_user_input_to_csv = lambda data, path: data.to_csv(path)
        utils.load_data_predict = lambda path: path
        utils.preprocess_data = lambda data, flag: (data, [])
        utils.predict_labels = lambda *args: (['Not Chronic Kidney Disease'], 'Image not uploaded')
        spec = importlib.util.spec_from_file_location('renal_test_app', ROOT / 'app.py')
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'utils': utils}): spec.loader.exec_module(self.module)
        self.client = self.module.app.test_client()
    def test_empty_and_invalid(self):
        self.assertEqual(self.client.post('/predict', data={}).status_code, 400)
        self.assertEqual(self.client.post('/predict', data={'age': 'nan'}).status_code, 400)
    def test_inference_failure_not_normal(self):
        with patch.object(self.module, 'predict_labels', side_effect=RuntimeError('bad checkpoint')):
            response = self.client.post('/predict', data={'age': '48.5'})
        self.assertEqual(response.status_code, 503)
        self.assertIn(b'no diagnosis', response.data)
    def test_isolated_request_files(self):
        paths = []
        def prediction(frame, *args):
            self.assertTrue(Path(frame).exists())
            paths.append(str(frame))
            return (['Not Chronic Kidney Disease'], 'Image not uploaded')
        with patch.object(self.module, 'predict_labels', side_effect=prediction):
            for _ in range(2):
                self.assertEqual(self.client.post('/predict', data={'age': '48.5'}).status_code, 200)
        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(all(not Path(p).exists() for p in paths))
    def test_page_and_size_limit(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.module.app.config['MAX_CONTENT_LENGTH'] = 16
        self.assertEqual(self.client.post('/predict', data={'age': '1' * 100}).status_code, 413)


if __name__ == '__main__': unittest.main()
