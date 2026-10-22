import ast
import math
from pathlib import Path
import unittest


def parser():
    source = Path(__file__).resolve().parents[1] / 'project' / 'app.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_answer')
    scope = {'math': math, 'column_mappings': {'appetite': {'good': 0, 'poor': 1}, 'red_blood_cells': {'abnormal': 0, 'normal': 1}}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), scope)
    return scope['parse_answer']


class InputValidation(unittest.TestCase):
    def test_decimal_and_missing(self):
        parse = parser()
        self.assertEqual(parse('serum_creatinine', '0.8'), 0.8)
        self.assertIsNone(parse('age', ' '))
    def test_category_polarity(self):
        parse = parser()
        self.assertEqual(parse('appetite', 'good'), 0)
        self.assertEqual(parse('red_blood_cells', 'normal'), 1)
    def test_invalid_inputs(self):
        parse = parser()
        for value in ('nan', 'inf', '-1', 'abc'):
            with self.assertRaises(ValueError): parse('age', value)
        with self.assertRaises(ValueError): parse('appetite', 'excellent')


if __name__ == '__main__': unittest.main()
