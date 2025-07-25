import pandas as pd
from table_evaluator import TableEvaluator


def test_table_evaluator_init():
    real_data = pd.DataFrame(
        {
            'col1': [1, 2, 3, 4, 5],
            'col2': ['A', 'B', 'A', 'C', 'B'],
            'col3': [10.1, 11.2, 12.3, 13.4, 14.5],
        }
    )
    fake_data = pd.DataFrame(
        {
            'col1': [6, 7, 8, 9, 10],
            'col2': ['B', 'C', 'A', 'A', 'C'],
            'col3': [15.1, 16.2, 17.3, 18.4, 19.5],
        }
    )
    evaluator = TableEvaluator(real_data, fake_data)
    assert evaluator.real is not None
    assert evaluator.fake is not None
    assert len(evaluator.real) == len(real_data)
    assert len(evaluator.fake) == len(fake_data)


def test_evaluate_method():
    real_data = pd.DataFrame(
        {
            'col1': [1, 2, 3, 4, 5],
            'col2': ['A', 'B', 'A', 'C', 'B'],
            'col3': [10.1, 11.2, 12.3, 13.4, 14.5],
        }
    )
    fake_data = pd.DataFrame(
        {
            'col1': [6, 7, 8, 9, 10],
            'col2': ['B', 'C', 'A', 'A', 'C'],
            'col3': [15.1, 16.2, 17.3, 18.4, 19.5],
        }
    )
    evaluator = TableEvaluator(real_data, fake_data)
    results = evaluator.evaluate(target_col='col1', target_type='regr', return_outputs=True)
    assert isinstance(results, dict)
    expected_keys = [
        'Overview Results',
        'Regressor MSE-scores',
        'Privacy Results',
        'Jensen-Shannon distance',
        'Kolmogorov-Smirnov statistic',
    ]
    for key in expected_keys:
        assert key in results


def test_evaluate_method_class():
    real_data = pd.DataFrame(
        {
            'col1': [1, 2, 3, 4, 5],
            'col2': ['A', 'B', 'A', 'C', 'B'],
            'col3': [10.1, 11.2, 12.3, 13.4, 14.5],
        }
    )
    fake_data = pd.DataFrame(
        {
            'col1': [6, 7, 8, 9, 10],
            'col2': ['B', 'C', 'A', 'A', 'C'],
            'col3': [15.1, 16.2, 17.3, 18.4, 19.5],
        }
    )
    evaluator = TableEvaluator(real_data, fake_data, cat_cols=['col1', 'col2'])
    results = evaluator.evaluate(target_col='col2', target_type='class', return_outputs=True)
    assert isinstance(results, dict)
    expected_keys = [
        'Overview Results',
        'Classifier F1-scores and their Jaccard similarities:',
        'Privacy Results',
        'Jensen-Shannon distance',
        'Kolmogorov-Smirnov statistic',
    ]
    for key in expected_keys:
        assert key in results
