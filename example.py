import pandas as pd
from table_evaluator import TableEvaluator

if __name__ == "__main__":
    real = pd.read_csv("data/real_test_sample.csv")
    fake = pd.read_csv("data/fake_test_sample.csv")

    # example of how to use the TableEvaluator
    # it is recommended to use a sample of your data (e.g. 100k rows) for the evaluation
    # and to use the same number of rows for both real and fake data
    # te = TableEvaluator(real, fake, size=100000)
    te = TableEvaluator(real, fake)

    te.visual_evaluation()
    te.evaluate(target_col='Age')