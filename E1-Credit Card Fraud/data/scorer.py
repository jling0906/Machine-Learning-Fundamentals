import argparse
import pandas as pd
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser(description='Read in predicted class and true class')
parser.add_argument('--predicted_file', type=str, help='location of predicted file')
parser.add_argument('--true_file', type=str, help='location of file with true classes')
args = parser.parse_args()

predict = pd.read_csv(args.predicted_file)
true = pd.read_csv(args.true_file)
print(f1_score(true['Class'], predict['Class']))
