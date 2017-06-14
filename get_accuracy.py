import numpy as np
import sys
from sklearn.metrics import accuracy_score

"""
Usage: python get_accuracy.py FILE_NAME
"""

predicted_file = sys.argv[1]
true_file = 'results/TrueLabels.npy'

predicted_labels = np.load(predicted_file)
true_labels = np.load(true_file)

accuracy = accuracy_score(true_labels, predicted_labels)
print('Accuracy: {:0.4f}'.format(accuracy))
