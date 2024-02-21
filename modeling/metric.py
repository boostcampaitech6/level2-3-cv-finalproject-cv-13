from typing import Tuple

import torch
import torcheval.metrics as metrics
from collections import defaultdict

# from torcheval.metrics 

class Metric:
    def __init__(self):
        # self.__total_metrics = defaultdict(list)
        self.__batch_metrics = defaultdict(float)
        self.batch_len = 0

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
        """ This method returns metric value.
        Args:
            y_true: array-like of shape (n_samples,)
            y_pred: array-like of shape (n_samples,)
        Returns:
            dict: dictionary containing the metric value
        """
        result = self.calculate(y_true, y_pred)
        self.step_update(result)
        self.batch_len += 1
        return result        
    
    def preprocess(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float]:
        """ This method returns the confusion matrix data(TN, FP, FN, TP).
        Args:
            y_true: ground truth (Shape: (N,))
            y_pred: predicted values (Shape: (N,))
        Returns:
            tuple: confusion matrix (TN, FP, FN, TP)
        """
        conf = metrics.functional.binary_confusion_matrix(y_true, y_pred)
        return conf[0][0], conf[0][1], conf[1][0], conf[1][1]
    
    def calculate(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
        """ This method calculates the metric value.
        Args:
            y_true: ground truth (Shape: (N,))
            y_pred: predicted values (Shape: (N,))
        Returns:
            dict: dictionary containing the metric value 
        """
        tn, fp, fn, tp = self.preprocess(y_true, y_pred)
        result = {
            "acc": (tp + tn) / (tp + tn + fp + fn),
            "recall": tp / (tp + fn),
            "precision": tp / (tp + fp),
            "specificity": tn / (tn + fp),
            "f1": 2*tp / (2*tp + fp + fn),
            "threat": tp / (tp + fn + fp),
            "mcc": (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5,
            "auroc": metrics.functional.binary_auroc(y_true, y_pred)
        }
        return result
    
    def step_update(self, result: dict) -> None:
        """ This method updates the batch metrics.
        Args:
            result: dictionary containing the metric value
        Returns:
            None
        """
        for k, v in result.items():
            self.__batch_metrics[k] += v

    def update(self) -> dict:
        """ This method returns the average of the batch metrics.
        Args:
            None
        Returns:
            dict: dictionary containing the average of the batch metrics
        """
        for k, v in self.__batch_metrics.items():
            self.__batch_metrics[k] = (v / self.batch_len).numpy()
            # self.__total_metrics[k].append(self.__batch_metrics[k])
        return self.__batch_metrics

    def clear_batch_metrics(self):
        """ This method clears the batch metrics.
        """
        self.__batch_metrics.clear()


if __name__ == '__main__':
    # softmax => argmax
    # metric = metrics.BinaryAccuracy()
    metric = Metric()
    input = torch.tensor([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
    target = torch.tensor([1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1])
    metric(input, target)

