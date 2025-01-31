import os
import numpy as np
from sklearn import metrics
from typing import List, Tuple, Union
from ascii_graph import Pyasciigraph
import joblib
import json
import torch

class LogisticRegressionPytorch(torch.nn.Module):
    def __init__(self, input_dim, output_dim = 1, date_trained=None, tag=None):
        super(LogisticRegressionPytorch, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.date_trained = date_trained
        self.tag = tag

    def forward(self, x):
            return torch.sigmoid(self.linear(x))

    def train_loop(
                        model,
                        train_emb,
                        train_labels,
                        epochs: int = 20000, 
                        ):
            """Taining loop for LogisticRegressionPytorch object.
            
            :param model: LogisticRegressionPytorch model object
            :type model: LogisticRegressionPytorch
            :param train_emb: embedding for the training features.
            :type train_emb: Numpy.Ndarray.
            :param train_labels: labels for training set.
            :type train_labels: Numpy.NdArray.
            :param epochs: number of epochs
            :type epochs: int
            :returns: model after training. 
            :rtype: LogisticRegressionPytorch
            """

            # Converting the dataset into Pytorch tensors.
            train_emb   =torch.from_numpy(train_emb.astype(np.float32))
            train_labels=torch.from_numpy(train_labels.astype(np.float32))
            train_labels=train_labels.view(train_labels.shape[0],1)

            criterion   = torch.nn.BCELoss()
            optimizer   = torch.optim.SGD(model.parameters(),lr=0.01)

            for epoch in range(epochs+1):
                
                y_prediction=model(train_emb)
                loss=criterion(y_prediction,train_labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # if (epoch+1)%10000 == 0:
                #     print('epoch:', epoch+1,',loss=',loss.item())
            
            return model
    

    def save_model(self, file_path: str):
        """Save the model to the specified file path.

        :param file_path: The path (including filename) to save the model.
        :type file_path: str
        """
        torch.save(self.state_dict(), file_path)
        
    def load_model(file_path: str, input_dim: int, output_dim: int, date_trained=None, tag=None):
        """Load the model from the specified file path.

        :param file_path: The path (including filename) to load the model.
        :type file_path: str
        :param input_dim: The input dimension for the model.
        :type input_dim: int
        :param output_dim: The output dimension for the model.
        :type output_dim: int
        :param date_trained: The date the model was trained, defaults to None
        :type date_trained: str, optional
        :param tag: The tag of the model, defaults to None
        :type tag: str, optional
        :return: The loaded model
        :rtype: LogisticRegressionPytorch
        """
        model = LogisticRegressionPytorch(input_dim, output_dim, date_trained, tag)
        model.load_state_dict(torch.load(file_path))
        return model

    
    def calc_confusion_matrix(
                            test_labels , 
                            predictions ,
                            tag_name : str 
                            ):
        """calculate accuracy, confusion matrix parts and return them.
        :param test_labels: labels for the test embeddings.
        :type test_labels: NdArray
        :param predictions: prediction from the classifer for the test_labels.
        :type predictions: NdArray
        :returns: accuracy,false positive rate, false negative rate, true positive rate, \
                true negative rate, false positive, false negative, true positive, true negative.
        :rtype: list of strings  
        """
        accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
        confusion_matrix = metrics.confusion_matrix(test_labels, predictions)
        FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
        FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        TP = np.diag(confusion_matrix)
        TN = confusion_matrix.sum() - (FP + FN + TP)
        ALL_SUM = FP + FN + TP + TN
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        return [ 
                f'false-positive-rate: {FPR[0] :.4f}  \n', 
                f'false-negative-rate: {FNR[0] :.4f}  \n',
                f'true-positive-rate : {TPR[0] :.4f}  \n',
                f'true-negative-rate : {TNR[0] :.4f}  \n\n',
                f'false-positive :  {FP[0]} out of {ALL_SUM[0]}  \n',
                f'false-negative : {FN[0]}  out of {ALL_SUM[0]} \n',
                f'true-positive : {TP[0]} out of {ALL_SUM[0]}  \n',
                f'true-negative : {TN[0]} out of {ALL_SUM[0]}  \n\n',
                f'>Accuracy : {accuracy:.4f}\n\n',
                f"Classification Report : \n\n{metrics.classification_report(test_labels, predictions)}\n\n",
                f"Index 0 is class {tag_name}\n",
                "Index 1 is class other \n\n"
                ]
    
  
