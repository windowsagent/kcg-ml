import os
import numpy as np
from sklearn import metrics
from typing import List, Tuple, Union
from ascii_graph import Pyasciigraph
import joblib
import json
import torch

class LogisticRegressionPytorch(torch.nn.Module):
    def __init__(self, input_dim, output_dim, date_trained=None, tag=None):
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
    
    def generate_report(
                        reports_output_folder : str,
                        tag_name : str, 
                        text_file_lines : List[str], 
                        model_name: str,
                        ):
        """generate text file with text file lines provided, 
        save it in output directory.
        :param reports_output_folder: output folder for saving report file.
        :type reports_output_folder: str
        :param tag_name: name of the classifer tag.
        :type tag_name: str
        :param model_name: name of the model .
        :type  model_name: str
        :rtype: None. 
        """

        model_file_name = f'model-report-{model_name}-tag-{tag_name}'
        text_file_path = os.path.join(reports_output_folder ,f'{model_file_name}.txt' )
        with open( text_file_path ,"w+", encoding="utf-8") as f:
            f.writelines(text_file_lines)
        f.close()

        return
