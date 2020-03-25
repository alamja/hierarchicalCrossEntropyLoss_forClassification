import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torchvision

class show_results:
    def __init__(self, class_num, device, model, trained_model_name = None, root_dir = './trained/'):
        self.class_num = class_num
        self.device = device
        self.trained_model_name = trained_model_name
        self.root_dir = root_dir
        if trained_model_name == 'latest':
            self.trained_model_name = sorted(os.listdir(root_dir))[-1]
            print('load model:{}'.format(self.trained_model_name))
        
        if trained_model_name is not None:
            self.model = model
            self.model.load_state_dict(torch.load('{}{}'.format(self.root_dir, self.trained_model_name)))

    def _predict(self, dataset, purpose, model, batch_size):
        if model is None:
            print('loaded pre-loaded model!')
            model = self.model
        batch_size = batch_size
        loader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size,
                                            shuffle=False, num_workers=4)
        model.to(self.device)
        model.eval()
        
        predicted = []
        answers = []
        
        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            outputs = model(inputs)
            preds = outputs.to('cpu')
            if purpose == 'Classification':
                _, preds = torch.max(outputs, 1)
            elif purpose == 'Regression':
                preds = outputs.to('cpu')
                for i in range(len(preds)):
                    preds[i] = preds[i] * (self.class_num - 1)
            
            for i in range(len(preds)):
                predicted.append(int(preds[i]))
                answers.append(int(labels[i]))
    #         print(len(predicted), predicted)        
    #         print(len(answers), answers)
        return predicted, answers

    def calc_confusion_matrix(self, dataset, purpose, model = None, batch_size=8):
        pred, ans = self._predict(dataset, purpose, model, batch_size)
        # print(len(pred), pred)
        classes_int = [int(x) for x in dataset.classes]
        cf = confusion_matrix(ans, pred, classes_int)

        df = pd.DataFrame(data=cf, index=dataset.classes, columns=dataset.classes)
        return df.style.background_gradient(cmap='Reds', axis=1)
        # sns.heatmap(df, annot=True)

    def calc_classification_report(self, dataset, purpose, model = None, batch_size=8):
        pred, ans = self._predict(dataset, purpose, model, batch_size)
        # print('Precision: {}'.format(precision_score(ans, pred, average='macro')))
        # print('Recall: {}'.format(recall_score(ans, pred, average='macro')))
        # print('F1: {}'.format(f1_score(ans, pred, average='macro')))

        df = pd.DataFrame(classification_report(ans, pred, output_dict=True))
        return df.style.background_gradient(cmap='Reds', axis=1)

    def show_error_image(self, model, dataset, purpose, min_threshold=None, max_threshold=None, batch_size=8):
        batch_size = batch_size
        loader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=batch_size,
                                                shuffle=False, num_workers=4)
        model.to(self.device)
        model.eval()
        
        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            outputs = model(inputs)
            labels = labels.to('cpu')
            if purpose == 'Classification':
                _, preds = torch.max(outputs, 1)
            elif purpose == 'Regression':
                preds = outputs.to('cpu')
                for i in range(len(preds)):
                    preds[i] = preds[i] * (self.class_num - 1)
            
            for i in range(len(preds)):
                p = int(preds[i])
                l = int(labels[i])
                if p != l:
                    if min_threshold is not None:
                        if p < min_threshold:
                            continue
                    if max_threshold is not None:
                        if p > max_threshold:
                            continue
                            
                    fig, ax = plt.subplots()
                    img = inputs[i].to('cpu')
                    img = img.numpy().transpose((1, 2, 0))
    #                 inp = np.clip(inp, 0, 1)
                    ax.imshow(img)
                    ax.set_title('predicted:{}, label:{}'.format(p, l))
    #         print(len(predicted), predicted)        
    #         print(len(answers), answers)