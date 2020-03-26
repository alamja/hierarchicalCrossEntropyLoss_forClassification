# hierarchicalCrossEntropyLoss_forClassification
Hierarchical CrossEntropy Loss function on PyTorch for taking account hierarchical categories.  

## Motivation
I sometimes want to set hierarchical structured categories to our dataset. For example, when the labels are not completely independent (i.g. Progress recognition. When the labels are continuous numbers, and they have a connection with each others.), I believe we should measure the distance from the estimated labels to the right labels. The one conceivable way is to train the model as a Regression using MSE or the other Metrics for Regression. Because, it allows us to measure the difference between the answer and the estimated label. I guess it would be useful in some datasets.  
However, I tried this approach for my own dataset. It didn't work well. Also, I could not find such a Loss function in PyTorch original implementation and could not find any implementation for Classification (I've found some implementations for NLP, but I wanted to make this Loss function more simple than them.). So, it motivated me to develop custom Loss Function.

## Environments
I just used this version. I haven't confirmed if it works on the other versions.
- Python 3+
- PyTorch 1.4
- Scikit-Learn 0.22.1

## Supposed actual directory layout
    .
    └── data            # Image root directory
        ├── train       # Training images root
        │   ├── 0       # Each categories (In this case, this dataset has n-classes)
        │   ├── ...
        │   └── n
        ├── val         # Validation images root (sub folders are same as 'train')
        └── test        # Test images root (sub folders are same as 'train')


## Supposed categories hierarchical structure (i.g. ClassNum=9)
Here is the sample of the hierarchy which supposed to use.

    .
    ├── G1              # Hierarchy level 1
    │   ├── SG1         # Hierarchy level 2
    │   │   ├── 0       # Hierarchy level 0
    │   │   └── 1
    │   └── SG2  
    │       ├── 2
    │       └── 3  
    └── G2
        ├── SG3
        │   ├── 4
        │   ├── 5
        │   └── 6
        └── SG4  
            ├── 7
            └── 8 
    

To specify the above hierarchy to the hierarchicalCrossEntropyLoss function. You can prepare the hierarchy as follows.

```python
 h_dict = {'0': 'Normal Multiclass Classification', # this value is meaningless
           '1': {'0': [1, 1, 1, 1, 0, 0, 0, 0, 0],
                 '1': [0, 0, 0, 0, 1, 1, 1, 1, 1]},
           '2': {'0': [1, 1, 0, 0, 0, 0, 0, 0, 0],
                 '1': [0, 0, 1, 1, 0, 0, 0, 0, 0],
                 '2': [0, 0, 0, 0, 1, 1, 1, 0, 0],
                 '3': [0, 0, 0, 0, 0, 0, 0, 1, 1]}}
```

```python
from core.hierarchicalCrossEntropyLoss import hierarchicalCrossEntropyLoss as h_loss
from utils import data_convert_funcs as convert_funcs

label_spilitters = []
for i in range(1, len(hierarchy_dict)):
    label_spilitters.append(convert_funcs.convertHierarchyDict2labelSplitters(h_dict[str(i)]))

coefficient = [0.50, 0.25, 0.25]  # You can adjust the effectiveness of each hierarchy
criterion = h_loss(coefficient, h_dict, device=device)
```

Or, if you want to use class weight, you can set them as belows.

```python
from core.hierarchicalCrossEntropyLoss import hierarchicalCrossEntropyLoss as h_loss
from utils import data_convert_funcs as convert_funcs
from sklearn.utils.class_weight import compute_class_weight

label_spilitters = []
for i in range(1, len(hierarchy_dict)):
    label_spilitters.append(convert_funcs.convertHierarchyDict2labelSplitters(h_dict[str(i)]))

coefficient = [0.50, 0.25, 0.25]  # You can adjust the effectiveness of each hierarchy

sample_weights = compute_class_weight(class_weight='balanced', 
                                      classes=np.unique(labels_list['train']),
                                      y=labels_list['train'])
sample_weights = [torch.FloatTensor(sample_weights).to(device)] # weight for level 0
for i in range(len(coefficient)-1):
    temp_classes = convert_funcs.convertClass2HierarchicalClass(labels_list['train'], label_spilitters[i])
    temp_weights = compute_class_weight(class_weight='balanced', 
                                        classes=np.unique(temp_classes),
                                        y=temp_classes)
    temp_weights = convert_funcs.convertClassWeights2HierarchicalClassWeights(temp_weights, label_spilitters[i])
    temp_weights = torch.FloatTensor(temp_weights).to(device)
    sample_weights.append(temp_weights)

criterion = h_loss(coefficient, h_dict, sample_weights, device=device)    
```

## Results

* Trained as Regression using MSE Loss
![Trained as Regression using MSE Loss](/images/Regression.png)

* PyTorch Original CrossEntropy Loss
![PyTorch Original CrossEntropy Loss](/images/CELoss.png)

* PyTorch Original CrossEntropy Loss with Class weight
![PyTorch Original CrossEntropy Loss with Class weight](/images/WeightedCELoss.png)

* Hierarchical CrossEntropy Loss with Class weight (Coefficient=[0.34, 0.33, 0.33])
![Hierarchical CrossEntropy Loss with Class weight (Coefficient=[0.34, 0.33, 0.33])](/images/HierarchicalCELoss(Conf0.34_0.33_0.33).png)

* Hierarchical CrossEntropy Loss with Class weight (Coefficient=[0.50, 0.10, 0.40])
![Hierarchical CrossEntropy Loss with Class weight (Coefficient=[0.50, 0.10, 0.40])](/images/HierarchicalCELoss(Conf0.5_0.1_0.4).png)

## Conclusion

In this experimentations, the improvements are not clear. However, I guess when the number of classes is bigger than this example, it would be useful to make the results desireble. 