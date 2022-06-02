# ExplaiNN

<img src="Figures\ExplaiNN.png" style="zoom:55%;" />

ExplaiNN is an adaptation of neural additive models ([NAMs](https://arxiv.org/abs/2004.13912)) for genomic tasks wherein predictions are computed as a linear combination of multiple independent CNNs, each consisting of a single convolutional filter and fully connected layers. This approach brings together the expressivity of CNNs with the interpretability of linear models, providing global (cell state level) as well as local (individual sequence level) insights of the biological processes studied.

ExplaiNN can be found with the following link: [biorxiv manuscript](https://www.biorxiv.org/content/10.1101/2022.05.20.492818v1).

## Installation

Explainn library is available on pip and can be installed with:

```
pip install explainn==0.1.5
```

Note that torch should be installed in the environment prior to explainn. If you encounter **ERROR: No matching distribution** type of errors, try to install the following libraries first:

```
numpy==1.21.6
h5py==3.6.0
tqdm==4.64.0
pandas==1.3.5
matplotlib==3.5.2
```

A normal successful installation should finish in a few minutes.

## Example of training an ExplaiNN model on TF binding data

Here I give an example of how one can train and interpret an ExplaiNN model on predicting the binding of 3 TFs: FOXA1, MAX, and JUND. The dataset can be found [here](https://drive.google.com/drive/folders/1tFWWTCUoE2Jg0zrMvKKtTqEBuwkkJ1bl). 

### Initialize the model

Imports:

```python
from explainn import tools
from explainn import networks
from explainn import train
from explainn import test
from explainn import interpretation

import torch
import os
from torch import nn
from sklearn.metrics import average_precision_score
from sklearn import metrics
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
```

Model and parameter initialization:

```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 15
batch_size = 128
learning_rate = 0.001

dataloaders, target_labels, train_out = tools.load_datas("data/tf_peaks_TEST_sparse_Remap.h5", batch_size,
                                                         0, True)

target_labels = [i.decode("utf-8") for i in target_labels]

num_cnns = 100
input_length = 200
num_classes = len(target_labels)
filter_size = 19


model = networks.ExplaiNN(num_cnns, input_length, num_classes, filter_size).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```

### Train the model

Code for training the model

```python
os.makedirs("Output_weights")
weights_folder = "Output_weights"

model, train_error, test_error = train.train_explainn(dataloaders['train'],
                                                dataloaders['valid'], model,
                                                device, criterion, optimizer, num_epochs,
                                                weights_folder, name_ind, verbose=True,
                                                trim_weights=False) 

showPlot(train_error, test_error, "Loss trend", "Loss")
```

<img src="Figures\example_train.png" style="zoom:100%;" />

### Testing the model

```python
model.load_state_dict(torch.load("checkpoints/model_epoch_9_.pth"))

labels_E, outputs_E = test.run_test(model, dataloaders['test'], device)
pr_rec = average_precision_score(labels_E, outputs_E)

no_skill_probs = [0 for _ in range(len(labels_E[:, 0]))]
ns_fpr, ns_tpr, _ = metrics.roc_curve(labels_E[:, 0], no_skill_probs)

roc_aucs = {}
raw_aucs = {}
roc_prcs = {}
raw_prcs = {}
for i in range(len(target_labels)):
    nn_fpr, nn_tpr, threshold = metrics.roc_curve(labels_E[:, i], outputs_E[:, i])
    roc_auc_nn = metrics.auc(nn_fpr, nn_tpr)

    precision_nn, recall_nn, thresholds = metrics.precision_recall_curve(labels_E[:, i], outputs_E[:, i])
    pr_auc_nn = metrics.auc(recall_nn, precision_nn)

    raw_aucs[target_labels[i]] = nn_fpr, nn_tpr
    roc_aucs[target_labels[i]] = roc_auc_nn

    raw_prcs[target_labels[i]] = recall_nn, precision_nn
    roc_prcs[target_labels[i]] = pr_auc_nn

print(roc_prcs)
print(roc_aucs)
```

```
{'MAX': 0.825940403552367, 'FOXA1': 0.8932791261118389, 'JUND': 0.749391895435854}
{'MAX': 0.8031278582930756, 'FOXA1': 0.8065550331791597, 'JUND': 0.7463422694967192}
```

### Interpretation

#### Unit/filter annotation

Visualizing filters

```python
dataset, data_inp, data_out =tools.load_single_data("data/tf_peaks_TEST_sparse_Remap.h5", 
                                                     batch_size, 0, False)

predictions, labels = interpretation.get_explainn_predictions(dataset, model, device,
                                                              isSigmoid=True)

# only well predicted sequences
pred_full_round = np.round(predictions)
arr_comp = np.equal(pred_full_round, labels)
idx = np.argwhere(np.sum(arr_comp, axis=1) == len(target_labels)).squeeze()

data_inp = data_inp[idx, :, :]
data_out = data_out[idx, :]

dataset = torch.utils.data.TensorDataset(data_inp, data_out)
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size, shuffle=False,
                                                  num_workers=0)

activations = interpretation.get_explainn_unit_activations(data_loader, model, device)
pwms = interpretation.get_pwms_explainn(activations, data_inp, data_out, filter_size)
interpretation.pwm_to_meme(pwms, "data/explainn_filters.meme")
```

Tomtom annotation:

```bash
./../../meme-5.3.3/src/tomtom -oc MAX_JUND_FOXA1_tomtom explainn_filters.meme ../../tomtom_results/JASPAR2020_CORE_vertebrates_non-redundant_pfms_meme.txt
```

#### Output layer weights visualization

Reading the tomtom's annotation:

```python
tomtom_results = pd.read_csv("data/MAX_JUND_FOXA1_tomtom/tomtom.tsv",
                                        sep="\t",comment="#")

filters_with_min_q = tomtom_results.groupby('Query_ID').min()["q-value"]

tomtom_results = tomtom_results[["Target_ID", "Query_ID", "q-value"]]
tomtom_results = tomtom_results[tomtom_results["q-value"]<0.05]

cisbp_motifs = {}
with open("data/JASPAR2020_CORE_vertebrates_non-redundant_pfms_meme.txt") as f:
    for line in f:
        if "MOTIF" in line:
            motif = line.strip().split()[-1]
            name_m = line.strip().split()[-2]
            cisbp_motifs[name_m] = motif

filters = tomtom_results["Query_ID"].unique()
annotation = {}
for f in filters:
    t = tomtom_results[tomtom_results["Query_ID"] == f]
    target_id = t["Target_ID"]
    if len(target_id) > 5:
        target_id = target_id[:5]
    ann = "/".join([cisbp_motifs[i] for i in target_id.values])
    annotation[f] = ann

annotation = pd.Series(annotation)
```

Retrieving weights:

```python
weights = model.final.weight.detach().cpu().numpy()

filters = ["filter"+str(i) for i in range(num_cnns)]
for i in annotation.keys():
    filters[int(i.split("filter")[-1])] = annotation[i]

weight_df = pd.DataFrame(weights, index=target_labels, columns=filters)
# focusing on annotated filters only
weight_df = weight_df[[i for i in weight_df.columns if not i.startswith("filter")]]
```

Visualizing the weights:

```
plt.figure(figsize=(15, 10))
sns.clustermap(weight_df, cmap=sns.diverging_palette(145, 10, s=60, as_cmap=True),
               row_cluster=False, figsize=(30, 20), vmax=0.5, vmin=-0.5)
plt.show()
```

<img src="Figures\weights_TF.png" style="zoom:55%;" />

#### Individual unit importance

Visualizing one of the MYC/MAX filters (unit #67):

```python
unit_outputs = interpretation.get_explainn_unit_outputs(data_loader, model, device)

unit_importance = interpretation.get_specific_unit_importance(activations, model, unit_outputs, 67, target_labels)

filter_key = "filter"+str(67)
title = annotation[filter_key] if filter_key in annotation.index else filter_key
fig, ax = plt.subplots()
datas = [filt_dat for filt_dat in unit_importance]
ax.boxplot(datas, notch=True, patch_artist=True, boxprops=dict(facecolor="#228833", color="#228833"))
fig.set_size_inches(18.5, 10.5)
plt.title(title)
plt.ylabel("Unit importance")
plt.xticks(range(1, len(target_labels)+1), target_labels)
plt.xticks(rotation=90)
plt.show()
```

<img src="Figures\importance_TF.png" style="zoom:55%;" />

#### Execution time

ExplaiNN's execution times is subject to **1)** the size of the training dataset and **2)** the number of units used. For an approximation to execution times please refer to [Figure 2D](https://www.biorxiv.org/content/biorxiv/early/2022/05/25/2022.05.20.492818/F2.large.jpg).
