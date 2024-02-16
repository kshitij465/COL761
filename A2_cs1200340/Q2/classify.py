"""
DO NOT ALTER THIS SCRIPT

Computes the macro averaged ROC-AUC scores by training an SVM over graph
features vectors.
"""
from argparse import ArgumentParser
from random import seed, shuffle

from numpy import unique
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

parser = ArgumentParser()
parser.add_argument("-g", "--graphs", type=str, required=True, 
                    help="Path to graph.txt, e.g., ../dataset/AIDS/graph.txt")
parser.add_argument("-f", "--features", type=str, required=True, 
                    help="Path to features_kerberosid.txt,\
                    e.g., ../dataset/AIDS/features_csz228001.txt")
parser.add_argument("-s", "--seed", type=int, default=0)

print("args laarha")
args = parser.parse_args()
print("args aagya")
SEED = args.seed
seed(SEED)

# * ----- Read the labels
with open(args.graphs, "r") as reader:
    labels = []
    while True:
        line = reader.readline()
        if not line:
            break
        line = line.strip()
        if line.startswith("#"):
            label = line.split()[-1]
            labels.append(int(label))

# * ----- Read the features
features_dict = {}
with open(args.features, "r") as reader:
    for line in reader.readlines():
        line = line.strip().split()
        id = int(line[0])
        fv = line[2:]
        fv = [int(i) for i in fv]
        features_dict[id] = fv
features = []

# * ----- Ensure that the features follow the graphs' order. 
for i in range(len(labels)):
    features.append(features_dict[i])
del features_dict

# * ----- Create train/test splits
indices = list(range(len(labels)))
train_indices, test_indices = train_test_split(
    indices,
    train_size=0.8,
    test_size=0.2,
    random_state=SEED,
    shuffle=True,
    stratify=labels
)
del indices

# Create a balanced training dataset.
idx_0 = [i for i in train_indices if labels[i] == 0]
idx_1 = [i for i in train_indices if labels[i] == 1]

if len(idx_0) < len(idx_1):
    idx_1 = idx_1[:len(idx_0)]
else:
    idx_0 = idx_0[:len(idx_1)]

train_indices = idx_0 + idx_1

shuffle(train_indices)

features_train = []
features_test = []
labels_train = []
labels_test = []
for i in train_indices:
    features_train.append(features[i])
    labels_train.append(labels[i])

for i in test_indices:
    features_test.append(features[i])
    labels_test.append(labels[i])

# print("----- Class distribution")
# print("train:", unique(labels_train, return_counts=True)[1] / len(labels_train))
# print("test:", unique(labels_test, return_counts=True)[1] / len(labels_test))
# print()

# * ----- Trian the model
model = SVC(kernel="rbf", probability=True, random_state=SEED)
model.fit(features_train, labels_train)

# Probability of the class with the greater label.
proba_train = model.predict_proba(features_train)[:, 1]
proba_test = model.predict_proba(features_test)[:, 1]

pred_train = model.predict(features_train)
pred_test = model.predict(features_test)
roc_auc_train = roc_auc_score(y_true=labels_train, y_score=proba_train, average="macro")
roc_auc_test = roc_auc_score(y_true=labels_test, y_score=proba_test, average="macro")

print(f"Train ROC_AUC: {roc_auc_train}")
print(f"Test ROC_AUC: {roc_auc_test}")
