import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from typing import Dict


def evaluate_node_classification(
    y_labels: list,
    z: np.array,
    test_size: float = 0.4,
    random_state: int = 345,
) -> Dict[str, float]:
    
    z_train, z_test, y_train, y_test = train_test_split(
        z,
        y_labels,
        test_size=test_size,
        stratify=y_labels, 
        random_state=random_state
    )
    
    scaler = StandardScaler().fit(z_train)
    z_train, z_test = scaler.transform(z_train), scaler.transform(z_test)
    
    clf = LogisticRegression(max_iter=250).fit(z_train, y_train)
    
    auc_train = roc_auc_score(
        y_true=y_train,
        y_score=clf.predict_proba(z_train),
        multi_class="ovr",
    )
    auc_test = roc_auc_score(
        y_true=y_test,
        y_score=clf.predict_proba(z_test),
        multi_class="ovr",
    )
    return {
        "train_auc": auc_train,
        "test_auc": auc_test
    }