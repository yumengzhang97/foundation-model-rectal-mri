from sklearn.linear_model import LogisticRegression
import numpy as np
import sklearn.metrics as skm
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import pandas as pd


base_dir = r'/home/chaimeleon/persistent-home/classificationmodel_original_sagittal/total_work/normalized2/embedding_LR/'
csv_location = '/home/chaimeleon/persistent-home/Complete_set/Experiments/Results/UMedPT_LR'
csv_prefix = 'sagittal_original'


def calculate_metrics(y_true, y_pred):
    y_pred_binary = (y_pred > 0.5)
    
    auc = skm.roc_auc_score(y_true, y_pred)
    balanced_acc = skm.balanced_accuracy_score(y_true, y_pred_binary)
    sensitivity = skm.recall_score(y_true, y_pred_binary)
    specificity = skm.recall_score(y_true, y_pred_binary, pos_label=0)
    f1 = skm.f1_score(y_true, y_pred_binary)
    
    fScore = 0.4 * auc + 0.2 * balanced_acc + 0.2 * sensitivity + 0.2 * specificity 
    return {
        'fScore': fScore,
        'auc': auc,
        'balanced_acc': balanced_acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1
    }


# Load saved embeddings and labels
X_train_val = np.load(os.path.join(base_dir, "train_val_embeddings.npy"))  # Features for training
X_test = np.load(os.path.join(base_dir, "test_embeddings.npy"))  # Features for testing

y_train_val_evi = np.load(os.path.join(base_dir, "train_val_evi_labels.npy"))  # EVI labels
y_train_val_mfi = np.load(os.path.join(base_dir, "train_val_mfi_labels.npy"))  # MFI labels
y_test_evi = np.load(os.path.join(base_dir, "test_evi_labels.npy"))  # EVI test labels
y_test_mfi = np.load(os.path.join(base_dir, "test_mfi_labels.npy"))  # MFI test labels


print("X_train_val shape:", X_train_val.shape)
print("X_test shape:", X_test.shape)
print("y_train_val_evi shape:", y_train_val_evi.shape)
print("y_train_val_mfi shape:", y_train_val_mfi.shape)
print("y_test_evi shape:", y_test_evi.shape)
print("y_test_mfi shape:", y_test_mfi.shape)

# ======== 2) Build a Pipeline ========
pipe = Pipeline([
    ('scaler', StandardScaler()),         # Step 1: standardize features
    ('pca', PCA()),                       # Step 2: PCA dimensionality reduction
    ('clf', LogisticRegression())         # Step 3: Logistic Regression
])

# ======== 3) Define the parameter search space ========
param_grid = {
    # Candidate values for the number of PCA components
    'pca__n_components': [50, 100, 150, 200],
    
    # Hyperparameters for Logistic Regression
    'clf__C': [0.001, 0.01, 0.1, 1, 10], # overfitting or underfitting
    'clf__solver': ['liblinear'], #optimizer
    'clf__penalty': ['l2'],  # If you want to try 'l1', you need solver='liblinear' or 'saga'
    'clf__max_iter': [1000, 2000]
    # You can also add 'clf__class_weight': ['balanced', None], etc.
}

# param_grid = {
#     # Candidate values for the number of PCA components
#     'pca__n_components': [50, 100, 150, 200],
    
#     # Hyperparameters for Logistic Regression
#     'clf__C': [0.001, 0.01, 0.1, 1, 10], # overfitting or underfitting
#     'clf__solver': ['liblinear', 'saga'], #optimizer
#     'clf__penalty': ['l1'],  # If you want to try 'l1', you need solver='liblinear' or 'saga'
#     'clf__max_iter': [1000]
#     # You can also add 'clf__class_weight': ['balanced', None], etc.
# }


# Create a StratifiedKFold to ensure stratification
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_evi = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv_splitter,     # Use the StratifiedKFold splitter
    n_jobs=-1
)

grid_mfi = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv_splitter,             # 5-fold cross-validation
    scoring='roc_auc',  # Use AUC as the scoring metric
    n_jobs=-1           # Parallelize across all available CPU cores
)

grid_evi.fit(X_train_val, y_train_val_evi)
grid_mfi.fit(X_train_val, y_train_val_mfi)

best_model1 = grid_evi.best_estimator_
print("Best params of evi:", grid_evi.best_params_)
print("Best CV AUC of evi:", grid_evi.best_score_)

best_model2 = grid_mfi.best_estimator_
print("Best params of mfi:", grid_mfi.best_params_)
print("Best CV AUC of mfi:", grid_mfi.best_score_)

# ======== 5) Evaluate on the test set ========
y_pred_prob1 = best_model1.predict_proba(X_test)[:, 1]  # Probability for the positive class
metrics_evi = calculate_metrics(y_test_evi, y_pred_prob1)

y_pred_prob2 = best_model2.predict_proba(X_test)[:, 1]  # Probability for the positive class
metrics_mfi = calculate_metrics(y_test_mfi, y_pred_prob2)

#save to csv
predictions_df = pd.DataFrame({
    'y_test_evi': y_test_evi,
    'y_pred_prob_evi': y_pred_prob1,
    'y_test_mfi': y_test_mfi,
    'y_pred_prob_mfi': y_pred_prob2
})
predictions_df.to_csv(os.path.join(csv_location,f'{csv_prefix}.csv'), index=False)

print(f'fscore_evi: {metrics_evi["fScore"]:.4f}, auc_evi: {metrics_evi["auc"]:.4f}, balanced_acc: {metrics_evi["balanced_acc"]:.4f}, sensitivity_evi: {metrics_evi["sensitivity"]:.4f}, specificity_evi: {metrics_evi["specificity"]:.4f}, f1: {metrics_evi["f1"]:.4f}')
print(f'fscore_mfi: {metrics_mfi["fScore"]:.4f}, auc_mfi: {metrics_mfi["auc"]:.4f}, balanced_acc: {metrics_mfi["balanced_acc"]:.4f}, sensitivity_mfi: {metrics_mfi["sensitivity"]:.4f}, specificity_mfi: {metrics_mfi["specificity"]:.4f}, f1: {metrics_mfi["f1"]:.4f}')


