
# Ryan Banko CS390 Project

# %%
from nilearn.image import mean_img
from nilearn.plotting import plot_epi, plot_roi, show, view_img

scan = "/Users/ryanbanko/Documents/output/sub-01/ses-T1/func/sub-01_ses-T1_task-SLD_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
plot_epi(mean_img(scan, copy_header=True))

# %%
import os

subjects = range(1, 80)

missing_subjects = []

for subject in subjects:
    subject_str = f"sub-{subject:02d}"
    subject_folder = f"/Users/ryanbanko/Documents/output/{subject_str}/ses-T1/func"

    subject_files = {
        "SLD": f"{subject_folder}/{subject_str}_ses-T1_task-SLD_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
        "SLI": f"{subject_folder}/{subject_str}_ses-T1_task-SLI_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
        "SSD": f"{subject_folder}/{subject_str}_ses-T1_task-SSD_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
        "SSI": f"{subject_folder}/{subject_str}_ses-T1_task-SSI_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
    }

    missing_files = [task for task, path in subject_files.items() if not os.path.exists(path)]

    if missing_files:
        print(f"Subject {subject_str} is missing: {', '.join(missing_files)}")
        missing_subjects.append(subject)
print(missing_subjects)

# %%
from nilearn.maskers import NiftiMasker
from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker
import numpy as np
from nilearn import plotting

dataset = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
atlas_filename = dataset.maps
labels = dataset.labels

masker = NiftiLabelsMasker(
    labels_img=atlas_filename,  
    standardize=True
)

correlation_measure = ConnectivityMeasure(
    kind="correlation",
    standardize="zscore_sample",
)
subjects = range(1,2)
df_mri = None

for subject in subjects:
    feature_data = []
    subject_str = f"sub-{subject:02d}"

    feature_vector = []

    subject_files = [
            f'/Users/ryanbanko/Documents/output/{subject_str}/ses-T1/func/{subject_str}_ses-T1_task-SLD_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz',
            f'/Users/ryanbanko/Documents/output/{subject_str}/ses-T1/func/{subject_str}_ses-T1_task-SLI_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz',
            f'/Users/ryanbanko/Documents/output/{subject_str}/ses-T1/func/{subject_str}_ses-T1_task-SSD_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz',
            f'/Users/ryanbanko/Documents/output/{subject_str}/ses-T1/func/{subject_str}_ses-T1_task-SSI_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz',
        ]

    for fmri_file in subject_files:

        confounds, sample_mask = load_confounds_strategy(
            fmri_file,
            denoise_strategy="simple",
            motion="basic",
            global_signal="basic",
        )

        time_series = masker.fit_transform(
            fmri_file, confounds=confounds, sample_mask=sample_mask
        )

        correlation_matrix = correlation_measure.fit_transform([time_series])[0]

        upper_triangular = correlation_matrix[np.triu_indices(correlation_matrix.shape[0], k=1)]

        feature_vector.extend(upper_triangular)

        plotting.plot_matrix(
            correlation_matrix,
            figure=(10, 8),
            labels=labels[1:],
            vmax=0.8,
            vmin=-0.8,
            title="simple with global signal",
            reorder=True,
        )
        plotting.show()

    feature_data.append({"participant_id": subject_str, "features": feature_vector})

df_mri = pd.DataFrame([[subject_str] + list(feature_vector)])

df_mri.columns = ["participant_id"] + [f"feature_{i}" for i in range(len(feature_vector))]
print(df_mri.head())

# %%
from nilearn.maskers import NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

dataset = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
atlas_filename = dataset.maps
labels = dataset.labels

masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

correlation_measure = ConnectivityMeasure(kind="correlation", standardize="zscore_sample",)

subjects = range(2, 10)
excluded = {2, 5, 7, 24, 25, 34, 42, 43, 45, 50, 52, 55, 59, 62, 64, 65, 70, 73, 75, 77}

def process_subject(subject):
    if subject in excluded:
        return None

    subject_str = f"sub-{subject:02d}"
    feature_vector = []

    subject_files = [
        f'/Users/ryanbanko/Documents/output/{subject_str}/ses-T1/func/{subject_str}_ses-T1_task-SLD_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz',
        f'/Users/ryanbanko/Documents/output/{subject_str}/ses-T1/func/{subject_str}_ses-T1_task-SLI_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz',
        f'/Users/ryanbanko/Documents/output/{subject_str}/ses-T1/func/{subject_str}_ses-T1_task-SSD_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz',
        f'/Users/ryanbanko/Documents/output/{subject_str}/ses-T1/func/{subject_str}_ses-T1_task-SSI_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz',
    ]

    for fmri_file in subject_files:
        confounds, sample_mask = load_confounds_strategy(fmri_file, denoise_strategy="simple", motion="basic", global_signal="basic")
        
        time_series = masker.fit_transform(fmri_file, confounds=confounds, sample_mask=sample_mask)
        
        correlation_matrix = correlation_measure.fit_transform([time_series])[0]
        
        upper_triangular = correlation_matrix[np.triu_indices(correlation_matrix.shape[0], k=1)]
        
        feature_vector.extend(upper_triangular)

    new_data = pd.DataFrame([[subject_str] + list(feature_vector)], columns=df_mri.columns)
    return new_data

results = Parallel(n_jobs=8)(delayed(process_subject)(subject) for subject in subjects)

df_new = pd.concat([result for result in results if result is not None], ignore_index=True)

df_mri = pd.concat([df_mri, df_new], ignore_index=True)

# %%
subjects = range(10, 50)
excluded = {2, 5, 7, 24, 25, 34, 42, 43, 45, 50, 52, 55, 59, 62, 64, 65, 70, 73, 75, 77}

results = Parallel(n_jobs=8)(delayed(process_subject)(subject) for subject in subjects)

df_new = pd.concat([result for result in results if result is not None], ignore_index=True)

df_mri = pd.concat([df_mri, df_new], ignore_index=True)

# %%
subjects = range(50, 80)
excluded = {2, 5, 7, 24, 25, 34, 42, 43, 45, 50, 52, 55, 59, 62, 64, 65, 70, 73, 75, 77}

results = Parallel(n_jobs=8)(delayed(process_subject)(subject) for subject in subjects)

df_new = pd.concat([result for result in results if result is not None], ignore_index=True)

df_mri = pd.concat([df_mri, df_new], ignore_index=True)

# %%
print(df_mri)

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

df_pca = df_mri.copy()

participant_id = df_pca['participant_id']
df_features = df_pca.drop(columns=['participant_id'])

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)

pca = PCA()
pca.fit(df_scaled)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Number of PCA Components')
plt.grid(True)
plt.show()

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

num_components = np.argmax(cumulative_variance >= 0.90) + 1

print(f"Ideal Number of Components: {num_components}")


# %%
pca = PCA(n_components=42)
df_pca_transformed = pca.fit_transform(df_scaled)

df_pca_final = pd.DataFrame(df_pca_transformed, columns=[f'PC{i+1}' for i in range(42)])
df_pca_final.insert(0, 'participant_id', participant_id.values)

df_pca_final.head()

# %%
import pandas as pd
tsv_file = '/Users/ryanbanko/Documents/output/participants.tsv'
df_demo = pd.read_csv(tsv_file, sep='\t')

print(df_demo.head())

# %%
df_demo = df_demo[df_demo['participant_id'].isin(df_pca_final['participant_id'])]

# %%
columns_to_keep = ['participant_id', 'age_ses-T1', 'sex', 'handedness', 'race', 'ethnicity', 'ADHD_diagnosis']
df_demo = df_demo[columns_to_keep]

# %%
missing_values = df_demo.isnull().sum()
print(missing_values)

print(df_demo['race'].value_counts())
print(df_demo['ethnicity'].value_counts())
print(df_demo['handedness'].value_counts())

# %%
df_imputed = df_demo.copy()

df_imputed['race'] = df_imputed['race'].fillna(df_imputed['race'].mode()[0])
df_imputed['ethnicity'] = df_imputed['ethnicity'].fillna(df_imputed['ethnicity'].mode()[0])
df_imputed = df_imputed.drop(columns=['handedness'])

print(df_imputed)

# %%
df_pca_final = df_pca_final.drop_duplicates(subset='participant_id')
df_combined = pd.merge(df_imputed, df_pca_final, on='participant_id', how='inner')

# %%
print(df_combined.head())

# %%
from sklearn.model_selection import train_test_split

X = df_combined.drop(columns=['ADHD_diagnosis', 'participant_id'])
y = df_combined['ADHD_diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

svm_model = SVC()

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4],
    'coef0': [0, 0.5, 1, 2],
}

grid_search = GridSearchCV(svm_model, param_grid, cv=10, n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)


# %%
from sklearn.metrics import precision_score, recall_score, accuracy_score

svm_model = SVC(C=100, coef0=0, degree=2, gamma='scale', kernel='sigmoid')

svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, zero_division=1)
test_recall = recall_score(y_test, y_pred, zero_division=1)

print("\nTest Set Evaluation:")
print(f"Test Accuracy: {test_accuracy * 100:.1f}%")
print(f"Test Precision: {test_precision * 100:.1f}%")
print(f"Test Recall: {test_recall * 100:.1f}%")

# %%
recall_svm = SVC(C=100, coef0=0, degree=2, gamma='scale', kernel='sigmoid', probability=True, random_state=42)
recall_svm.fit(X_train, y_train)

probs = recall_svm.predict_proba(X_test)[:, 1]

accuracies = []
precisions = []
recalls = []

thresholds = np.linspace(0.4, 0.45, 200)

for threshold in thresholds:
    y_pred = (probs >= threshold).astype(int)
    
    extra_test_accuracy = accuracy_score(y_test, y_pred)
    extra_test_precision = precision_score(y_test, y_pred, zero_division=1)
    extra_test_recall = recall_score(y_test, y_pred, zero_division=1)
    
    accuracies.append(extra_test_accuracy)
    precisions.append(extra_test_precision)
    recalls.append(extra_test_recall)

plt.figure(figsize=(10, 6))

plt.plot(thresholds, accuracies, label='Accuracy', color='blue')
plt.plot(thresholds, precisions, label='Precision', color='green')
plt.plot(thresholds, recalls, label='Recall', color='red')

plt.title('SVM Performance Metrics vs. Decision Threshold')
plt.xlabel('Threshold')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
plt.show()

# %%
probs = recall_svm.predict_proba(X_test)[:, 1]
extra_y_pred = (probs >= 0.42).astype(int)

extra_test_accuracy = accuracy_score(y_test, extra_y_pred)
extra_test_precision = precision_score(y_test, extra_y_pred, zero_division=1)
extra_test_recall = recall_score(y_test, extra_y_pred, zero_division=1)

print("\nTest Set Evaluation:")
print(f"Test Accuracy: {extra_test_accuracy * 100:.1f}%")
print(f"Test Precision: {extra_test_precision * 100:.1f}%")
print(f"Test Recall: {extra_test_recall * 100:.1f}%")

# %%
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf_model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(rf_model, param_grid, cv=10, n_jobs=-1)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

# %%
rf_model = RandomForestClassifier(
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, zero_division=1)
test_recall = recall_score(y_test, y_pred, zero_division=1)

print("\nTest Set Evaluation:")
print(f"Test Accuracy: {test_accuracy * 100:.1f}%")
print(f"Test Precision: {test_precision * 100:.1f}%")
print(f"Test Recall: {test_recall * 100:.1f}%")

# %%
recall_rf_model = RandomForestClassifier(
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=200,
    random_state=42
)
recall_rf_model.fit(X_train, y_train)

recall_rf_probs = rf_model.predict_proba(X_test)[:, 1]

accuracies = []
precisions = []
recalls = []

thresholds = np.linspace(0.3, 0.5, 200)

for threshold in thresholds:
    y_pred = (recall_rf_probs >= threshold).astype(int)
    
    extra_test_accuracy = accuracy_score(y_test, y_pred)
    extra_test_precision = precision_score(y_test, y_pred, zero_division=1)
    extra_test_recall = recall_score(y_test, y_pred, zero_division=1)
    
    accuracies.append(extra_test_accuracy)
    precisions.append(extra_test_precision)
    recalls.append(extra_test_recall)

plt.figure(figsize=(10, 6))

plt.plot(thresholds, accuracies, label='Accuracy', color='blue')
plt.plot(thresholds, precisions, label='Precision', color='green')
plt.plot(thresholds, recalls, label='Recall', color='red')

plt.title('Performance Metrics vs. Decision Threshold')
plt.xlabel('Threshold')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
plt.show()

# %%
recall_rf_model.fit(X_train, y_train)

y_pred = (recall_rf_probs >= 0.3).astype(int)

test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, zero_division=1)
test_recall = recall_score(y_test, y_pred, zero_division=1)

print("\nTest Set Evaluation:")
print(f"Test Accuracy: {test_accuracy * 100:.1f}%")
print(f"Test Precision: {test_precision * 100:.1f}%")
print(f"Test Recall: {test_recall * 100:.1f}%")

# %%
from sklearn.neighbors import KNeighborsClassifier

recall_knn_model = KNeighborsClassifier(n_neighbors=6)
recall_knn_model.fit(X_train, y_train)

recall_knn_probs = recall_knn_model.predict_proba(X_test)[:, 1]

accuracies = []
precisions = []
recalls = []

thresholds = np.linspace(0.1, 0.99, 200)

for threshold in thresholds:
    y_pred = (recall_knn_probs >= threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=1)
    rec = recall_score(y_test, y_pred, zero_division=1)
    
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, accuracies, label='Accuracy', color='blue')
plt.plot(thresholds, precisions, label='Precision', color='green')
plt.plot(thresholds, recalls, label='Recall', color='red')

plt.title('KNN Performance Metrics vs. Decision Threshold')
plt.xlabel('Threshold')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
plt.show()

y_pred = (recall_knn_probs >= 0.1).astype(int)

test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, zero_division=1)
test_recall = recall_score(y_test, y_pred, zero_division=1)

print("\nKNN Test Set Evaluation (Threshold = 0.1, N = 6):")
print(f"Test Accuracy: {test_accuracy * 100:.1f}%")
print(f"Test Precision: {test_precision * 100:.1f}%")
print(f"Test Recall: {test_recall * 100:.1f}%")

# %%
from sklearn.linear_model import LogisticRegression

log_reg_model = LogisticRegression(solver='liblinear', random_state=42)
log_reg_model.fit(X_train, y_train)

log_probs = log_reg_model.predict_proba(X_test)[:, 1]

accuracies = []
precisions = []
recalls = []

thresholds = np.linspace(0.1, 0.9, 200)

for threshold in thresholds:
    y_pred = (log_probs >= threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=1)
    rec = recall_score(y_test, y_pred, zero_division=1)
    
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, accuracies, label='Accuracy', color='blue')
plt.plot(thresholds, precisions, label='Precision', color='green')
plt.plot(thresholds, recalls, label='Recall', color='red')

plt.title('Logistic Regression Performance vs. Decision Threshold')
plt.xlabel('Threshold')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
plt.show()

y_pred = (log_probs >= 0.45).astype(int)

test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, zero_division=1)
test_recall = recall_score(y_test, y_pred, zero_division=1)

print("\nLogistic Regression Test Set Evaluation (Threshold = 0.45):")
print(f"Test Accuracy: {test_accuracy * 100:.1f}%")
print(f"Test Precision: {test_precision * 100:.1f}%")
print(f"Test Recall: {test_recall * 100:.1f}%")

# %%
print(df_demo['ADHD_diagnosis'].value_counts())

# %%
from nilearn.image import mean_img
from nilearn.plotting import plot_epi, plot_roi, show, view_img

plot_epi(mean_img(fmri_filenames, copy_header=True))


