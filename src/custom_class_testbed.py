# %%
import pandas as pd
import numpy as np
from lightgbm import Booster, LGBMClassifier
import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from custom_class import CustomLGBMClassifier

print(f"lightgbm version: {lgb.__version__}")

# %%

# Load the data
# Replace 'train.csv' and 'test.csv' with your actual file paths
train_data = pd.read_csv('src/data/train.csv')
test_data = pd.read_csv('src/data/test.csv')

# Assuming 'target' is your target variable
# Modify these according to your actual feature and target columns
X = train_data.drop('target', axis=1)
y = train_data['target']

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%

# Initialize LightGBM model
model = CustomLGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    num_leaves=31,
    random_state=42,
    # custom_param=123,
)

# Train the model
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],

)


# %%

# Make predictions on validation set
val_predictions = model.predict(X_val)

# Evaluate binary classification model
val_accuracy = accuracy_score(y_val, val_predictions)
val_roc_auc = roc_auc_score(y_val, val_predictions)

print()
print(f"Test Acc: {val_accuracy:.4f}")
print(f"Test AUC: {val_roc_auc:.4f}")



X_test = test_data.drop('target', axis=1)
y_test = test_data['target']

# Make predictions on test set
test_predictions = model.predict(X_test)

# Evaluate the model on test set
test_accuracy = accuracy_score(y_test, test_predictions)
test_roc_auc = roc_auc_score(y_test, test_predictions)


print()
print(f"Test Acc: {test_accuracy:.4f}")
print(f"Test AUC: {test_roc_auc:.4f}")

model1_proba = model.predict_proba(X_test)

print(model1_proba[:10])

# %%
# for p in dir(model):
#     if not p.startswith("__") and not callable(getattr(model, p)):
#         print(p, getattr(model, p))

# # %%
# # save model to pickle file
# with open('model.pkl', 'wb') as f:
#     pickle.dump(model, f)

# #save the model tree structure                  
# model.booster_.save_model('model.txt')
# print("Model saved as model.txt")

# # %%
# booster_model = Booster(
#     model_file='model.txt'
# )

# # restore model from pickle file
# with open('model.pkl', 'rb') as f:
#     orig_model = pickle.load(f)


# # create stub classier
# model2 = LGBMClassifier()

# # populae the stub classifier with the attributes of the original model
# for p in dir(orig_model):
#     if not p.startswith("__") and not callable(getattr(orig_model, p)):
#         print(p, getattr(orig_model, p))
#         try:
#             setattr(model2, p, getattr(orig_model, p))
#         except:
#             print(f">>>>>>Error setting {p}")


# # Load the booster model into the new classifier
# model2._Booster = booster_model
# model2.fitted_ = True

# X_test = test_data.drop('target', axis=1)
# y_test = test_data['target']

# # Make predictions on test set
# test_predictions = model2.predict(X_test)

# # Evaluate the model on test set
# test_accuracy = accuracy_score(y_test, test_predictions)
# test_roc_auc = roc_auc_score(y_test, test_predictions)

# print()
# print(f"Test Acc: {test_accuracy:.4f}")
# print(f"Test AUC: {test_roc_auc:.4f}")

# # Get probability predictions
# model2_proba = model2.predict_proba(X_test)

# # Check if the probability predictions are the same
# print(f"probabilities the same {np.allclose(model1_proba, model2_proba)}")


