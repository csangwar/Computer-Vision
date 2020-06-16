# Coded by u301671
# Import all the necessary packages

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import argparse
import pickle
import h5py

# Read data from HDF5 database, split the file into train and test in the ratio of 75:25
input_file = "./catsanddogs_kaggle/datasets/kaggle_dogs_vs_cats/hdf5/features.hdf5"
db = h5py.File(input_file, "r")
print("Dataset shape:", db["features"].shape)
print("Dataset shape:", db["labels"].shape)
i = int(db["labels"].shape[0] * 0.75)

# Identify the C hyperparameter using GridsearchCv
print("[INFO] Tuning Parameters.....")
params = {"C":  [0.0001, 0.001, 0.01, 0.1, 1.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3,
                     n_jobs=-1, verbose=True)
model.fit(db["features"][:i], db["labels"][:i])
print("[INFO]: Best hyperparameters: {}".format(model.best_params_))

# Generate a classification report for the model
print("[INFO] evaluating.....")
preds = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:], preds,
      target_names=db["label_names"]))

# Compute the accuracy with extra precision
acc = accuracy_score(db["labels"][i:], preds)
print("[INFO] score: {}".format(acc))

# Serialize the model to a disk
print("[INFO] saving model....")
f = open("./catsanddogs_kaggle/dog_vs_cats/output/dogs_vs_cats.pickle", "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

# Close the database
db.close()


