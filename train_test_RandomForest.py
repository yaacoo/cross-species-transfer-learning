import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from os import path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from adapt.instance_based import TrAdaBoost

# Load the list of tissues:
with open('../data/list_of_tissues.pkl', 'rb') as f:
    tissues = pkl.load(f)

# Iterate over the tissues:
for current_tissue in tissues:
    print(current_tissue)
    # Check if a file does not exists:
    if not path.exists("../data/"+current_tissue+"_MCA.pkl"):
        print("Tissue was filtered out in pre-processing")
        continue
    # Load the pickle objects of the tissues:
    with open("../data/"+current_tissue+"_MCA.pkl", "rb") as f:
        MCA_current_tissue = pkl.load(f) 
    with open("../data/"+current_tissue+"_HCL.pkl", "rb") as f:
        HCL_current_tissue = pkl.load(f)
    with open("../data/"+current_tissue+"_MCA_y.pkl", "rb") as f:
        MCA_current_tissue_y = pkl.load(f)
    with open("../data/"+current_tissue +"_HCL_y.pkl", "rb") as f:
        HCL_current_tissue_y = pkl.load(f)

    # Verify that there are no missing values in both datasets:
    if MCA_current_tissue.isnull().sum().sum() != HCL_current_tissue.isnull().sum().sum() != 0 :
        print("There are missing values in the data")
    else:
        print("There are no missing values in the data")
    # List the genes are not expressed (have a column sum smaller than 50 across all cells)
    genesNotExpressed=list(set(list(MCA_current_tissue.columns[MCA_current_tissue.sum(axis=0) < 50]) + 
                                list(HCL_current_tissue.columns[HCL_current_tissue.sum(axis=0) < 50])))
    # Remove the genes that are not expressed:
    MCA_current_tissue = MCA_current_tissue.drop(genesNotExpressed, axis=1)
    HCL_current_tissue = HCL_current_tissue.drop(genesNotExpressed, axis=1)

    # Verify that the columns are in the same order:
    if (all (MCA_current_tissue.columns == HCL_current_tissue.columns )):
        print("The columns are in the same order")
    else:
        print("The columns are not in the same order")
    #Verify that the dataset and labels have the same number of cells:
    (MCA_current_tissue.shape[0] == len(MCA_current_tissue_y)) and (
        HCL_current_tissue.shape[0] == len(HCL_current_tissue_y))

    # Train-test split for each atlas:
    MCA_X_train, MCA_X_test, MCA_y_train, MCA_y_test = train_test_split(MCA_current_tissue, MCA_current_tissue_y, random_state=4, train_size=0.8)
    HCL_X_train, HCL_X_test, HCL_y_train, HCL_y_test = train_test_split(HCL_current_tissue, HCL_current_tissue_y, random_state=4, train_size=0.8)

    # Train target on target as a "gold standard" for comparison, as the maximal prediction performance that can be achieved:
    # Set current classifier:
    current_algorithm="RandomForest"

    # Train and fit model without a standard scaler since it scaling is not recommended for the sparse single-cell data:
    HCL_trained_pipe = make_pipeline(RandomForestClassifier(max_depth=50,random_state=1))
    HCL_trained_pipe.fit(HCL_X_train, HCL_y_train)

    # Predict target on target
    report_target_on_target=pd.DataFrame(classification_report(HCL_y_test, HCL_trained_pipe.predict(HCL_X_test), zero_division=0, output_dict=True)).transpose()
    report_target_on_target["Tissue"]=current_tissue
    report_target_on_target["Algorithm"]=current_algorithm
    report_target_on_target["Type"]="Target on target"

    # Train and fit the source domain (MCA) without transfer learning 
    MCA_trained_pipe = make_pipeline(RandomForestClassifier(max_depth=50,random_state=1))
    MCA_trained_pipe.fit(MCA_X_train, MCA_y_train)

    # Predict MCA on HCL
    report_source_on_target=pd.DataFrame(classification_report(HCL_y_test, MCA_trained_pipe.predict(HCL_X_test), zero_division=0, output_dict=True)).transpose()
    report_source_on_target["Tissue"]=current_tissue
    report_source_on_target["Algorithm"]=current_algorithm
    report_source_on_target["Type"]="Source on target"

    # Perform transfer learning with TrAdaBoost (instance based transfer learning)
    # Set a size of target data "available" to adjust the model as 10% of the souce training set size:
    targetAvailable=round( 0.1 * len(MCA_y_train) )

    # Train the on source (MCA), and use instance based transfer learning using  10% of the source training set size:
    MCA_trained_TrAdaBoost = TrAdaBoost(RandomForestClassifier(max_depth=50,random_state=1), n_estimators=10, Xt=HCL_X_train[:targetAvailable], yt=HCL_y_train[:targetAvailable], random_state=1)
    # To make the comparison "fair", we will reduce the number of training samples from the source by 10% as well since in theory we add 10% of the target data:
    MCA_trained_TrAdaBoost.fit(MCA_X_train[:(len(MCA_y_train)-targetAvailable)], MCA_y_train[:(len(MCA_y_train)-targetAvailable)])

    report_wTL=pd.DataFrame(classification_report(HCL_y_test, MCA_trained_TrAdaBoost.predict(HCL_X_test), zero_division=0, output_dict=True)).transpose()
    report_wTL["Tissue"]=current_tissue
    report_wTL["Algorithm"]=current_algorithm
    report_wTL["Type"]="With transfer learning"

    # Concatenate the reports:
    report=pd.concat([report_target_on_target, report_source_on_target, report_wTL])
    # Save the report:
    report.to_csv("../results/"+current_tissue+"_"+current_algorithm+"_report.csv")
