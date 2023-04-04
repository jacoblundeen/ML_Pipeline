"""
605.649 Introduction to Machine Learning
Dr. Donnelly
Programming Project #01
20220910
Jacob M. Lundeen

The purpose of this assignment is to give you an introduction to the basics steps data processing while
developing a small toolkit of functions to support your tasks in future programming assignments. In this
project you will pre-process six datasets and build the skeleton of a basic machine learning pipeline that you
will reuse in the later projects.

In this course, you will use six datasets that you will download from the UCI Machine Learning Repository1.
Details about these datasets can be found on the last page of these instructions. Although these
datasets all differ, you will build a general set of tools that can handle any dataset. Test your functions well;
your future self will thank you.
"""

import pandas as pd
import numpy as np
from statistics import mean


# Function to read in the data set. For those data sets that do not have header rows, this will accept a tuple of
# column names. It is defaulted to fill in NA values with '?'.
def read_data(data, names=(), fillna=True):
    if not names:
        return pd.read_csv(data)
    if not fillna:
        return pd.read_csv(data, names=names)
    else:
        return pd.read_csv(data, names=names, na_values='?')


# The missing_values() function takes in the data set and the column name and then fills in the missing values of the
# column with the column mean. It does this 'inplace' so there is no copy of the data set made.
def missing_values(data, column_name):
    data[column_name].fillna(value=data[column_name].mean(), inplace=True)


# The cat_data() function handles ordinal and nominal categorical data. For the ordinal data, we use a mapper that maps
# the ordinal data to integers so they can be utilized in the ML algorithms. For nominal data, Pandas get_dummies()
# function is used.
def cat_data(data, var_name='', ordinal=False, data_name=''):
    if ordinal:
        if data_name == 'cars':
            buy_main_mapper = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
            door_mapper = {'2': 2, '3': 3, '4': 4, '5more': 5}
            per_mapper = {'2': 2, '4': 4, 'more': 5}
            lug_mapper = {'small': 0, 'med': 1, 'big': 2}
            saf_mapper = {'low': 0, 'med': 1, 'high': 2}
            class_mapper = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
            mapper = [buy_main_mapper, buy_main_mapper, door_mapper, per_mapper, lug_mapper, saf_mapper, class_mapper]
            count = 0
            for col in data.columns:
                data[col] = data[col].replace(mapper[count])
                count += 1
            return data
        elif data_name == 'forest':
            month_mapper = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
                            'oct': 10, 'nov': 11, 'dec': 12}
            day_mapper = {'sun': 1, 'mon': 2, 'tue': 3, 'wed': 4, 'thu': 5, 'fri': 6, 'sat': 7}
            data.month = data.month.replace(month_mapper)
            data.day = data.day.replace(day_mapper)
            return data
        elif data_name == 'cancer':
            class_mapper = {2: 0, 4: 1}
            data[var_name] = data[var_name].replace(class_mapper)
            return data
    else:
        return pd.get_dummies(data, columns=var_name, prefix=var_name)


# The discrete() function transforms real-valued data into discretized values. This function provides the ability to do
# both equal width (pd.cut()) and equal frequency (pd.qcut()). The function also provides for discretizing a single
# feature or the entire data set.
def discrete(data, equal_width=True, num_bin=20, feature=""):
    if equal_width:
        if not feature:
            for col in data.columns:
                data[col] = pd.cut(x=data[col], bins=num_bin)
            return data
        else:
            data[feature] = pd.cut(x=data[feature], bins=num_bin)
            return data
    else:
        if not feature:
            for col in data.columns:
                data[col] = pd.qcut(x=data[col], q=num_bin, duplicates='drop')
            return data
        else:
            data[feature] = pd.qcut(x=data[feature], q=num_bin)
            return data


# The standardization() function performs z-score standardization on a given train and test set. The function
# will standardize either an individual feature or the entire data set. If the standard deviation of a variable is 0,
# then the variable is constant and adds no information to the regression, so it can be dropped from the data set.
def standardization(train, test=pd.DataFrame(), feature=''):
    if test.empty:
        for col in train.columns:
            if train[col].std() == 0:
                train.drop(col, axis=1, inplace=True)
            else:
                train[col] = (train[col] - train[col].mean()) / train[col].std()
        return train
    elif not feature:
        for col in train.columns:
            if train[col].std() == 0:
                train.drop(col, axis=1, inplace=True)
                test.drop(col, axis=1, inplace=True)
                file2.write("\nThe " + str(col) + "feature was dropped for having a STD of 0.")
            else:
                test[col] = (test[col] - train[col].mean()) / train[col].std()
                train[col] = (train[col] - train[col].mean()) / train[col].std()
        return train, test
    else:
        test[feature] = (test[feature] - train[feature].mean()) / train[feature].std()
        train[feature] = (train[feature] - train[feature].mean()) / train[feature].std()
        return train, test


# The cross_val() function performs the cross validation of the data set. For classification, the data is stratified
# (to include the validation set).
def cross_val(data, class_var, pred_type, k=5, validation=False):
    if pred_type == 'classification':
        val_split, data_splits = class_split(data, k, class_var, validation)
    elif pred_type == 'regression':
        val_split, data_splits = reg_split(data, k, validation)
    else:
        print("Please provide a prediction choice.")
        return
    # Here is where the program rotates through the k-folds of the data. The For loop identifies each test set, removes
    # it from the folds, and uses the rest of the folds for the training set. Regression data (both training and test)
    # are standardized. The results, validation data set, and test set are then returned.
    results = []
    for item in range(k):
        train = data_splits.copy()
        test = data_splits[item]
        del train[item]
        train = pd.concat(train, sort=False)
        if k == 10:
            if item == 0:
                file3.write("For the abalone data set with k=10:")
            file3.write("\nThe test fold size is: " + str(len(test)))
            file3.write("\nThe train fold size is: " + str(len(train)))
        if pred_type == 'regression':
            train, test = standardization(train.copy(), test.copy())
            if item == 0:
                file2.write("The normalized test data:\n" + str(test))
                file2.write("\nThe normalized train data:\n" + str(train))
        results.append(np.full(len(test), null_model(train.copy(), class_var, pred_type=pred_type)))
        if k == 10 and item == 9:
            file3.write("\nThe results are: " + str(results))
    if pred_type == 'regression':
        norm_data = standardization(pd.concat(data_splits))
        return val_split, norm_data, [item for subresult in results for item in subresult]
    else:
        return val_split, pd.concat(data_splits), [item for subresult in results for item in subresult]


# The class_split() function handles splitting and stratifying classification data. Returns validation set and
# data_splits.
def class_split(data, k, class_var, validation=False):
    # Group the data set by class variable using pd.groupby()
    grouped = data.groupby([class_var])
    grouped_l = []
    grouped_val = []
    grouped_dat = []
    data_splits = []
    # Create stratified validation set using np.split(). 20% from each group is appended to the validation set, the rest
    # will be used for the k-folds.
    for name, group in grouped:
        val, dat = np.split(group, [int(0.2 * len(group))])
        grouped_val.append(val)
        grouped_dat.append(dat)
    # Split the groups into k folds
    for i in range(len(grouped_dat)):
        grouped_l.append(np.array_split(grouped_dat[i], k))
    # Reset indices of the folds
    for item in range(len(grouped_l)):
        for jitem in range(len(grouped_l[item])):
            grouped_l[item][jitem].reset_index(inplace=True, drop=True)
    # Combine folds from each group to create stratified folds
    for item in range(k):
        tempo = grouped_l[0][item]
        for jitem in range(1, len(grouped_l)):
            tempo = pd.concat([tempo, grouped_l[jitem][item]], ignore_index=True)
        data_splits.append(tempo)
    grouped_val = pd.concat(grouped_val)
    return grouped_val, data_splits


# The reg_split() function creates the k-folds for regression data.
def reg_split(data, k, validation=False):
    if validation:
        val_fold, data_fold = np.split(data, [int(.2 * len(data))])
        data_fold = np.array_split(data_fold, k)
        return val_fold, data_fold
    else:
        data_fold = np.array_split(data, k)
        val_fold = 0
        return val_fold, data_fold


def k2_cross(data_splits, k, class_var, pred_type):
    results = []
    data = pd.concat(data_splits, sort=False)
    dfs = np.array_split(data, 2)
    if pred_type == 'regression':
        dfs[0], dfs[1] = standardization(dfs[0], dfs[1], class_var)
    results.append(np.full(len(dfs[0]), null_model(dfs[0], class_var, pred_type=pred_type)))
    results.append(np.full(len(dfs[1]), null_model(dfs[1], class_var, pred_type=pred_type)))
    return results


# The eval_metrics() function returns the classification or regression metrics.
def eval_metrics(true, predicted, eval_type='regression'):
    # For regression, we create the correlation matrix and then calculate the R2, Person's Correlation, and MSE.
    if eval_type == 'regression':
        corr_matrix = np.corrcoef(true, predicted)
        r2 = round(corr_matrix[0, 1] ** 2, 10)
        persons = round(corr_matrix[0, 1], 10)
        mse = round(np.square(np.subtract(true, predicted)).mean(), 10)
        file1.write("\nThe R\u00b2 Coefficient is " + str(r2))
        file1.write("\nPerson's Correlation is " + str(persons))
        file1.write("\nThe Mean Squared Error is " + str(mse))
    # For classification, we calculate Precision, Recall, and F1 scores.
    elif eval_type == 'classification':
        precision = []
        recall = []
        f_1 = []
        count = 0
        for label in np.unique(true):
            true_pos = np.sum((true == label) & (predicted == label))
            false_pos = np.sum((true != label) & (predicted == label))
            false_neg = np.sum((true == label) & (predicted != label))
            if true_pos & false_pos == 0:
                precision.append(0)
            else:
                precision.append(true_pos / (true_pos + false_pos))
            if true_pos + false_neg == 0:
                recall.append(0)
            else:
                recall.append(true_pos / (true_pos + false_neg))
            if precision[count] + recall[count] == 0:
                f_1.append(0)
            else:
                f_1.append(2 * (precision[count] * recall[count]) / (precision[count] + recall[count]))
            count += 1
        file1.write(("\nThe Precision is " + str(round(np.mean(precision), 4))))
        file1.write("\nThe Recall is " + str(round(np.mean(recall), 4)))
        file1.write("\nThe F1 score is " + str(round(np.mean(f_1), 4)))
    else:
        print("Please choose a prediction method.")
        return


# The null_model() function will return the mean value of the target variable or the most common class for
# classification.
def null_model(train, class_var, pred_type="regression"):
    if pred_type == 'regression':
        return np.mean(train[class_var])
    elif pred_type == 'classification':
        unique_elem, count = np.unique(train[class_var], return_counts=True)
        return unique_elem[count == count.max()]
    else:
        print("Please choose a prediction method.")
        return


if __name__ == '__main__':
    # This first section read in the six data sets. 5 of the 6 data sets must have their column names hardcoded
    # (the forest data set is the only one that doesn't). A tuple is created with the column names and then passed
    # to the read_data() function along with the name of the data set. The house data is the only data set that does not
    # need missing values changed to '?'.
    ab_names = ('sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight',
                'shell_weight', 'rings')
    abalone = read_data('abalone.data', ab_names)

    cancer_names = ('code_num', 'clump_thick', 'unif_size', 'unif_shape', 'adhesion', 'epithelial_size', 'bare_nuclei',
                    'bland_chromatin', 'norm_nucleoli', 'mitosis', 'class')
    cancer = read_data('breast-cancer-wisconsin.data', cancer_names)

    car_names = ('buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class')
    cars = read_data('car.data', car_names)

    forest = read_data('forestfires.csv')

    house_names = ('class', 'infants', 'water_sharing', 'adoption_budget', 'physician_fee', 'salvador_aid',
                   'religious_schools', 'andit_sat_ban', 'aid_nic_contras', 'mx_missile', 'immigration',
                   'synfuels_cutback', 'edu_spending', 'supderfund_sve', 'crime', 'duty_free', 'export_admin_africa')
    house = read_data('house-votes-84.data', house_names, fillna=False)

    machine_names = ('vendor', 'model', 'myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax', 'prp', 'erp')
    machine = read_data('machine.data', machine_names)

    # Next we handle the missing values. The cancer data set is the only one that supposedly has any missing values, but
    # we check all 6 data sets to make sure. Once confirmed that only the cancer data set missing values, we impute the
    # missing values with the column mean.
    print(pd.isna(abalone).sum())
    print(pd.isna(cancer).sum())
    print(pd.isna(cars).sum())
    print(pd.isna(forest).sum())
    print(pd.isna(house).sum())
    print(pd.isna(machine).sum())
    missing_values(cancer, 'bare_nuclei')
    print(pd.isna(cancer).sum())

    # Here we handle categorical data
    abalone = cat_data(abalone, [ab_names[0]])
    cancer = cat_data(cancer, var_name='class', data_name='cancer', ordinal=True)
    cars = cat_data(cars, var_name=car_names, data_name='cars', ordinal=True)
    forest = cat_data(forest, data_name='forest', ordinal=True)
    house = cat_data(house, var_name=list(house_names))
    machine = cat_data(machine, var_name=list(machine_names[0:2]))

    print(abalone.columns)
    print(cancer.columns)
    print(cars.columns)
    print(forest.columns)
    print(house.columns)
    print(machine.columns)

    # Here we present an example of discretization
    temp = discrete(abalone.copy())
    print(temp)

    # Here we print the results of the Null Model using 5-fold CV
    file1 = open("Results.txt", "w")
    file2 = open("Standard.txt", "w")
    file3 = open("K-Fold.txt", "w")
    file1.write("Below are the metrics for classification of sex_I for the abalone data set.")
    val_data, test_data, prediction = cross_val(abalone, class_var='sex_I', k=10, validation=False,
                                                pred_type="classification")
    eval_metrics(test_data['sex_I'].tolist(), prediction, 'classification')

    file1.write("\n\nBelow are the metrics for classification of class for the breast cancer data set.")
    val_data, test_data, prediction = cross_val(cancer, class_var='class', k=5, validation=False,
                                                pred_type='classification')
    eval_metrics(test_data['class'].tolist(), prediction, 'classification')

    file1.write("\n\nBelow are the metrics for classification of class for the car data set.")
    val_data, test_data, prediction = cross_val(cars, class_var='class', k=5, validation=False,
                                                pred_type='classification')
    eval_metrics(test_data['class'].tolist(), prediction, 'classification')

    file1.write("\n\nBelow are the metrics for regression of area for the forest data set.")
    val_data, test_data, prediction = cross_val(forest, class_var='area', k=5, validation=True, pred_type='regression')
    eval_metrics(test_data['area'].tolist(), prediction, 'regression')

    file1.write("\n\nBelow are the metrics for classification of class for the house data set.")
    val_data, test_data, prediction = cross_val(house, class_var='class_republican', k=5, validation=False,
                                                pred_type='classification')
    eval_metrics(test_data['class_republican'].tolist(), predicted=prediction, eval_type='classification')

    file1.write("\n\nBelow are the metrics for regression of ERP for the machine data set.")
    val_data, test_data, prediction = cross_val(data=machine, class_var='erp', k=5, validation=True,
                                                pred_type='regression')
    eval_metrics(test_data['erp'].tolist(), predicted=prediction, eval_type='regression')
    file1.close()
    file2.close()
    file3.close()
