# import dependencies
import Functions as func
import sklearn.model_selection as model_selection
import sklearn.ensemble as ensemble

#===============================================================================
# map dictionaries
#===============================================================================

split_map = {
    'train test split' : func.train_test_split_indices,
    'k folds' : sklearn.model_selection.KFold,
}

model_selection_map = {
    'grid search' : sklearn.model_selection.GridSearchCV, # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    'random search' : sklearn.model_selection.RandomizedSearchCV, # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
}

model_map = {

}

#===============================================================================
# default option dictionaries
#===============================================================================
