# import dependencies
import numpy as np

#===============================================================================
# Split
#===============================================================================

def check_for_imbalanced_data(DF, **kwargs):
    """Checks the array is imbalanced. Returns a dictionary with keys, ['imbalanced', 'bincount', 'unique']; its values are whether the data set is imbalanced and the bincount and unique arrays, which are both sorted by bincount."""

    # options
    threshold = kwargs['threshold'] if 'threshold' in kwargs else 0.2
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    column = kwargs['column'] if 'column' in kwargs else 'target'

    # check input
    assert column in DF, f"{column} is not a column in the given DataFrame, try using: column = <desired column>"
    y = DF.target.to_numpy(np.int32)
    assert y.shape[0] == y.size, "to bincounts, input array must be 1D"

    bincount = np.bincount(y) # how many observations for each unique value
    unique = np.unique(y) # find all the unique values
    N = y.shape[0] # total number of observations

    for i,v in enumerate(unique):
        assert y[y==v].size == bincount[i], "for some reason, the unique values and the bincounts don't match"

    imbalanced = np.any(bincount/N < threshold) # find out if the data is imbalanced
    if verbose:
        print("\nchecking if data is imbalanced...")
        if imbalanced:
            print(f"data is imbalanced (threshold = {threshold})")
        else:
            print(f"data is not imbalanced (threshold = {threshold})")

        print("unique values line up with bincounts")

    # sort bincount and uniques
    idx = bincount.argsort() # find the indicies to sort bincount

    return {'imbalanced':imbalanced, 'bincount':bincount[idx], 'unique':unique[idx]}

def split_stratify(DF, **kwargs):
    """Stratifies the given DataFrame based on the specified column (default: column = 'target') and returns a dictionary of np.int32 index arrays with keys ['train', 'validate', 'test']."""

    # options
    train_size = kwargs['train_size'] if 'train_size' in kwargs else 0.6
    validate_size = kwargs['validate_size'] if 'validate_size' in kwargs else 0.2
    shuffle = kwargs['shuffle'] if 'shuffle' in kwargs else True
    seed = kwargs['seed'] if 'seed' in kwargs else 0
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False

    if verbose: print("\nStratifying observations...")

    check = check_for_imbalanced_data(DF, **kwargs)

    # set up empty containers for indices
    idx = np.arange(DF.shape[0])
    indices = {
        'train'    : np.array([], dtype=np.int32),
        'validate' : np.array([], dtype=np.int32),
        'test'     : np.array([], dtype=np.int32)
    }

    for i,(value,count) in enumerate(zip(check['unique'],check['bincount'])):
        idx1 = idx[ y == value ]
        N = idx1.size
        assert N == count, "For some reason, the counts don't line up."
        N_train = int(N*train_size)
        N_validate = int(N*validate_size)
        indices['train'] = np.hstack(( indices['train'], idx1[:N_train] ))
        indices['validate'] = np.hstack(( indices['validate'], idx1[N_train:N_train+N_validate]))
        indices['test'] = np.hstack(( indices['test'], idx1[N_train+N_validate:]))

    if shuffle:
        if verbose: print("shuffling stratified observations...")
        for key in ['train', 'validate', 'test']:
            np.random.RandomState(seed=seed).shuffle(indices[key])

    return indices

def split_straight(DF, **kwargs):
    """Splits the given DataFrame (w\\o stratifing) based on the specified column (default: column = 'target') and returns a dictionary of np.int32 index arrays with keys ['train', 'validate', 'test']."""

    # options
    train_size = kwargs['train_size'] if 'train_size' in kwargs else 0.6
    validate_size = kwargs['validate_size'] if 'validate_size' in kwargs else 0.2
    shuffle = kwargs['shuffle'] if 'shuffle' in kwargs else True
    seed = kwargs['seed'] if 'seed' in kwargs else 0
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False

    # find number of observations for each dataset
    N = DF.shape[0]
    N_train = int(N*train_size)
    N_validate = int(N*validate_size)

    if shuffle:
        if verbose: print("shuffling observations...")
        idx = np.random.RandomState(seed=seed).permutation(N)
    else:
        idx = np.arange(N)

    if verbose: print("splitting observations w/o stratifying...")
    indices = {
        'train'    : idx[ : N_train ],
        'validate' : idx[ N_train : N_train + N_validate ],
        'test'     : idx[ N_train + N_validate : ]
    }

    return indices

def train_test_split_indices(DF, **kwargs):
    """Returns the indicies for the train, validate, and test sets as a dictionary with keys: 'train', 'validate', and 'test', with int32 numpy arrays as their values."""

    # options
    stratify = kwargs['stratify'] if 'stratify' in kwargs else False

    if stratify:
        return split_stratify(DF, **kwargs)
    else:
        return split_straight(DF, **kwargs)

def kfolds_indices(DF, **kwargs):
    """Splits the DataFrame k-fold, returns a dictionary with int keys from 0 to k; the values are the index dictionaries for eack of the k folds. The index dictionaries are np.int32 arrays with keys ['train', 'validate', 'test']."""

    # options
    k = kwargs['k'] if 'k' in kwargs else 10
    stratify = kwargs['stratify'] if 'stratify' in kwargs else False
    column = kwargs['column'] if 'column' in kwargs else 'target'
    shuffle = kwargs['shuffle'] if 'shuffle' in kwargs else True

    # get an array of indices
    N = DF.shape[0]
    Nk = N//k
    if shuffle:
        idx = np.random.RandomState(seed=seed).permutation(N)
    else:
        idx = np.arange(N)

    indices = {}
    for i in range(k):
        idx1 = idx[ i*N1 : (i+1)*N1 ]
        DF1 = DF.iloc[idx1]
        if stratify:
            indices[i] = split_stratify(DF1, **kwargs)
        else:
            indices[i] = split_straight(DF1, **kwargs)

#===============================================================================
# Model Selection
#===============================================================================
