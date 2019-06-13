# import dependencies
from BaseObject import BaseObject
from defaults import split_map

class Split(BaseObject):

    def __init__(self, name, options, map_dict=split_map):
        super().__init__(name, options, map_dict)

    def get_indices_for_sets(self, *args):
        """
        Use the function specified in the model selection mapping dictionary to split data into train, validate, and test sets. Whatever function being used should return the indicies of the datasets with options to have them shuffled and stratified.

        For train-validate-split, the indicies should be contained in a dictionary with keys ['train', 'validate', 'test']; the values should be np.int32 arrays.

        For k-folds, the indicies should be contained in a nested dictionary. The root level should have int keys from 0 to number of folds; the values should be the same kind of dictionary from train-validate-split.
        """
        return self.map[self.name](*args, **self.options)
