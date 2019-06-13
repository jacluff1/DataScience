# import dependencies
from BaseObject import BaseObject

class ModelSelection(BaseObject):

    def __init__(self, name, options, map_dict=split_map):
        super().__init__(name, options, map_dict)
