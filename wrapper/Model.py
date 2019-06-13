# import dependencies
from BaseObject import BaseObject
from defaults import model

class Model(BaseObject):

    def __init__(self, name, options, map_dict=model_map):
        super().__init__(name, options, map_dict)
