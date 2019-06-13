# import dependencies

class BaseObject:

    def __init__(self, name, options, map_dict):
        self.update(name, options)

    def update(self, name, options, map_dict):
        assert name in map_dict, f"{name} must be a key in the map dictionary. keys for dictionary provided are {map_dict.keys()}."
        self.name = name
        self.options = options
        self.map = map_dict
