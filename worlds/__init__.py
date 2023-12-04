from worlds.line import LineWorld
from worlds.craft import CraftWorld
from worlds.light import LightWorld

def load(EnvName):
    cls_name = EnvName
    try:
        cls = globals()[cls_name]
        return cls()
    except KeyError:
        raise Exception("No such world: {}".format(cls_name))
