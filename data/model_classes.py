'''
For each 3D model class in shapenet there is an assigned id, which is needed to download this class.
Here we can add assignements of name of model class to its id and url, to make the code in dataloader less messy.
Additionally we might need more parameters per class in the future. For example if we convert one class to 2D
that has a surface that reflects light a lot, or maybe for mugs camera has to be turned certain way to capture the hole,
parameters for that calculation should be adapted.
'''
import string


class ModelClass():
    name: string
    id: string

    def get_url(self):
        return 'http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/' + self.id + '.zip'


class Bottle(ModelClass):
    name = 'bottle'
    id = '02876657'


class Mug(ModelClass):
    name = 'mug'
    id = '03797390'
