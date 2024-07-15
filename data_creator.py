from data.dataconverter import *

'''
Generates standard dataset for the reconstruction network.
'''

conv = DataConverter(
    res=256,
    classes=[Bottle(), Mug()],
    min_order=1,
    tact_order=10,
    tact_number=1,
    rand_rotations=5,
    split=(0.8, 0.1, 0.1),
    save_float=False
)

conv.generate_2d_dataset(
    regenerate=False,
    show_results=False,
    redownload=False
)