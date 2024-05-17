# tum-adlr-02

## Dependencies

- Python 3.10
- TODO add other requirements


## Documentation

### DataLoader

The `DataLoader` class facilitates handling the ShapeNet dataset. 
It downloads the dataset, converts 3D models 
to 2D images, and displays random samples from both the 3D models 
and the 2D images.

#### Features

- **Automatic Downloading**: Downloads the ShapeNet "bottle" dataset if not already present.
- **Conversion to 2D Images**: Converts 3D models to 2D images using Pyrender.
- **Randomly sampling points:**
- **Sample Visualization**: Displays random samples from the original 3D models and the converted 2D images.

#### Usage

An example usage is provided at the bottom of the file `dataloader.py`

