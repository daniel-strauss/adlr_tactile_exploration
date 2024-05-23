# tum-adlr-02

## Requirements

- Python 3.10

- Python Packages:
  - requests
  - zipfile
  - numpy
  - pyrender
  - trimesh
  - matplotlib
  - torch
  - torchvision
  - torchaudio
  - pandas
  - scikit-image

## Documentation

### DataConverter

The `DataConverter` class facilitates handling the ShapeNet dataset. 
It downloads the dataset, converts 3D models to 2D images, generates randomly selected tactile points on the outline of the objects, and displays random samples from both the 3D models and the 2D images and random samples from data pairs.

#### Features

- **Automatic Downloading**: Downloads the ShapeNet "bottle" dataset if not already present.
- **Conversion to 2D Images**: Converts 3D models to 2D images using Pyrender.
- **Randomly sampling points:** Randomly samples tactile points from the outline of the object and saves them as images as input data for the reconstruction network.
- **Sample Visualization**: Displays random samples from the original 3D models, the converted 2D images and data pairs of tactile points and object shape for testing purposes. 

#### Usage

An example usage is provided at the bottom of the file `dataconverter.py`

