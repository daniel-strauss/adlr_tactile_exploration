import os
import requests
import zipfile
import numpy as np
import pyrender
import trimesh
import random
import matplotlib.pyplot as plt
########################################
from model_classes import *


class DataLoader:
    # todo: many more parameters will be passed to this class, find good way to do that, without making it messy
    # todo: make that multiple classes or an arbitrary class that is not bottle can be loaded

    input_path = './input_data/3D_shapes'
    output_path = './input_data/2D_shapes'

    def __init__(self,
                 res=250,
                 classes=[Bottle()]
                 ):

        self.bottle_id = '02876657'  # ShapeNetCore.v1 ID for 'bottle'
        self.res = res

        self.classes = classes

    def create_traning_and_validation_batches(self):
        # todo: create training and validation batches and load them into main memory somehow such that no time is lost
        # during training when loading from disk. at i2dl they did it also somehow, but I forgot
        pass

    def download_dataset(self, redownload=True):

        if not os.path.exists(self.input_path):
            os.makedirs(self.input_path)

        for cls in self.classes:
            cls_path = os.path.join(self.input_path, cls.name)
            if redownload or not os.path.exists(cls_path):
                print(f"Downloading bottle dataset from {cls.get_url()}...")
                response = requests.get(cls.get_url(), stream=True)
                zip_path = os.path.join(self.input_path, 'bottle.zip')
                with open(zip_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            file.write(chunk)
                print("Download complete. Extracting files...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(cls_path)
                os.remove(zip_path)
                print("Extraction complete.")
            else:
                print("class " + cls.name + " already downloaedd. Skipping download.")

    def generate_2d_dataset(self, regenerate=True, show_results=False):

        self.download_dataset(redownload=regenerate)

        # create output directory, if it doesnt exist
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        for cls in self.classes:
            self.generate_2d_dataset_aux(cls, regenerate=regenerate, show_results=show_results)

    def generate_2d_dataset_aux(self, cls: ModelClass, regenerate=True, show_results=False):

        # Check if conversion has been done before
        if (not regenerate
                and os.path.exists(os.path.join(self.output_path, cls.name))):
            print("2D images for class" + cls.name + " already exist. Skipping conversion.")
            return

        # Load ShapeNet dataset for class
        mesh_files = self.files_in_dir(
            os.path.join(self.input_path, cls.name),
            '.obj')

        print("Converting ", len(mesh_files), " files ...")
        for mesh_file in mesh_files:
            # Render mesh to 2D image
            c_model = trimesh.load_mesh(mesh_file)

            if type(c_model) == trimesh.base.Trimesh:
                mesh = pyrender.Mesh.from_trimesh(c_model)
                scene = pyrender.Scene()
                scene.add(mesh)
            elif type(c_model) == trimesh.scene.scene.Scene:
                scene = pyrender.Scene.from_trimesh_scene(c_model)
            else:
                raise ValueError("Modeltype not known. Type: " + type(c_model) + " |  from file: " + mesh_file)

            # add camera
            camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
            s = np.sqrt(2) / 2
            camera_pose = np.array([
                [0.0, -s, s, 0.3],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, s, s, 0.35],
                [0.0, 0.0, 0.0, 1.0],
            ])
            scene.add(camera, pose=camera_pose)
            # add no light on purpose to the scene
            renderer = pyrender.OffscreenRenderer(viewport_width=self.res, viewport_height=self.res)
            image, depth = renderer.render(scene)
            image = self.convert_to_black_white(image)

            if show_results:
                # set True, to have a look at the 3d scene
                if False:
                    pyrender.Viewer(scene, use_raymond_lighting=True, viewport_size=(800, 600))
                plt.imshow(image)
                plt.pause(.1)

            renderer.delete()

            # Save 2D image
            data_id = mesh_file.split(os.sep)[-2]
            data_path = os.path.join(self.output_path, cls.name, data_id)
            image_path = os.path.join(data_path, "image.npy")
            sampled_points_path = os.path.join(data_path, "sample_points.npy")
            os.makedirs(data_path, exist_ok=True)

            np.save(image_path, image)

            # @JAN: you are to create and store the sample points somewere
            # else in the code if that makes more sense to you
            sample_points = self.create_sample_points(image)
            np.save(sampled_points_path, sample_points)

    def create_sample_points(self, image):
        # TODO finish function

        return np.array([])

    def convert_to_black_white(self, image):
        # calculate lighness by summing up rgbs
        image = image.sum(axis=2)  # todo: there is a better meassures for lighness
        # background is white, thus turning every not white pixel black
        return image <= 3 * 255 - 10

    def display_random_3d_samples(self, cls: ModelClass, num_samples=5):
        # todo: some samples from the dataset can not be displayed using this code

        self.download_dataset()
        shape_path = os.path.join(self.input_path, cls.id)

        mesh_files = self.files_in_dir(shape_path, '.obj')

        sample_files = random.sample(mesh_files, num_samples)

        for sample_file in sample_files:
            c_trimesh = trimesh.load_mesh(sample_file)
            scene = pyrender.Scene.from_trimesh_scene(c_trimesh)
            pyrender.Viewer(scene, use_raymond_lighting=True, viewport_size=(800, 600))

    def display_random_2d_samples(self, num_samples=5):
        # todo: also plot sample points on top of image

        image_files = self.files_in_dir(self.output_path, 'image.npy')

        if len(image_files) == 0:
            print("No 2D images found. Please run the conversion first.")
            return

        num_samples = min(len(image_files), num_samples)
        sample_files = random.sample(image_files, num_samples)

        fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
        for ax, sample_file in zip(axes, sample_files):
            img = np.load(sample_file)
            ax.imshow(img)
            ax.axis('off')
        plt.show()

    def files_in_dir(self, dir, name_end):
        file_paths = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(name_end):
                    file_paths.append(os.path.join(root, file))

        return file_paths


# Example usage:
dataloader = DataLoader(
    classes=[Mug(), Bottle()]
)

# run this line to see whether you can download the data from shapenet and display the files correctly
#dataloader.display_random_3d_samples()

# run these to lines to check whether you can convert the models to 2d, find sample points (todo) and
# save and load the results correctly
dataloader.generate_2d_dataset(show_results=False, regenerate=False)
dataloader.display_random_2d_samples()
