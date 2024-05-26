import os
import requests
import zipfile
import numpy as np
import pyrender
import trimesh
import random
import matplotlib.pyplot as plt
import pandas as pd
########################################
from data.model_classes import *


#####################################################
############### STATIC METHODS ######################
#####################################################

# grayscale images should be of shape [1,res,res] instead of [res,res] for the
# network to work
def add_channel(image):
    return np.array([image])

def mesh_file_to_scene(mesh_file):
    c_model = trimesh.load_mesh(mesh_file)

    if type(c_model) == trimesh.base.Trimesh:
        mesh = pyrender.Mesh.from_trimesh(c_model)
        scene = pyrender.Scene()
        scene.add(mesh)
    elif type(c_model) == trimesh.scene.scene.Scene:
        scene = pyrender.Scene.from_trimesh_scene(c_model)
    else:
        raise ValueError("Modeltype not known. Type: " + type(c_model) + " |  from file: " + mesh_file)
    return scene


def files_in_dir(dir, name_end):
    # searches directory and all subdirectorys for files ending with name_end
    # returns paths to all these files
    file_paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(name_end):
                file_paths.append(os.path.join(root, file))

    return file_paths


def convert_to_boolean_image(image):
    # Convert object pixels to 1 by using depth map. Background is zero
    return image > 0

def find_outline(image):
    # Only returns reachable outline and avoids inner holes. Iteratively approximates outline from both sides for each dimension.
    # Removes duplicates from corners afterwards.
    r, c = image.shape
    xy = []

    # Left and right
    for i in range(r):
        p = 0
        q = c - 1
        while p <= q and not image[i, p]:
            p +=1
        while p <= q and not image[i, q]:
            q -= 1
        if p <= q:
            xy.append([i,p])
            xy.append([i,q])
    
    # Top and bottom
    for i in range(c):
        p = 0
        q = r - 1
        while p <= q and not image[p, i]:
            p +=1
        while p <= q and not image[q, i]:
            q -= 1
        if p <= q:
            xy.append([p,i])
            xy.append([q,i])
    
    xy = np.array(xy)
    xy = np.unique(xy, axis=0)

    return xy

def find_outline_old(image):
    # returns the points of the outline by checking if for every pixel of the object at least one neighboring pixel is background
    # todo: avoid holes in object
    w, h = image.shape
    xy = []
    for i in range(w):
        for j in range(h):
            if image[i, j] and not (image[max(0, i-1), j] and image[i, max(0, j-1)] and image[min(w-1, i+1), j] and image[i, min(h-1, j+1)]):
                xy.append([i, j])
    return np.array(xy)

def generate_tactile_images(outline, path, l_path, l_label_path, res=250, amount=10, order=5):
    # generates 'amount' images of tactile points for each number of total tactile points up to 'order'
        r, _ = outline.shape
        paths = []
        for o in range(1, order+1):
            for n in range(amount):
                idx = np.random.choice(r, o)
                points = outline[idx,:]
                image = np.full((res, res), False)
                image[points[:,0], points[:,1]] = True
                
                file_path = os.path.join(path, f"o{o}n{n}tactile.npy")
                paths.append([os.path.join(l_path, f"o{o}n{n}tactile.npy"), l_label_path])
                np.save(file_path, add_channel(image))
        return pd.DataFrame(paths, columns=['image', 'label'])



#####################################################
################## DATA CONVERTER ###################
#####################################################

class DataConverter:
    # todo: many more parameters will be passed to this class, find good way to do that, without making it messy
    # todo: make that multiple classes or an arbitrary class that is not bottle can be loaded

    input_path = './datasets/3D_shapes'
    output_path = './datasets/2D_shapes'

    def __init__(self,
                 res=256,
                 classes=[Bottle()],
                 tact_order=5,
                 tact_number=10
                 ):

        self.res = res
        self.classes = classes

        # Maximum number of tactile points per dataset
        self.tact_order = tact_order
        # Number of sampled tactile images per order of tactile points
        self.tact_number = tact_number

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

    def generate_2d_dataset(self, regenerate=True, show_results=False, redownload=False):

        self.download_dataset(redownload=redownload)

        # create output directory, if it doesnt exist
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        frames = []

        csv_path = os.path.join(self.output_path, 'annotations.csv')
        # if os.path.exists(csv_path):
        #     df = pd.read_csv(csv_path)
        #     df.reset_index(drop=True, inplace=True)
        #     frames.append(df)

        for cls in self.classes:
            # Check if conversion has been done before
            if (not regenerate
                    and os.path.exists(os.path.join(self.output_path, cls.name))):
                print("2D images for class" + cls.name + " already exist. Skipping conversion.")
                continue
            df = self.generate_2d_dataset_aux(cls, regenerate=regenerate, show_results=show_results)
            frames.append(df)
        
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df.reset_index(drop=True, inplace=True)
            df.to_csv(csv_path, index=False)
            

    def generate_2d_dataset_aux(self, cls: ModelClass, regenerate=True, show_results=False):

        frames = []

        # Load ShapeNet dataset for class
        mesh_files = files_in_dir(
            os.path.join(self.input_path, cls.name),
            '.obj')

        print("Converting ", len(mesh_files), " files ...")
        for mesh_file in mesh_files:
            # Render mesh to 2D image
            scene = mesh_file_to_scene(mesh_file)

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
            _, depth = renderer.render(scene)
            image = convert_to_boolean_image(depth)

            if show_results:
                # set True, to have a look at the 3d scene
                if False:
                    pyrender.Viewer(scene, use_raymond_lighting=True, viewport_size=(800, 600))
                plt.imshow(image)
                plt.pause(.1)

            renderer.delete()

            # TODO: Simplify the following block

            # Save 2D image
            data_id = mesh_file.split(os.sep)[-2]
            local_path = os.path.join(cls.name, data_id)
            data_path = os.path.join(self.output_path, local_path)
            image_path = os.path.join(data_path, "image.npy")
            l_image_path = os.path.join(local_path, "image.npy")
            outline_path = os.path.join(data_path, "outline.npy")
            tactile_path = os.path.join(data_path, "tactile_points")
            l_tactile_path = os.path.join(local_path, "tactile_points")


            os.makedirs(tactile_path, exist_ok=True)
            np.save(image_path, add_channel(image))

            # Generate and save tactile point images
            outline = find_outline(image)
            np.save(outline_path, add_channel(outline))
            df = generate_tactile_images(outline, tactile_path, l_tactile_path, l_image_path, self.res, amount=self.tact_number, order=self.tact_order)
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def display_random_3d_samples(self, num_samples=5):
        # todo: some samples from the dataset can not be displayed using this code

        self.download_dataset(redownload=False)
        shape_path = os.path.join(self.input_path)

        mesh_files = files_in_dir(shape_path, '.obj')

        sample_files = random.sample(mesh_files, num_samples)

        for sample_file in sample_files:
            scene = mesh_file_to_scene(sample_file)
            pyrender.Viewer(scene, use_raymond_lighting=True, viewport_size=(800, 600))

    def display_random_2d_samples(self, num_samples=5):
        # todo: also plot sample points on top of image

        image_files = files_in_dir(self.output_path, 'image.npy')

        if len(image_files) == 0:
            print("No 2D images found. Please run the conversion first.")
            return

        num_samples = min(len(image_files), num_samples)
        sample_files = random.sample(image_files, num_samples)

        fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
        for ax, sample_file in zip(axes, sample_files):
            img = np.load(sample_file)[0]
            ax.imshow(img)
            ax.axis('off')
        plt.show()
    
    def display_random_data_pairs(self, num_samples=5):

        image_files = files_in_dir(self.output_path, 'image.npy')

        if len(image_files) == 0:
            print("No 2D images found. Please run the conversion first.")
            return
        
        num_samples = min(len(image_files), num_samples)
        sample_files = random.sample(image_files, num_samples)

        fig, axes = plt.subplots(num_samples, 2, figsize=(80,80))
        fig.tight_layout()
        for i in range(num_samples):
            sample_file = sample_files[i]
            img = np.load(sample_file)
            tactile_path = os.path.join(os.path.dirname(sample_file), 'tactile_points')
            tactile_files = files_in_dir(tactile_path, 'tactile.npy')
            tactile_file = random.sample(tactile_files, 1)
            pts = np.load(tactile_file[0])

            axes[i, 0].imshow(img[0])
            axes[i, 0].axis('off')
            axes[i, 1].imshow(pts[0])
            axes[i, 1].axis('off')
        plt.show()


#Example usage:
'''
dataconverter = DataConverter(
    classes=[Mug(), Bottle()],
)

# run this line to see whether you can download the data from shapenet and display the files correctly
#dataloader.display_random_3d_samples(num_samples=10)

# run these to lines to check whether you can convert the models to 2d, find sample points and
# save and load the results correctly

dataconverter.generate_2d_dataset(show_results=False, regenerate=True)
dataconverter.display_random_2d_samples(num_samples=5)

dataconverter.display_random_data_pairs(num_samples=5)
'''

if __name__ == "__main__":
    # generate data
    dataconverter = DataConverter(
        classes=[Mug(), Bottle()],
        tact_order = 30
    )
    # set regenerate to true, if you run this after changes in dataconverter have been made
    dataconverter.generate_2d_dataset(show_results=False, regenerate=True)