# Tactile Exploration of Objects (tum-adlr-02)

![](docs/video_readme.gif)

Robot grasping relies on accurate spatial models from
sensory data of depth cameras, cameras or tactile exploration.
Related work relies often on point cloud data from cameras
in combination with sparse tactile data. We explore a novel
approach to tactile exploration with only sparse tactile data
availabe. For this task we prepare our own dataset, train a
reconstruction network for shape prediction and enhance the
tactile exploration with reinforcement learning. The results show
an increase in performance in comparison to a random policy.
 
- [Click here to view the *Project Report*](docs/ADLR_final_report.pdf)
- [Click here to view the *Project Poster*](docs/adlr-02-poster.pdf)

To get accesss to the trained models, feel free to mail one of the authors. 
They where added to gitignore due to their size. 


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
  - jupyter
  - notebook
  - tqdm



##  Project Structure


```sh
project-root/
├── README.md
├── demo.py
├── package_versions.txt
├── data/
│   └── 2D_shapes/
│       ├── train.csv
│       ├── test.csv
│       ├── eval.csv
│       ├── bottle/
│       └── mug/
├── docs/
│   ├── ADLR_final_report.pdf
│   ├── adlr-02-poster.pdf
│   ├── video_readme.gif
│   └── video_readme_c.gif
├── outputs/
│   ├── reconstruction_models/
│   ├── rl_models/
│   └── rl_runs/
├── src/
│   ├── __init__.py
│   ├── util_functions.py
│   ├── data_preprocessing/
│   ├── evaluation/
│   ├── neural_nets/
│   ├── plots_plakat/
│   ├── showcase/
│   ├── stable_baselines_code/
│   └── train_reconstruction/
└── __pycache__/
```


###  Project Index
<details open>
	<summary><b><code>ADLR_TACTILE_EXPLORATION.GIT/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/package_versions.txt'>package_versions.txt</a></b></td>
				<td>- The code file `package_versions.txt` serves as a reference for creating an environment within the project using Conda<br>- It specifies the necessary package versions and dependencies required for the project to run smoothly on a Linux-64 platform<br>- This file plays a crucial role in ensuring the correct setup and configuration of the project environment.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- src Submodule -->
		<summary><b>src</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/util_functions.py'>util_functions.py</a></b></td>
				<td>- Implements utility functions for image array manipulation, conversion, and processing<br>- Includes functions for converting image arrays to point lists, adding color dimensions, converting array shapes, combining two images, and adding a zero channel<br>- These functions facilitate image processing and manipulation within the codebase architecture.</td>
			</tr>
			</table>
			<details>
				<summary><b>stable_baselines_code</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/stable_baselines_code/example_usage_environment.py'>example_usage_environment.py</a></b></td>
						<td>- Implement a dummy neural network for processing image data and generating convex hull vertices<br>- The code sets up an environment using the network, dataset, loss function, and reward function<br>- It then runs a sample loop to interact with the environment, taking random actions until completion.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/stable_baselines_code/callback.py'>callback.py</a></b></td>
						<td>- Implements a custom callback for adding data to TensorBoard during training<br>- Manages logging of rewards, losses, and metrics at specified intervals<br>- Handles visualization of images and provides hooks for various training events.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/stable_baselines_code/reward_functions.py'>reward_functions.py</a></b></td>
						<td>- Define various reward functions based on losses, metrics, and occurrences in the codebase to calculate rewards for different scenarios<br>- Functions include dummy_reward, basic_reward, complex_reward, improve_reward, reward_1, and reward_2, each serving a specific purpose in determining the final reward value.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/stable_baselines_code/environment.py'>environment.py</a></b></td>
						<td>- Implements a custom environment following the gym interface, allowing interaction with a reconstruction network for shape inference<br>- Handles actions, observations, rendering, and resets, facilitating reinforcement learning training with different reward functions<br>- Supports visualization of grasp points and ray casting.</td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>showcase</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/showcase/rl_agent_plots_jan.py'>rl_agent_plots_jan.py</a></b></td>
						<td>- Generates plots showcasing reinforcement learning agent performance using Stable Baselines3<br>- Loads pre-trained models, runs simulations, and saves visualizations based on rewards achieved<br>- Facilitates evaluation and comparison of RL models through visual representation of agent behavior and performance metrics.</td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>evaluation</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/evaluation/reconstruction_validator.py'>reconstruction_validator.py</a></b></td>
						<td>- Validate neural network reconstruction accuracy on evaluation datasets using a custom RecNet model<br>- Load data, infer dataset metrics, and print results for training, validation, and test sets.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/evaluation/rl_perform_gp.py'>rl_perform_gp.py</a></b></td>
						<td>- Evaluate and store statistics for reinforcement learning models using Stable Baselines3<br>- Load pre-trained models, run evaluations, and save results for future analysis<br>- The code interacts with a custom environment and neural network components to assess model performance<br>- This file plays a crucial role in analyzing and optimizing RL policies within the project architecture.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/evaluation/rl_perform_gp_daniel.py'>rl_perform_gp_daniel.py</a></b></td>
						<td>- Generate statistical data on reinforcement learning policies using stable baselines and neural networks<br>- The code evaluates multiple models on a dataset, calculating mean and standard deviation of rewards per grasp<br>- Results are saved for further analysis.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/evaluation/rl_perform.py'>rl_perform.py</a></b></td>
						<td>- Evaluate and store statistics of RL policies using PPO algorithm on a dataset<br>- Load pre-trained models, run evaluations, and save results for future reference<br>- The code interacts with a custom environment and neural network components to analyze policy performance.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/evaluation/rl_eval.py'>rl_eval.py</a></b></td>
						<td>- Implementing reinforcement learning evaluation using Stable Baselines3, the code in rl_eval.py initializes a ShapeEnv environment with a RecNet neural network and complex reward function<br>- It loads a pre-trained PPO model and runs multiple episodes to evaluate the agent's performance<br>- This file serves as a crucial component for assessing the reinforcement learning model within the project architecture.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/evaluation/rl_test.py'>rl_test.py</a></b></td>
						<td>- Implementing reinforcement learning evaluation using Stable Baselines3, the code in rl_test.py sets up a ShapeEnv environment with a RecNet model and custom reward function<br>- It trains a PPO model, evaluates its performance, and saves the trained model for future use<br>- Additionally, it includes an example run function for demonstration purposes.</td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>train_reconstruction</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/train_reconstruction/tuner.py'>tuner.py</a></b></td>
						<td>- Optimize hyperparameters for neural network training using Ray Tune's BOHB algorithm<br>- Search for the best configuration to minimize loss during reconstruction tasks<br>- Save the best trial's results for further analysis and model improvement.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/train_reconstruction/trainer.py'>trainer.py</a></b></td>
						<td>- Handles loading and configuring the best trial model for reconstruction tasks<br>- Merges the best trial configuration with new settings, such as epochs and workers<br>- Sets up scaling and run configurations for the model.</td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>neural_nets</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/neural_nets/utility_functions.py'>utility_functions.py</a></b></td>
						<td>- Train reconstruction models using specified configurations, datasets, and neural network models<br>- Utilize DataLoader for training and validation, handling checkpoints for model saving<br>- Report training progress and results<br>- The function encapsulates the training process for neural network models in the project's architecture.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/neural_nets/rec_net.py'>rec_net.py</a></b></td>
						<td>- Implements a neural network for image reconstruction using a pre-trained UNet model<br>- Handles inference on input data and evaluation metrics calculation<br>- Offers the flexibility to run on CPU or GPU<br>- Includes a utility for generating a dummy reconstruction based on convex hull image processing.</td>
					</tr>
					</table>
					<details>
						<summary><b>models</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/neural_nets/models/unet.py'>unet.py</a></b></td>
								<td>- Implements UNet neural network architecture for image segmentation<br>- Defines contracting and expansive blocks for encoding and decoding<br>- Supports different depths and channel configurations<br>- The forward method processes input through encoder, bottleneck, and decoder, producing a final output<br>- Multiple UNet variations cater to varying model complexities.</td>
							</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
			<details>
				<summary><b>data_preprocessing</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/data_preprocessing/data_creator.py'>data_creator.py</a></b></td>
						<td>- Generates a standard dataset for the reconstruction network by creating 2D datasets with specified parameters like resolution, classes, and rotations<br>- The code utilizes a DataConverter to preprocess the data and generate the required dataset for training the network.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/data_preprocessing/dataconverter.py'>dataconverter.py</a></b></td>
						<td>- The `DataConverter` class in the provided codebase facilitates the conversion of 3D shapes to 2D images for specific object classes<br>- It manages the download of datasets, generation of 2D images, and creation of tactile point datasets<br>- Additionally, it offers functionalities to display random 3D and 2D samples, aiding in visualizing the processed data.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/data_preprocessing/reconstruction_dataset.py'>reconstruction_dataset.py</a></b></td>
						<td>- Enables visualization and transformation of tactile data for reconstruction and reinforcement datasets<br>- Facilitates displaying data pairs and batches, loading images and labels, and applying transformations like tensor conversion, random flipping, and orientation adjustments<br>- Supports dataset creation and manipulation for machine learning tasks.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/src/data_preprocessing/model_classes.py'>model_classes.py</a></b></td>
						<td>- Define model classes with assigned IDs and URLs for easy access in the data loader, streamlining the process and reducing complexity<br>- Future-proof by allowing for additional parameters per class, accommodating potential conversions or specific requirements like light reflection or camera angles for different objects.</td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
</details>

---





















