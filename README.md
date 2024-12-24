# Tactile Exploration of Objects (tum-adlr-02)

![](outputs/presenation_resources/video_readme.gif)

Robot grasping relies on accurate 3D models from
sensory data of depth cameras, cameras or tactile exploration.
Related work relies often on point cloud data from cameras
in combination with sparse tactile data. We explore a novel
approach to tactile exploration with only sparse tactile data
availabe. For this task we prepare our own dataset, train a
reconstruction network for shape prediction and enhance the
tactile exploration with reinforcement learning. The results show
an increase in performance in comparison to a random policy.
 
- [Click here to view the *Project Report*](outputs/presenation_resources/ADLR_final_report.pdf)
- [Click here to view the *Project Poster*](outputs/presenation_resources/adlr-02-poster.pdf)


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
└── adlr_tactile_exploration.git/
    ├── README.md
    ├── deprecated
    │   ├── neural_nets
    │   │   ├── models
    │   │   ├── trainer.py
    │   │   └── weight_inits.py
    │   ├── notebooks
    │   │   ├── Reconstruction.ipynb
    │   │   ├── Reconstruction_2.ipynb
    │   │   ├── debug_gpu_not_available.ipynb
    │   │   ├── example_usage_trainer.ipynb
    │   │   └── example_usage_trainer_PARAM_SEARCH.ipynb
    │   ├── rl_time_tester.py
    │   ├── stuff.txt
    │   └── train_from_dict_2024-06-22_03-16-14
    │       ├── .validate_storage_marker
    │       ├── basic-variant-state-2024-06-22_03-16-14.json
    │       ├── experiment_state-2024-06-22_03-16-14.json
    │       ├── train_from_dict_05210_00000_0_batch_size=16,channels=32,depth=7,lr=0.0071_2024-06-22_03-16-15
    │       └── train_from_dict_05210_00001_1_batch_size=2,channels=32,depth=9,lr=0.0003_2024-06-22_03-16-15
    ├── outputs
    │   ├── plots_plakat
    │   │   ├── color_shemes.py
    │   │   ├── performance_plot.png
    │   │   ├── performance_plot_gp.png
    │   │   ├── plot_data
    │   │   ├── reconstruction_net_plots.py
    │   │   ├── rl_agent_plots.py
    │   │   ├── rl_functions_performance_plot.py
    │   │   └── rl_functions_train_plot.py
    │   └── presenation_resources
    │       ├── ADLR_final_report.pdf
    │       ├── adlr-02-poster.pdf
    │       ├── video_readme.gif
    │       └── video_readme_c.gif
    ├── package_versions.txt
    ├── plots_plakat
    │   └── temp
    │       └── rl_plots
    ├── readme-ai.md
    └── src
        ├── data_preprocessing
        │   ├── data_creator.py
        │   ├── dataconverter.py
        │   ├── model_classes.py
        │   └── reconstruction_dataset.py
        ├── evaluation
        │   ├── reconstruction_validator.py
        │   ├── rl_eval.py
        │   ├── rl_perform.py
        │   ├── rl_perform_gp.py
        │   ├── rl_perform_gp_daniel.py
        │   └── rl_test.py
        ├── neural_nets
        │   ├── models
        │   ├── rec_net.py
        │   └── utility_functions.py
        ├── showcase
        │   └── rl_agent_plots_jan.py
        ├── stable_baselines_code
        │   ├── callback.py
        │   ├── environment.py
        │   ├── example_usage_environment.py
        │   └── reward_functions.py
        ├── train_reconstruction
        │   ├── trainer.py
        │   └── tuner.py
        └── util_functions.py
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
	<details> <!-- deprecated Submodule -->
		<summary><b>deprecated</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/rl_time_tester.py'>rl_time_tester.py</a></b></td>
				<td>- Implement a script that tests reinforcement learning performance using a neural network model<br>- The script loads pre-trained models and datasets, initializes the environment, and runs the RL agent through a series of actions<br>- The primary goal is to evaluate the model's behavior and performance in a simulated environment.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/stuff.txt'>stuff.txt</a></b></td>
				<td>Identify and list the best reward indexes and corresponding rewards from the provided data in the deprecated/stuff.txt file.</td>
			</tr>
			</table>
			<details>
				<summary><b>train_from_dict_2024-06-22_03-16-14</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/experiment_state-2024-06-22_03-16-14.json'>experiment_state-2024-06-22_03-16-14.json</a></b></td>
						<td>- The provided code file, located at `deprecated/train_from_dict_2024-06-22_03-16-14/experiment_state-2024-06-22_03-16-14.json`, plays a crucial role in managing trial data within the project architecture<br>- It facilitates the storage and retrieval of trial-specific information essential for the experiment's state management<br>- This file serves as a key component in tracking and analyzing trial outcomes, contributing significantly to the project's overall functionality and data handling capabilities.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/basic-variant-state-2024-06-22_03-16-14.json'>basic-variant-state-2024-06-22_03-16-14.json</a></b></td>
						<td>- The code file defines training configurations for an open-source project<br>- It specifies parameters like algorithm choice, stopping criteria, resource allocation, and checkpoint settings<br>- This file plays a crucial role in orchestrating the training process by providing essential setup details for running experiments effectively within the project's architecture.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/.validate_storage_marker'>.validate_storage_marker</a></b></td>
						<td>Enables validation of storage markers within the project architecture, ensuring data integrity and consistency.</td>
					</tr>
					</table>
					<details>
						<summary><b>train_from_dict_05210_00001_1_batch_size=2,channels=32,depth=9,lr=0.0003_2024-06-22_03-16-15</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/train_from_dict_05210_00001_1_batch_size=2,channels=32,depth=9,lr=0.0003_2024-06-22_03-16-15/result.json'>result.json</a></b></td>
								<td>- Implement a model training process from a dictionary input, generating results in a JSON file<br>- This code file plays a crucial role in the project's architecture by enabling the training of models based on specified parameters and storing the results for analysis and evaluation.</td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/train_from_dict_05210_00001_1_batch_size=2,channels=32,depth=9,lr=0.0003_2024-06-22_03-16-15/events.out.tfevents.1719018984.daniel-MS-7A38'>events.out.tfevents.1719018984.daniel-MS-7A38</a></b></td>
								<td>Facilitates training neural networks from dictionary data, capturing events for analysis.</td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/train_from_dict_05210_00001_1_batch_size=2,channels=32,depth=9,lr=0.0003_2024-06-22_03-16-15/params.json'>params.json</a></b></td>
								<td>- Extracts hyperparameters for a specific training session from a JSON file<br>- This information is crucial for configuring the training process within the project architecture.</td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/train_from_dict_05210_00001_1_batch_size=2,channels=32,depth=9,lr=0.0003_2024-06-22_03-16-15/error.txt'>error.txt</a></b></td>
								<td>- Handles training data for neural networks, utilizing a custom dataset structure<br>- The code interacts with the project's data loading components, ensuring seamless access to training samples.</td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>train_from_dict_05210_00000_0_batch_size=16,channels=32,depth=7,lr=0.0071_2024-06-22_03-16-15</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/train_from_dict_05210_00000_0_batch_size=16,channels=32,depth=7,lr=0.0071_2024-06-22_03-16-15/result.json'>result.json</a></b></td>
								<td>- Improve model training efficiency by utilizing a dictionary-based approach<br>- This code file enhances the architecture by enabling training from dictionary data, optimizing performance and resource utilization.</td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/train_from_dict_05210_00000_0_batch_size=16,channels=32,depth=7,lr=0.0071_2024-06-22_03-16-15/params.json'>params.json</a></b></td>
								<td>Extracts hyperparameters for a specific training session from a JSON file.</td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/train_from_dict_05210_00000_0_batch_size=16,channels=32,depth=7,lr=0.0071_2024-06-22_03-16-15/events.out.tfevents.1719018978.daniel-MS-7A38'>events.out.tfevents.1719018978.daniel-MS-7A38</a></b></td>
								<td>Enables training neural networks from dictionary data, capturing events for analysis.</td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/train_from_dict_05210_00000_0_batch_size=16,channels=32,depth=7,lr=0.0071_2024-06-22_03-16-15/error.txt'>error.txt</a></b></td>
								<td>Handle data loading errors in the neural network training process to prevent file not found exceptions, ensuring smooth execution of the training pipeline within the project architecture.</td>
							</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
			<details>
				<summary><b>notebooks</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/notebooks/Reconstruction.ipynb'>Reconstruction.ipynb</a></b></td>
						<td>- Summary:
The code file "Reconstruction.ipynb" in the "deprecated/notebooks" directory of the project focuses on the reconstruction aspect, likely related to data or model reconstruction<br>- It plays a crucial role in the project's architecture by handling the process of reconstructing specific components, contributing to the overall functionality and data flow within the codebase.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/notebooks/debug_gpu_not_available.ipynb'>debug_gpu_not_available.ipynb</a></b></td>
						<td>- Debug GPU availability and details in the deprecated notebook to verify CUDA support and GPU information for PyTorch operations<br>- The code checks PyTorch version, CUDA availability, prints CUDA version, number of GPUs, and GPU details if available<br>- It ensures proper GPU utilization for enhanced performance in the project's machine learning workflows.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/notebooks/example_usage_trainer_PARAM_SEARCH.ipynb'>example_usage_trainer_PARAM_SEARCH.ipynb</a></b></td>
						<td>- The code file `example_usage_trainer_PARAM_SEARCH.ipynb` provides an illustrative demonstration of how to utilize the trainer class within the project<br>- It showcases a practical example of how the trainer class can be effectively employed, serving as a reference point for developers looking to leverage this component within the codebase architecture.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/notebooks/Reconstruction_2.ipynb'>Reconstruction_2.ipynb</a></b></td>
						<td>- The code file `Reconstruction_2.ipynb` in the `deprecated/notebooks` directory facilitates automatic reloading of code changes during development<br>- This functionality ensures that the codebase stays up-to-date with any modifications made, enhancing the efficiency of the development process within the project architecture.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/notebooks/example_usage_trainer.ipynb'>example_usage_trainer.ipynb</a></b></td>
						<td>- The code file `example_usage_trainer.ipynb` provides an illustrative demonstration of how to utilize the trainer class within the project<br>- It showcases the practical application of the trainer functionality, offering a clear guide on how to interact with this essential component of the codebase architecture.</td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>neural_nets</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/neural_nets/trainer.py'>trainer.py</a></b></td>
						<td>- Facilitates neural network training by instantiating models, optimizers, and dataloaders based on hyperparameters<br>- Logs progress using TensorBoard and leverages Ray for hyperparameter search<br>- The class aims to streamline training processes and prevent redundant code for managing neural network training tasks within the project architecture.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/neural_nets/weight_inits.py'>weight_inits.py</a></b></td>
						<td>- Initialize neural network weights using Kaiming and Xavier methods for Convolutional and Linear layers, respectively<br>- Ensure proper initialization for both weights and biases to improve model training and convergence.</td>
					</tr>
					</table>
					<details>
						<summary><b>models</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/deprecated/neural_nets/models/unet.py'>unet.py</a></b></td>
								<td>- Implements a UNet neural network with adaptable depth and configurable parameters for image segmentation tasks<br>- The code defines encoder and decoder blocks, along with the forward pass logic for processing input images through the network architecture.</td>
							</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- outputs Submodule -->
		<summary><b>outputs</b></summary>
		<blockquote>
			<details>
				<summary><b>plots_plakat</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/rl_functions_train_plot.py'>rl_functions_train_plot.py</a></b></td>
						<td>- Generates training plots for reinforcement learning functions<br>- Visualizes data from observation and reward directories.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/color_shemes.py'>color_shemes.py</a></b></td>
						<td>Define the primary color scheme used for image channels in the project's plot outputs.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/reconstruction_net_plots.py'>reconstruction_net_plots.py</a></b></td>
						<td>Generates visual plots for the reconstruction network in the project, aiding in the visualization of data processing and model performance.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/rl_agent_plots.py'>rl_agent_plots.py</a></b></td>
						<td>- Generates plots showcasing reinforcement learning agent performance using Stable Baselines3<br>- Utilizes a custom environment with complex reward functions and neural networks<br>- Supports termination based on the number of successful generations<br>- Saves plots for each iteration and generation in a specified directory.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/rl_functions_performance_plot.py'>rl_functions_performance_plot.py</a></b></td>
						<td>- Generates performance plots for various RL models based on statistics data<br>- Determines mean and standard deviation, plots accuracy over grasping points or steps, and saves the plots as images<br>- Displays model performance comparison and highlights key metrics.</td>
					</tr>
					</table>
					<details>
						<summary><b>plot_data</b></summary>
						<blockquote>
							<details>
								<summary><b>diff_after_free</b></summary>
								<blockquote>
									<details>
										<summary><b>obs500k-diff_reward_from_punish_miss_free_rays____from_rl_models</b></summary>
										<blockquote>
											<details>
												<summary><b>punish_miss_free_rays</b></summary>
												<blockquote>
													<details>
														<summary><b>obs500k7.zip_0</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/plot_data/diff_after_free/obs500k-diff_reward_from_punish_miss_free_rays____from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721502278.rl-trainer-2.198617.6'>events.out.tfevents.1721502278.rl-trainer-2.198617.6</a></b></td>
																<td>- The provided code file generates visual plots to analyze the difference in rewards after freeing rays in a reinforcement learning environment with 500k observations<br>- This analysis aids in understanding the impact of freeing rays on reward outcomes, contributing to the broader architecture's evaluation and decision-making process.</td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/plot_data/diff_after_free/obs500k-diff_reward_from_punish_miss_free_rays____from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721506581.rl-trainer-2.198617.7'>events.out.tfevents.1721506581.rl-trainer-2.198617.7</a></b></td>
																<td>- The provided code file generates visual plots illustrating the difference in rewards resulting from punishing missed free rays in a reinforcement learning model<br>- This analysis contributes to understanding the impact of this specific modification on the model's performance within the broader architecture of the codebase.</td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/plot_data/diff_after_free/obs500k-diff_reward_from_punish_miss_free_rays____from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721478002.rl-trainer-2.198617.0'>events.out.tfevents.1721478002.rl-trainer-2.198617.0</a></b></td>
																<td>- The provided code file generates visual plots to analyze the difference in rewards after freeing rays in a reinforcement learning environment with 500k observations<br>- This analysis helps in understanding the impact of freeing rays on the overall reward distribution, contributing to the project's architecture by providing insights into the effectiveness of this action within the system.</td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/plot_data/diff_after_free/obs500k-diff_reward_from_punish_miss_free_rays____from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721498232.rl-trainer-2.198617.5'>events.out.tfevents.1721498232.rl-trainer-2.198617.5</a></b></td>
																<td>- The provided code file generates visual plots illustrating the difference in rewards resulting from missed free rays in an observational dataset of 500k samples, compared to the rewards from punishing missed free rays in reinforcement learning models<br>- This visualization aids in understanding the impact of different reward mechanisms on the dataset, contributing to the overall architecture's analysis and decision-making processes.</td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/plot_data/diff_after_free/obs500k-diff_reward_from_punish_miss_free_rays____from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721494187.rl-trainer-2.198617.4'>events.out.tfevents.1721494187.rl-trainer-2.198617.4</a></b></td>
																<td>- The provided code file generates plots illustrating the difference in rewards after freeing rays, contributing to the visualization of reward variations in the project's architecture<br>- This visualization aids in understanding the impact of freeing rays on rewards, enhancing the overall comprehension of the project's dynamics.</td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/plot_data/diff_after_free/obs500k-diff_reward_from_punish_miss_free_rays____from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721490141.rl-trainer-2.198617.3'>events.out.tfevents.1721490141.rl-trainer-2.198617.3</a></b></td>
																<td>- The provided code file generates visual plots illustrating the difference in rewards resulting from missed free rays in an RL model<br>- This analysis contributes to understanding the impact of missed free rays on the model's performance, aiding in optimizing the reinforcement learning algorithm.</td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/plot_data/diff_after_free/obs500k-diff_reward_from_punish_miss_free_rays____from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721486095.rl-trainer-2.198617.2'>events.out.tfevents.1721486095.rl-trainer-2.198617.2</a></b></td>
																<td>- The provided code file generates visual plots illustrating the difference in rewards after freeing rays, contributing to the analysis of model performance in the project's reinforcement learning architecture<br>- This visualization aids in understanding the impact of freeing rays on rewards, enhancing insights into the model's behavior and performance.</td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/plot_data/diff_after_free/obs500k-diff_reward_from_punish_miss_free_rays____from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721482049.rl-trainer-2.198617.1'>events.out.tfevents.1721482049.rl-trainer-2.198617.1</a></b></td>
																<td>- The provided code file generates visual plots illustrating the difference in rewards resulting from missed free rays in an observation dataset of 500k samples<br>- This analysis contributes to the project's architecture by providing insights into the impact of missed free rays on reward outcomes, aiding in the optimization of the overall system's performance.</td>
															</tr>
															</table>
														</blockquote>
													</details>
												</blockquote>
											</details>
										</blockquote>
									</details>
								</blockquote>
							</details>
							<details>
								<summary><b>complex_after_free</b></summary>
								<blockquote>
									<details>
										<summary><b>obs500k-complex_reward_from_punish_miss_free_rays__from_rl_models</b></summary>
										<blockquote>
											<details>
												<summary><b>punish_miss_free_rays</b></summary>
												<blockquote>
													<details>
														<summary><b>obs500k7.zip_0</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/plot_data/complex_after_free/obs500k-complex_reward_from_punish_miss_free_rays__from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721497602.rl-trainer-2.197317.5'>events.out.tfevents.1721497602.rl-trainer-2.197317.5</a></b></td>
																<td>- The provided code file generates complex reward plots based on data from reinforcement learning models, contributing to the visualization and analysis of the project's performance and decision-making processes<br>- This visualization aids in understanding the impact of rewards and punishments on the model's behavior, enhancing insights into the project's architecture and outcomes.</td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/plot_data/complex_after_free/obs500k-complex_reward_from_punish_miss_free_rays__from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721493556.rl-trainer-2.197317.4'>events.out.tfevents.1721493556.rl-trainer-2.197317.4</a></b></td>
																<td>- The provided code file generates complex reward plots based on data from a reinforcement learning model in the project's architecture<br>- It visualizes the rewards obtained from punishing missed free rays, contributing to the project's overall analysis and decision-making processes.</td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/plot_data/complex_after_free/obs500k-complex_reward_from_punish_miss_free_rays__from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721477546.rl-trainer-2.197317.0'>events.out.tfevents.1721477546.rl-trainer-2.197317.0</a></b></td>
																<td>- The provided code file generates visual plots for complex reward data in the project's architecture, enhancing the understanding of reward dynamics in the system<br>- This visualization component plays a crucial role in analyzing and interpreting the impact of rewards on the overall system behavior.</td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/plot_data/complex_after_free/obs500k-complex_reward_from_punish_miss_free_rays__from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721481418.rl-trainer-2.197317.1'>events.out.tfevents.1721481418.rl-trainer-2.197317.1</a></b></td>
																<td>- The provided code file generates visual plots for complex reward data in the project's architecture, specifically focusing on observations related to free rays after a complex event<br>- This functionality enhances the project's visualization capabilities, providing insights into reward dynamics following certain actions.</td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/plot_data/complex_after_free/obs500k-complex_reward_from_punish_miss_free_rays__from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721505693.rl-trainer-2.197317.7'>events.out.tfevents.1721505693.rl-trainer-2.197317.7</a></b></td>
																<td>- The provided code file generates visual plots depicting reward data from complex scenarios in the project's architecture<br>- These plots help analyze and understand the impact of punishing missed free rays on rewards in a complex environment.</td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/plot_data/complex_after_free/obs500k-complex_reward_from_punish_miss_free_rays__from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721501647.rl-trainer-2.197317.6'>events.out.tfevents.1721501647.rl-trainer-2.197317.6</a></b></td>
																<td>- The provided code file generates visual plots depicting complex reward data after freeing rays in a simulation with 500k observations<br>- This functionality contributes to the project's architecture by providing insights into the impact of freeing rays on complex reward dynamics, aiding in the analysis and understanding of the simulation outcomes.</td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/plot_data/complex_after_free/obs500k-complex_reward_from_punish_miss_free_rays__from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721485464.rl-trainer-2.197317.2'>events.out.tfevents.1721485464.rl-trainer-2.197317.2</a></b></td>
																<td>- The provided code file generates complex reward plots based on data from reinforcement learning models in the project<br>- It visualizes the impact of punishing missed free rays on the overall reward system, contributing to a deeper understanding of the model's behavior and performance.</td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration.git/blob/master/outputs/plots_plakat/plot_data/complex_after_free/obs500k-complex_reward_from_punish_miss_free_rays__from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721489511.rl-trainer-2.197317.3'>events.out.tfevents.1721489511.rl-trainer-2.197317.3</a></b></td>
																<td>- Summary:
The provided code file generates complex reward plots based on data from a reinforcement learning model in the project's architecture<br>- It visualizes the impact of punishing missed free rays on the reward system, helping to analyze and optimize the model's performance.</td>
															</tr>
															</table>
														</blockquote>
													</details>
												</blockquote>
											</details>
										</blockquote>
									</details>
								</blockquote>
							</details>
						</blockquote>
					</details>
				</blockquote>
			</details>
		</blockquote>
	</details>
</details>

---





















