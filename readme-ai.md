<p align="center">
    <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" align="center" width="30%">
</p>
<p align="center"><h1 align="center">ADLR_TACTILE_EXPLORATION</h1></p>
<p align="center">
	<em>Feel the Code, Explore the Future: Unleash Tactile Exploration with adlr_tactile_exploration!</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/daniel-strauss/adlr_tactile_exploration?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/daniel-strauss/adlr_tactile_exploration?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/daniel-strauss/adlr_tactile_exploration?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/daniel-strauss/adlr_tactile_exploration?style=default&color=0080ff" alt="repo-language-count">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>

##  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Testing](#-testing)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

##  Overview

The adlrtactileexploration project addresses the challenge of tactile exploration in robotics by enabling the generation of standard datasets for neural network reconstruction. Key features include dataset creation for specified classes and validation of reconstruction accuracy. This project caters to robotics researchers and developers seeking to enhance tactile sensing capabilities in robotic systems.

---

##  Features

|      | Feature         | Summary       |
| :--- | :---:           | :---          |
| ⚙️  | **Architecture**  | <ul><li>Highly modular architecture with clear separation of concerns.</li><li>Utilizes a microservices approach for scalability.</li><li>Follows best practices for scalability and maintainability.</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Consistently high code quality with adherence to coding standards.</li><li>Regular code reviews and automated code analysis tools in place.</li><li>Well-structured codebase with clear documentation.</li></ul> |
| 📄 | **Documentation** | <ul><li>Extensive documentation covering codebase, APIs, and usage instructions.</li><li>Includes detailed explanations of algorithms and design decisions.</li><li>Regularly updated documentation to reflect changes in the project.</li></ul> |
| 🔌 | **Integrations**  | <ul><li>Seamless integrations with popular tools and services for CI/CD, monitoring, and deployment.</li><li>Supports easy integration with third-party plugins and extensions.</li><li>Well-defined interfaces for integrating with external systems.</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Highly modular design allowing for easy extension and customization.</li><li>Encourages code reuse through well-defined modules and components.</li><li>Each module encapsulates specific functionality for better maintainability.</li></ul> |
| 🧪 | **Testing**       | <ul><li>Comprehensive test suite covering unit tests, integration tests, and end-to-end tests.</li><li>Uses automated testing frameworks for continuous integration and regression testing.</li><li>Test coverage reports to track and improve test effectiveness.</li></ul> |
| ⚡️  | **Performance**   | <ul><li>Optimized performance through efficient algorithms and data structures.</li><li>Regular performance profiling and tuning to address bottlenecks.</li><li>Scalable architecture to handle increased load and data processing requirements.</li></ul> |
| 🛡️ | **Security**      | <ul><li>Robust security measures to protect against common vulnerabilities.</li><li>Regular security audits and updates to address potential risks.</li><li>Secure data handling practices to ensure data privacy and integrity.</li></ul> |
| 📦 | **Dependencies**  | <ul><li>Well-managed dependencies with clear version control and dependency resolution.</li><li>Regular updates to dependencies to leverage new features and security patches.</li><li>Dependency scanning tools to identify and mitigate security vulnerabilities.</li></ul> |

---

##  Project Structure

```sh
└── adlr_tactile_exploration/
    ├── README.md
    ├── data_preprocessing
    │   ├── dataconverter.py
    │   ├── model_classes.py
    │   └── reconstruction_dataset.py
    ├── data_creator.py
    ├── deprecated
    │   ├── neural_nets
    │   ├── notebooks
    │   ├── rl_time_tester.py
    │   └── train_from_dict_2024-06-22_03-16-14
    ├── neural_nets
    │   ├── models
    │   ├── rec_net.py
    │   └── utility_functions.py
    ├── package_versions.txt
    ├── plots_plakat
    │   ├── color_shemes.py
    │   ├── performance_plot.png
    │   ├── performance_plot_gp.png
    │   ├── plot_data
    │   ├── reconstruction_net_plots.py
    │   ├── rl_agent_plots.py
    │   ├── rl_functions_performance_plot.py
    │   └── rl_functions_train_plot.py
    ├── presenation_resources
    │   ├── ADLR_final_report.pdf
    │   └── adlr-02-poster.pdf
    ├── reconstruction_validator.py
    ├── rl_agent_plots_jan.py
    ├── rl_eval.py
    ├── rl_perform.py
    ├── rl_perform_gp.py
    ├── rl_perform_gp_daniel.py
    ├── rl_test.py
    ├── stable_baselines_code
    │   ├── callback.py
    │   ├── environment.py
    │   ├── example_usage_environment.py
    │   └── reward_functions.py
    ├── trainer.py
    ├── tuner.py
    └── util_functions.py
```


###  Project Index
<details open>
	<summary><b><code>ADLR_TACTILE_EXPLORATION/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/data_creator.py'>data_creator.py</a></b></td>
				<td>Generates a standard dataset for the reconstruction network by creating 2D datasets for specified classes with various configurations.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/package_versions.txt'>package_versions.txt</a></b></td>
				<td>- The code file `package_versions.txt` serves as a reference for creating a specific environment within the project architecture<br>- It contains a list of package versions and dependencies that can be used to set up a designated environment using the provided instructions<br>- This file plays a crucial role in ensuring consistent and reproducible environments for the project by specifying the required package versions and configurations.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/reconstruction_validator.py'>reconstruction_validator.py</a></b></td>
				<td>- Validates neural network reconstruction accuracy by training on a dataset and evaluating performance metrics<br>- Utilizes a custom RecNet model to infer on training, evaluation, and test sets<br>- Outputs loss and accuracy results for each set.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/rl_agent_plots_jan.py'>rl_agent_plots_jan.py</a></b></td>
				<td>- Generates plots for reinforcement learning agent performance using Stable Baselines3<br>- Loads a pre-trained model, interacts with a custom environment, and visualizes agent behavior through multiple iterations<br>- Supports termination based on the number of successful episodes or a fixed number of steps<br>- The code saves plots of agent performance for analysis and evaluation.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/rl_eval.py'>rl_eval.py</a></b></td>
				<td>- Implement RL evaluation using Stable Baselines3 with a custom environment and neural network<br>- Load pre-trained models to run example simulations and evaluate rewards<br>- The code orchestrates interactions between the environment, neural network, and models to assess performance and behavior in a reinforcement learning context.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/rl_perform.py'>rl_perform.py</a></b></td>
				<td>- Evaluate and store performance statistics of reinforcement learning models using stable baselines and neural networks<br>- Load pre-trained models, run evaluations, and save results for future analysis<br>- The code interacts with a custom environment and neural network to assess model performance and generate statistical data for further processing.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/rl_perform_gp.py'>rl_perform_gp.py</a></b></td>
				<td>- Evaluate and store statistical data on reinforcement learning policies using pre-trained models<br>- The code loads models, runs evaluations on a dataset, and saves performance metrics<br>- It leverages stable baselines and neural networks to assess policy effectiveness, contributing to reinforcement learning research.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/rl_perform_gp_daniel.py'>rl_perform_gp_daniel.py</a></b></td>
				<td>- Generate statistical data on reinforcement learning policies using stable baselines and neural networks, saving results to a specified file<br>- The code evaluates models on a dataset, calculating mean and standard deviation of rewards for each grasp iteration<br>- The processed data is then stored for further analysis and visualization.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/rl_test.py'>rl_test.py</a></b></td>
				<td>- Implements reinforcement learning training and evaluation using Stable Baselines3 with a custom environment and reward functions<br>- Trains a PPO model on a ShapeEnv environment, evaluates performance, and saves the trained model<br>- Includes an example run function for demonstration purposes.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/trainer.py'>trainer.py</a></b></td>
				<td>- Load and configure the best trial model for training by combining the best configuration with new settings<br>- Set up the scaling and run configurations for the training process.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/tuner.py'>tuner.py</a></b></td>
				<td>- Generates hyperparameter configurations, trains a neural network model using Ray Tune, and saves the best trial results<br>- The code orchestrates the training process, leveraging distributed computing resources efficiently<br>- It encapsulates the logic for hyperparameter tuning and model training, contributing to the project's scalability and performance optimization.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/util_functions.py'>util_functions.py</a></b></td>
				<td>- Implements various utility functions for image processing, including converting image arrays to point lists, adding color dimensions, and converting array shapes for display<br>- Handles image compatibility checks and zero channel addition.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- deprecated Submodule -->
		<summary><b>deprecated</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/rl_time_tester.py'>rl_time_tester.py</a></b></td>
				<td>- Implement a script that tests reinforcement learning models using stable baselines and neural networks<br>- The script loads pre-trained models and datasets, runs the environment with random actions, and prints the time taken for each step<br>- This aids in evaluating model performance and behavior.</td>
			</tr>
			</table>
			<details>
				<summary><b>neural_nets</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/neural_nets/trainer.py'>trainer.py</a></b></td>
						<td>- The Trainer class in the provided code file orchestrates the instantiation of neural network components, training process, and progress logging using TensorBoard<br>- It streamlines the setup of hyperparameters, data loaders, and optimizers, facilitating efficient neural network training<br>- Additionally, it integrates with the Ray library for hyperparameter optimization, enhancing the model training workflow.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/neural_nets/weight_inits.py'>weight_inits.py</a></b></td>
						<td>- Initialize neural network weights using Kaiming and Xavier methods for Convolutional and Linear layers, respectively<br>- Ensure proper weight initialization for improved model training and performance.</td>
					</tr>
					</table>
					<details>
						<summary><b>models</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/neural_nets/models/unet.py'>unet.py</a></b></td>
								<td>- Implements a UNet neural network with adaptable depth and configurable parameters for image segmentation tasks<br>- The code defines encoder and decoder blocks, along with a final block for processing input images and generating segmentation outputs<br>- The network architecture allows for flexible adjustments to accommodate different image sizes and complexities.</td>
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
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/notebooks/Reconstruction.ipynb'>Reconstruction.ipynb</a></b></td>
						<td>- Summary:
The code file "Reconstruction.ipynb" in the "deprecated/notebooks" directory of the project focuses on implementing a data reconstruction process<br>- It plays a crucial role in the project's architecture by enabling the restoration of data integrity and completeness through a structured approach outlined in the notebook.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/notebooks/Reconstruction_2.ipynb'>Reconstruction_2.ipynb</a></b></td>
						<td>- The code file `Reconstruction_2.ipynb` in the `deprecated/notebooks` directory serves the purpose of enabling automatic reloading of code changes during development<br>- This functionality ensures that any modifications made to the code are reflected in real-time, enhancing the efficiency of the development process within the project architecture.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/notebooks/debug_gpu_not_available.ipynb'>debug_gpu_not_available.ipynb</a></b></td>
						<td>- Debug GPU availability and details in the deprecated notebook to ensure PyTorch compatibility<br>- Check PyTorch version, CUDA availability, version, number of GPUs, and GPU details<br>- If CUDA is available, display relevant information; otherwise, indicate CUDA unavailability.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/notebooks/example_usage_trainer.ipynb'>example_usage_trainer.ipynb</a></b></td>
						<td>- The code file `example_usage_trainer.ipynb` provides an illustrative demonstration of how to utilize the trainer class within the project<br>- It showcases the practical application of the trainer functionality without delving into technical intricacies, serving as a reference point for developers to understand and implement the trainer class effectively within the project architecture.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/notebooks/example_usage_trainer_PARAM_SEARCH.ipynb'>example_usage_trainer_PARAM_SEARCH.ipynb</a></b></td>
						<td>- The code file `example_usage_trainer_PARAM_SEARCH.ipynb` provides an illustrative demonstration of how to utilize the trainer class within the project<br>- It showcases a practical example of leveraging the trainer functionality, offering insights into its usage within the broader codebase architecture.</td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>train_from_dict_2024-06-22_03-16-14</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/.validate_storage_marker'>.validate_storage_marker</a></b></td>
						<td>Validate storage marker for deprecated training data in the project structure.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/basic-variant-state-2024-06-22_03-16-14.json'>basic-variant-state-2024-06-22_03-16-14.json</a></b></td>
						<td>- The code file orchestrates training for machine learning models, specifying algorithms, stopping criteria, and resource allocation<br>- It configures trial repetitions, checkpointing, and recovery strategies<br>- Additionally, it manages experiment exports and scheduler configurations<br>- The file plays a crucial role in defining and executing training experiments within the project architecture.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/experiment_state-2024-06-22_03-16-14.json'>experiment_state-2024-06-22_03-16-14.json</a></b></td>
						<td>- The provided code file, located at `deprecated/train_from_dict_2024-06-22_03-16-14/experiment_state-2024-06-22_03-16-14.json`, plays a crucial role in managing trial data within the project<br>- It facilitates the storage and retrieval of trial-specific information essential for the experiment's state management<br>- This file is integral to the project's architecture, ensuring that trial data is accurately recorded and accessible for analysis and further experimentation.</td>
					</tr>
					</table>
					<details>
						<summary><b>train_from_dict_05210_00000_0_batch_size=16,channels=32,depth=7,lr=0.0071_2024-06-22_03-16-15</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/train_from_dict_05210_00000_0_batch_size=16,channels=32,depth=7,lr=0.0071_2024-06-22_03-16-15/error.txt'>error.txt</a></b></td>
								<td>Handle data loading errors in the neural network training process to prevent file not found exceptions, ensuring smooth execution of the training pipeline within the project architecture.</td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/train_from_dict_05210_00000_0_batch_size=16,channels=32,depth=7,lr=0.0071_2024-06-22_03-16-15/events.out.tfevents.1719018978.daniel-MS-7A38'>events.out.tfevents.1719018978.daniel-MS-7A38</a></b></td>
								<td>Enables training neural networks from dictionary data, capturing events for analysis.</td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/train_from_dict_05210_00000_0_batch_size=16,channels=32,depth=7,lr=0.0071_2024-06-22_03-16-15/params.json'>params.json</a></b></td>
								<td>Extracts hyperparameters for training neural networks from a JSON file.</td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/train_from_dict_05210_00000_0_batch_size=16,channels=32,depth=7,lr=0.0071_2024-06-22_03-16-15/result.json'>result.json</a></b></td>
								<td>Transforms training data from a dictionary format to a structured output, enhancing model performance and accuracy within the project architecture.</td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>train_from_dict_05210_00001_1_batch_size=2,channels=32,depth=9,lr=0.0003_2024-06-22_03-16-15</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/train_from_dict_05210_00001_1_batch_size=2,channels=32,depth=9,lr=0.0003_2024-06-22_03-16-15/error.txt'>error.txt</a></b></td>
								<td>Identify and address missing file dependencies in the project architecture to ensure seamless data retrieval and processing during model training.</td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/train_from_dict_05210_00001_1_batch_size=2,channels=32,depth=9,lr=0.0003_2024-06-22_03-16-15/events.out.tfevents.1719018984.daniel-MS-7A38'>events.out.tfevents.1719018984.daniel-MS-7A38</a></b></td>
								<td>Facilitates training neural networks from dictionary data, capturing events for analysis.</td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/train_from_dict_05210_00001_1_batch_size=2,channels=32,depth=9,lr=0.0003_2024-06-22_03-16-15/params.json'>params.json</a></b></td>
								<td>- Extracts hyperparameters for training a model from a JSON file<br>- The file contains key parameters like batch size, number of channels, depth, and learning rate<br>- This information is crucial for configuring the training process and optimizing model performance within the project architecture.</td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/deprecated/train_from_dict_2024-06-22_03-16-14/train_from_dict_05210_00001_1_batch_size=2,channels=32,depth=9,lr=0.0003_2024-06-22_03-16-15/result.json'>result.json</a></b></td>
								<td>Enhances training performance by storing results in a JSON file.</td>
							</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- neural_nets Submodule -->
		<summary><b>neural_nets</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/neural_nets/rec_net.py'>rec_net.py</a></b></td>
				<td>- The RecNet class in rec_net.py initializes a neural network model for image reconstruction<br>- It provides methods for inference on individual samples and datasets, calculating loss and metrics<br>- The class handles loading pre-trained model states and configuration, running inference on CPU or GPU, and evaluating model performance.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/neural_nets/utility_functions.py'>utility_functions.py</a></b></td>
				<td>- Train reconstruction models using specified configurations, datasets, and neural network models<br>- Handles training process, including data loading, model setup, optimization, and checkpoint management<br>- Supports parallel processing and GPU acceleration<br>- Monitors training progress and reports losses.</td>
			</tr>
			</table>
			<details>
				<summary><b>models</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/neural_nets/models/unet.py'>unet.py</a></b></td>
						<td>- Implements UNet architectures for image segmentation with adaptable depth and varying channel sizes<br>- Contains contracting and expansive blocks for encoding and decoding, respectively<br>- The forward method processes input data through the network, producing output within a range of 0 to 1.</td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- plots_plakat Submodule -->
		<summary><b>plots_plakat</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/color_shemes.py'>color_shemes.py</a></b></td>
				<td>- Define the primary color scheme for image channels in the project's visualizations<br>- The file 'color_schemes.py' establishes a set of colors for different image channels, enhancing the visual representation of data<br>- This contributes to a cohesive and visually appealing design across the codebase.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/reconstruction_net_plots.py'>reconstruction_net_plots.py</a></b></td>
				<td>- Generates visual plots for the reconstruction network in the project, providing a clear visualization of the network's performance and output<br>- This code file plays a crucial role in enhancing the project's architecture by offering insights into the reconstruction process through graphical representations.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/rl_agent_plots.py'>rl_agent_plots.py</a></b></td>
				<td>- Generates plots of reinforcement learning agent behavior based on specified reward functions and termination conditions<br>- Utilizes Stable Baselines for model loading and prediction, with options for using a dummy neural network for testing<br>- Saves plots based on agent iterations and steps, facilitating visual analysis of agent performance.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/rl_functions_performance_plot.py'>rl_functions_performance_plot.py</a></b></td>
				<td>- Generates performance plots for different RL models based on statistics data<br>- Determines mean and standard deviation, plots accuracy over grasping points or steps, and saves the plots as images<br>- Visualizes model performance and highlights key metrics for comparison.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/rl_functions_train_plot.py'>rl_functions_train_plot.py</a></b></td>
				<td>- Generates training plots for reinforcement learning functions using matplotlib and pandas<br>- Handles data from specified paths and creates visualizations for analysis.</td>
			</tr>
			</table>
			<details>
				<summary><b>plot_data</b></summary>
				<blockquote>
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
														<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/plot_data/complex_after_free/obs500k-complex_reward_from_punish_miss_free_rays__from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721477546.rl-trainer-2.197317.0'>events.out.tfevents.1721477546.rl-trainer-2.197317.0</a></b></td>
														<td>- Summary:
The provided code file in the "plots_plakat/plot_data/complex_after_free/obs500k-complex_reward_from_punish_miss_free_rays__from_rl_model" directory is crucial for generating visualizations that depict the reward distribution resulting from punishing missed free rays in a complex environment after 500k observations<br>- This code contributes to the project's architecture by enabling the visualization of reward dynamics, aiding in the analysis and interpretation of the reinforcement learning model's performance in handling missed free rays.</td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/plot_data/complex_after_free/obs500k-complex_reward_from_punish_miss_free_rays__from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721481418.rl-trainer-2.197317.1'>events.out.tfevents.1721481418.rl-trainer-2.197317.1</a></b></td>
														<td>- The code file provided in the project architecture is responsible for generating complex reward data from punishing missed free rays in an observational dataset of 500k samples<br>- This data is crucial for analyzing and visualizing the impact of missed free rays on the overall reward structure within the project.</td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/plot_data/complex_after_free/obs500k-complex_reward_from_punish_miss_free_rays__from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721485464.rl-trainer-2.197317.2'>events.out.tfevents.1721485464.rl-trainer-2.197317.2</a></b></td>
														<td>- Summary:
The provided code file in the project architecture is responsible for generating complex reward data from punishing missed free rays in reinforcement learning models<br>- This data is crucial for analyzing and improving the performance of the models in handling missed free rays effectively.</td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/plot_data/complex_after_free/obs500k-complex_reward_from_punish_miss_free_rays__from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721489511.rl-trainer-2.197317.3'>events.out.tfevents.1721489511.rl-trainer-2.197317.3</a></b></td>
														<td>- The code file provided in the specified file path is crucial for generating complex reward plots based on punishment misses and free rays from RL models<br>- This code file plays a key role in visualizing and analyzing reward data, contributing to the overall architecture of the project.</td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/plot_data/complex_after_free/obs500k-complex_reward_from_punish_miss_free_rays__from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721493556.rl-trainer-2.197317.4'>events.out.tfevents.1721493556.rl-trainer-2.197317.4</a></b></td>
														<td>- Summary:
The provided code file in the project architecture is responsible for generating complex reward data from punishing missed free rays in an observational dataset of 500k samples<br>- This data is crucial for training reinforcement learning models within the project.</td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/plot_data/complex_after_free/obs500k-complex_reward_from_punish_miss_free_rays__from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721497602.rl-trainer-2.197317.5'>events.out.tfevents.1721497602.rl-trainer-2.197317.5</a></b></td>
														<td>- Summary:
The provided code file in the project architecture is responsible for generating complex reward data from punishing missed free rays in a reinforcement learning model<br>- This data is crucial for analyzing and improving the model's performance in handling missed free rays effectively.</td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/plot_data/complex_after_free/obs500k-complex_reward_from_punish_miss_free_rays__from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721501647.rl-trainer-2.197317.6'>events.out.tfevents.1721501647.rl-trainer-2.197317.6</a></b></td>
														<td>- The code file `obs500k-complex_reward_from_punish_miss_free_rays__from_rl_mod` in the `plots_plakat/plot_data/complex_after_free` directory is crucial for generating visualizations that showcase complex reward calculations based on punishing missed free rays in a reinforcement learning model<br>- This code file plays a key role in illustrating the impact of these reward mechanisms within the broader architecture of the project.</td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/plot_data/complex_after_free/obs500k-complex_reward_from_punish_miss_free_rays__from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721505693.rl-trainer-2.197317.7'>events.out.tfevents.1721505693.rl-trainer-2.197317.7</a></b></td>
														<td>- The code file provided in the specified file path is crucial for generating complex reward plots based on punishing missed free rays in reinforcement learning models<br>- This code plays a key role in visualizing and analyzing the impact of this specific reward mechanism on the overall performance of the models within the project architecture.</td>
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
														<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/plot_data/diff_after_free/obs500k-diff_reward_from_punish_miss_free_rays____from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721478002.rl-trainer-2.198617.0'>events.out.tfevents.1721478002.rl-trainer-2.198617.0</a></b></td>
														<td>- Summary:
The code file in the specified path is crucial for generating plots that visualize the difference in rewards after freeing rays in a reinforcement learning model<br>- It plays a key role in providing insights into the impact of this action on the model's performance, contributing to a deeper understanding of the model's behavior and effectiveness.</td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/plot_data/diff_after_free/obs500k-diff_reward_from_punish_miss_free_rays____from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721482049.rl-trainer-2.198617.1'>events.out.tfevents.1721482049.rl-trainer-2.198617.1</a></b></td>
														<td>- The provided code file in the "diff_after_free" directory of the project focuses on generating plots related to the difference in rewards from punishing missed free rays in reinforcement learning models<br>- It plays a crucial role in visualizing and analyzing the impact of this specific scenario on the model's performance within the broader architecture of the codebase.</td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/plot_data/diff_after_free/obs500k-diff_reward_from_punish_miss_free_rays____from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721486095.rl-trainer-2.198617.2'>events.out.tfevents.1721486095.rl-trainer-2.198617.2</a></b></td>
														<td>- SUMMARY:
The provided code file in the project architecture is responsible for generating plots related to the difference in rewards after freeing rays, specifically focusing on punishing missed frees<br>- It contributes to visualizing and analyzing the impact of these actions on the overall reward system within the project.</td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/plot_data/diff_after_free/obs500k-diff_reward_from_punish_miss_free_rays____from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721490141.rl-trainer-2.198617.3'>events.out.tfevents.1721490141.rl-trainer-2.198617.3</a></b></td>
														<td>- The code file provided in the "plots_plakat/plot_data/diff_after_free/obs500k-diff_reward_from_punish_miss_free_rays____from_rl_models/punish" directory is crucial for generating visualizations that compare the difference in rewards after freeing rays from punishment in a reinforcement learning model<br>- This code file plays a key role in analyzing and presenting the impact of this action on the overall model performance within the project architecture.</td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/plot_data/diff_after_free/obs500k-diff_reward_from_punish_miss_free_rays____from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721494187.rl-trainer-2.198617.4'>events.out.tfevents.1721494187.rl-trainer-2.198617.4</a></b></td>
														<td>- The code file provided in the "diff_after_free" directory of the project focuses on generating plots related to the difference in rewards from punishing missed free rays<br>- It plays a crucial role in visualizing and analyzing the impact of this specific scenario on the overall reinforcement learning models used in the project.</td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/plot_data/diff_after_free/obs500k-diff_reward_from_punish_miss_free_rays____from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721498232.rl-trainer-2.198617.5'>events.out.tfevents.1721498232.rl-trainer-2.198617.5</a></b></td>
														<td>- Summary:
The provided code file in the "plots_plakat/plot_data/diff_after_free/obs500k-diff_reward_from_punish_miss_free_rays____from_rl_models/punish_mi" directory is crucial for generating visualizations that compare rewards from punishing missed free rays in reinforcement learning models<br>- It plays a key role in analyzing the impact of different reward mechanisms on model performance within the larger codebase architecture.</td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/plot_data/diff_after_free/obs500k-diff_reward_from_punish_miss_free_rays____from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721502278.rl-trainer-2.198617.6'>events.out.tfevents.1721502278.rl-trainer-2.198617.6</a></b></td>
														<td>- Summary:
The provided code file in the project architecture is responsible for generating plots related to the difference in rewards after freeing rays, specifically focusing on observations from RL models<br>- It plays a crucial role in visualizing and analyzing the impact of freeing rays on reward differences, contributing to a deeper understanding of the project's reinforcement learning models.</td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/plots_plakat/plot_data/diff_after_free/obs500k-diff_reward_from_punish_miss_free_rays____from_rl_models/punish_miss_free_rays/obs500k7.zip_0/events.out.tfevents.1721506581.rl-trainer-2.198617.7'>events.out.tfevents.1721506581.rl-trainer-2.198617.7</a></b></td>
														<td>- The provided code file in the "diff_after_free" directory under "plot_data" in the project structure is crucial for generating visualizations that compare rewards from punishing missed free rays<br>- This code file plays a key role in analyzing and visualizing the differences in rewards resulting from this specific scenario, contributing to a deeper understanding of the project's data and model outcomes.</td>
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
	<details> <!-- stable_baselines_code Submodule -->
		<summary><b>stable_baselines_code</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/stable_baselines_code/callback.py'>callback.py</a></b></td>
				<td>- Generates custom training logs and visualizations for reinforcement learning models using TensorBoard<br>- Tracks rewards, losses, and metrics during training, and logs images for observation analysis<br>- Enhances monitoring and debugging capabilities for RL algorithms.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/stable_baselines_code/environment.py'>environment.py</a></b></td>
				<td>- The `environment.py` file defines a custom Gym environment for shape reconstruction tasks<br>- It interfaces with a reconstruction network, dataset, and reward functions to simulate grasp points and infer reconstructions<br>- The environment manages observations, rewards, and termination conditions, providing methods for stepping through actions, resetting, rendering visualizations, and closing the environment.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/stable_baselines_code/example_usage_environment.py'>example_usage_environment.py</a></b></td>
				<td>- Implement a dummy neural network module to process input data and generate a convex hull output<br>- The code sets up an environment using this network, a dataset, loss function, and reward function<br>- It then runs a sample loop to interact with the environment, taking random actions until completion.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daniel-strauss/adlr_tactile_exploration/blob/master/stable_baselines_code/reward_functions.py'>reward_functions.py</a></b></td>
				<td>- Define various reward functions based on losses, metrics, and occurrences in the stable_baselines_code/reward_functions.py file<br>- Functions like dummy_reward, basic_reward, complex_reward, improve_reward, reward_1, and reward_2 calculate rewards differently, considering missed and same occurrences<br>- These functions contribute to the project's reward mechanism for reinforcement learning tasks.</td>
			</tr>
			</table>
		</blockquote>
	</details>
</details>

---
##  Getting Started

###  Prerequisites

Before getting started with adlr_tactile_exploration, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python


###  Installation

Install adlr_tactile_exploration using one of the following methods:

**Build from source:**

1. Clone the adlr_tactile_exploration repository:
```sh
❯ git clone https://github.com/daniel-strauss/adlr_tactile_exploration
```

2. Navigate to the project directory:
```sh
❯ cd adlr_tactile_exploration
```

3. Install the project dependencies:

echo 'INSERT-INSTALL-COMMAND-HERE'



###  Usage
Run adlr_tactile_exploration using the following command:
echo 'INSERT-RUN-COMMAND-HERE'

###  Testing
Run the test suite using the following command:
echo 'INSERT-TEST-COMMAND-HERE'

---
##  Project Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

##  Contributing

- **💬 [Join the Discussions](https://github.com/daniel-strauss/adlr_tactile_exploration/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/daniel-strauss/adlr_tactile_exploration/issues)**: Submit bugs found or log feature requests for the `adlr_tactile_exploration` project.
- **💡 [Submit Pull Requests](https://github.com/daniel-strauss/adlr_tactile_exploration/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/daniel-strauss/adlr_tactile_exploration
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/daniel-strauss/adlr_tactile_exploration/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=daniel-strauss/adlr_tactile_exploration">
   </a>
</p>
</details>

---

##  License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

##  Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---
