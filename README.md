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
│   ├── plots_plakat/
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

		<details open>
		  <summary><b>Project Index</b></summary>
		  <ul>
		    <li><b>README.md</b> – Project overview and instructions</li>
		    <li><b>demo.py</b> – Main demo script for running experiments</li>
		    <li><b>package_versions.txt</b> – Environment and dependency specification</li>
		    <li><b>data/</b> – Contains datasets (2D_shapes, train/test/eval splits, object folders)</li>
		    <li><b>docs/</b> – Project documentation, report, poster, and media</li>
		    <li><b>outputs/</b> – Model outputs, plots, trained models, and RL runs</li>
		    <li><b>src/</b> – All source code (data preprocessing, evaluation, neural nets, RL, utilities, etc.)</li>
		    <li><b>__pycache__/</b> – Python cache files (auto-generated)</li>
		  </ul>
		</details>
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





















