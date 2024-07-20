import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

from util_functions import add_zero_channel, convert_for_imshow, from_torch


class TensorboardCallback(BaseCallback):
    """



    A custom callback that derives from ``BaseCallback``.
    This Callback Class adds shit to tensorboard

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self._log_freq = 10  # log every n calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(
            formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

        self.ready_for_log = True

        self.rewards = []
        self.rec_losses = []
        self.metrics = []


    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """

        # You can have access to info from the env using self.locals.
        # for instance, when using one env (index 0 of locals["infos"]):
        # lap_count = self.locals["infos"][0]["lap_count"]
        # self.tb_formatter.writer.add_scalar("train/lap_count", lap_count, self.num_timesteps)

        if self.num_timesteps % self._log_freq == 0:
            self.ready_for_log = True

        done = self.locals['dones'][-1]
        if done:
            self.rewards.append(np.sum(self.locals['rewards']))
            infos = self.locals['infos'][-1]
            self.rec_losses.append(infos['losses'][-1])
            self.metrics.append(infos['metrics'][-1])

        if self.ready_for_log and done:
            self.ready_for_log = False

            # log reconstruction
            reward = np.mean(self.rewards)
            self.rewards = []
            rec_loss = np.mean(self.rec_losses)
            self.rec_losses = []
            metrics = np.mean(self.metrics)
            self.metrics = []

            self.tb_formatter.writer.add_scalar('custom/reward', np.mean(reward), self.num_timesteps)
            self.tb_formatter.writer.add_scalar('custom/rec_loss', np.mean(rec_loss), self.num_timesteps)
            self.tb_formatter.writer.add_scalar('custom/metrics', np.mean(metrics), self.num_timesteps)
            self.tb_formatter.writer.flush()

            # log images
            #print(self.training_env.envs[0].observation)
            #self.training_env.render()
            #print(self.training_env.envs[0].total_steps)
            #observation = from_torch(self.locals['obs_tensor'])

            #observation_b = self.training_env.envs[0].observation
            #observation_b = self.model.get_env().envs[0].observation,
            #print("diff: ", np.average(np.sqrt(observation_b-observation)))
            #reconstruction = observation[-1]
            #outline = infos['outline']

            #self.writer.add_image('Combined Image', img_tensor_1, self.num_timesteps)

            #obseration_img = add_zero_channel(observation)
            #self.tb_formatter.writer.add_image('images/Observation Image', obseration_img, self.num_timesteps)
        return True


    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


"""

def render(self, mode='human'):
    
    if len(self.grasp_points) > 0:
        gpa = np.array(self.grasp_points)
        self.ax_1.plot(gpa[-1][0], gpa[-1][1], 'ro', label='Last Grasp Point')
        self.ax_1.scatter(gpa[0:-1, 0], gpa[0:-1, 1], s=10, c='orange')

    self.ax_1.plot(self.c_rr, self.c_cc, 'b.', markersize=1)

    self.ax_1.plot(self.rc_points[0, 0], self.rc_points[0, 1], 'go', label='Alpha')
    self.ax_1.plot(self.rc_points[1, 0], self.rc_points[1, 1], 'bo', label='Beta')
    self.ax_1.scatter(self.rc_line[0], self.rc_line[1], s=.5, c='r')

    self.ax_1.legend()

    self.ax_2.clear()
    if self.observation_1D:
        self.ax_2.imshow(self.convert_for_imshow(self.observation))
    else:
        self.ax_2.imshow(self.convert_for_imshow(self.add_zero_channel(self.observation)))
    self.fig.canvas.draw()
    plt.pause(.1)

        return True



"""