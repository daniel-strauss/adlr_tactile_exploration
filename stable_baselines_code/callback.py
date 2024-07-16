import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


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
        self._log_freq = 1000  # log every n calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

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

        self.rewards.append(self.locals['rewards'][-1])
        infos = self.locals['infos'][-1]
        self.rec_losses.append(infos['losses'][-1])
        self.metrics.append(infos['metrics'][-1])


        if self.num_timesteps % self._log_freq == 0:
            # You can have access to info from the env using self.locals.
            # for instance, when using one env (index 0 of locals["infos"]):
            # lap_count = self.locals["infos"][0]["lap_count"]
            # self.tb_formatter.writer.add_scalar("train/lap_count", lap_count, self.num_timesteps)



            reward = np.mean(self.rewards)
            self.rewards = []
            rec_loss = np.mean(self.rec_losses)
            self.rec_losses = []
            metrics = np.mean(self.metrics)
            self.metrics = []

            self.tb_formatter.writer.add_scalar('custom/reward', np.mean(reward),  self.num_timesteps)
            self.tb_formatter.writer.add_scalar('custom/rec_loss', np.mean(rec_loss),  self.num_timesteps)
            self.tb_formatter.writer.add_scalar('custom/metrics', np.mean(metrics),  self.num_timesteps)

            self.tb_formatter.writer.flush()

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