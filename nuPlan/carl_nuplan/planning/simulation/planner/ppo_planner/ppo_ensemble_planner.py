from typing import Dict, List, Optional, Type

import torch
import numpy as np
import numpy.typing as npt

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from carl_nuplan.planning.gym.environment.helper.environment_area import RectangleEnvironmentArea
from carl_nuplan.planning.gym.environment.observation_builder.default.default_observation_builder import (
    DefaultObservationBuilder,
)
from carl_nuplan.planning.gym.environment.observation_builder.default.default_renderer import DefaultRenderer
from carl_nuplan.planning.gym.environment.trajectory_builder.action_trajectory_builder import ActionTrajectoryBuilder
from carl_nuplan.planning.gym.policy.ppo.ppo_config import GlobalConfig
from carl_nuplan.planning.gym.policy.ppo.ppo_model import PPOPolicy


class PPOEnsemblePlanner(AbstractPlanner):
    """
    Ensemble planner that uses multiple PPO policies to compute the action.
    TODO: Move observation and trajectory builder to hydra config.
    """

    def __init__(
        self,
        checkpoint_paths: List[str],
        trajectory_sampling: TrajectorySampling = TrajectorySampling(num_poses=80, interval_length=0.1),
        device: str = "cpu",
    ):
        """
        Initializes the PPOEnsemblePlanner.
        :param checkpoint_paths: List of paths to the PPO `.pth` checkpoint files.
        :param trajectory_sampling: TODO: remove, defaults to TrajectorySampling(num_poses=80, interval_length=0.1)
        :param device: literal describing the cuda device for inference, defaults to "cpu"
        """
        assert len(checkpoint_paths) > 0, "At least one checkpoint path must be provided."

        self._checkpoint_paths = checkpoint_paths
        self._trajectory_sampling = trajectory_sampling

        self._config = GlobalConfig()
        self._trajectory_builder = ActionTrajectoryBuilder()

        environment_area = RectangleEnvironmentArea()
        self._observation_builder = DefaultObservationBuilder(
            environment_area, DefaultRenderer(environment_area), inference=True
        )

        self._last_observation: Dict[str, npt.NDArray[np.float32]] = {}
        self._last_action: npt.NDArray[np.float32] = np.array([0.0, 0.0], dtype=np.float32)

        self._device = torch.device(device)
        self._deterministic = True

        # lazy loaded
        self._planner_initialization: Optional[PlannerInitialization] = None
        self._agents: List[PPOPolicy] = []

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """Inherited, see superclass."""

        self._planner_initialization = initialization

        for checkpoint_path in self._checkpoint_paths:
            agent = PPOPolicy(
                self._observation_builder.get_observation_space(),
                self._trajectory_builder.get_action_space(),
                config=self._config,
            ).to(self._device)

            state_dict = torch.load(checkpoint_path, map_location=self._device)
            agent.load_state_dict(state_dict, strict=True)
            agent.to(self._device)
            agent.eval()

            self._agents.append(agent)

        self._image = None

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """Inherited, see superclass."""
        ego_state, _ = current_input.history.current_state

        info = {"last_action": self._last_action}
        obs = self._observation_builder.build_observation(current_input, self._planner_initialization, info)

        obs_tensor = {
            "bev_semantics": torch.Tensor(obs["bev_semantics"][None, ...]).to(self._device, dtype=torch.float32),
            "measurements": torch.Tensor(obs["measurements"][None, ...]).to(self._device, dtype=torch.float32),
            "value_measurements": torch.Tensor(obs["value_measurements"][None, ...]).to(
                self._device, dtype=torch.float32
            ),
        }
        actions: List[npt.NDArray[np.float32]] = []
        for agent_idx in range(len(self._agents)):
            with torch.no_grad():
                (action, _, _, _, _, _, _, _, _, _, _) = self._agents[agent_idx].forward(
                    obs_tensor,
                    deterministic=self._deterministic,
                    lstm_state=None,
                    done=None,
                )

            action = action.squeeze().detach().cpu().numpy()
            actions.append(action)

        action = np.mean(actions, axis=0)

        self._last_observation = obs
        self._last_action = action

        return self._trajectory_builder.build_trajectory(action, ego_state, info)
