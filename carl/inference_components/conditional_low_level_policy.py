from abc import ABC, abstractmethod
from loguru import logger

import numpy as np
import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from carl.environment.env import GameEnv
from carl.environment.training_goal import TrainingGoal
from carl.inference_components.component import InferenceComponent, TrainingModule


class ConditionalLowLevelPolicy(InferenceComponent):
    @abstractmethod
    def __init__(
        self,
        conditional_low_level_policy_class: type[nn.Module],
        path_to_conditional_low_level_policy_weights: str,
        env: GameEnv,
    ) -> None:
        """
        Initialize the conditional low level policy network.
        params:
            conditional_low_level_policy: the conditional low level policy.
            env: the environment.
        """

        self.conditional_low_level_policy_class = conditional_low_level_policy_class
        self.path_to_conditional_low_level_policy_weights = path_to_conditional_low_level_policy_weights
        self.env = env

    @abstractmethod
    def construct_network(self) -> None:
        """
        Construct the networks.
        """

        raise NotImplementedError

    @abstractmethod
    def get_action(
        self, state: np.ndarray, state_after_k: np.ndarray, **conditional_low_level_policy
    ) -> Tensor:
        """
        Get the action from state to state_after_k.
        params:
            state: the state.
            state_after_k: the state after k steps.
            conditional_low_level_policy: the conditional low level policy.
        return:
            the action.
        """

        raise NotImplementedError


class TransformerConditionalLowLevelPolicy(ConditionalLowLevelPolicy):
    def __init__(
        self,
        conditional_low_level_policy_class: type[PreTrainedModel],
        path_to_conditional_low_level_policy_weights,
        env: GameEnv,
        training_module: TrainingModule | None = None,
    ) -> None:
        super().__init__(
            conditional_low_level_policy_class, path_to_conditional_low_level_policy_weights, env
        )
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cllp: PreTrainedModel | None = None
        self.training_module = training_module

    def get_component_training_module(self) -> TrainingModule | dict[str, TrainingModule] | None:
        return self.training_module

    def construct_network(self) -> None:
        # We do not put the cllp on the eval mode, because "from_pretrained" does it for us.
        # See: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
        self.cllp = self.instantiate_network(
            self.conditional_low_level_policy_class,
            self.path_to_conditional_low_level_policy_weights,
        )
        
    def get_network(self) -> PreTrainedModel | dict[str, PreTrainedModel]:
        return self.cllp

    def get_action(
        self, states: list[np.ndarray], subgoals: list[np.ndarray]) -> Tensor:

        encoded_boards = torch.cat([self.env.tokenizer.x_y_tokenizer(
             x=(state, state_after_k), y=0, training_goal=TrainingGoal.CLLP
        )[0] for state, state_after_k in zip(states, subgoals)])
        encoded_boards = encoded_boards.to(self.device)
        
        with torch.no_grad():
            logits: torch.Tensor = self.cllp(encoded_boards).logits # (BS, NUM_ACTIONS)
        return logits
