"""Thompson Sampling

This script contains two classes: SlotMachine() and ThompsonSampling().

SlotMachine() is a bernulli slot machine that returns a random reward
when played.

ThompsonSampling() is an agent that using the Thompson Sampling method
tries to estimate the probabilities of the SlotMachine()s given in
input and to get the highest reward.
"""

from typing import Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SlotMachine():
    """This is a "slot machine", when played it returns a reward.
    The reward can be 1 with probability p and 0 with probability 1-p.

    Args:
        p: Probability to return a positive reward when played.
        seed: seed for random generator.
    """

    p: float
    seed: Optional[int] = None

    def __post_init__(self):
        # Instantiate a random generator, so that each object is independent and deterministic (if seed is given)
        self.random_generator = np.random.default_rng(self.seed)

    def play(self) -> int:
        """Play one round of slot machine.

        Returns:
            1 if the reward is obtained, 0 otherwise.
        """

        # Positive reward returned with probability p
        return int(self.random_generator.uniform() < self.p)


class ThompsonSampling():
    """This is a "slot machine", when played it returns a reward.
    The reward can be 1 with probability p and 0 with probability 1-p.

    Args:
        slots: list of slot machines with unknown probabilities.
        seed: seed for random generator.
        alphas_prior: alpha values for the beta distributions used as prior
        betas_prior: alpha values for the beta distributions used as prior
    """

    def __init__(self,
        slots: list[SlotMachine],
        seed: Optional[int] = None,
        alphas_prior: Optional[np.ndarray] = None,
        betas_prior: Optional[np.ndarray] = None
    ) -> None:

        self.slots: list[SlotMachine] = slots
        self.seed: Optional[int] = seed

        # If priors are not provided set alpha and beta of all slots to 1, so to have a uniform distribution between 0 and 1
        self.alphas: np.ndarray = np.ones(len(self.slots)) if alphas_prior is None else alphas_prior
        self.betas: np.ndarray = np.ones(len(self.slots)) if betas_prior is None else betas_prior

        # Instantiate a random generator, so that each object is independent and deterministic (if seed is given)
        self.random_generator = np.random.default_rng(self.seed)

    def act(self) -> tuple[int, int]:
        """Decide which slot to play using the Thompson sampling strategy.
        Plays the slot and updates the relative probability distribution.

        Args:
            slot_index: index of which slot to play.

        Returns:
            index of the slot chosen and reward obtained.
        """

        # Sample a point from each distribution and choose the slot corresponding to the highest value
        chosen_slot = int(np.argmax(self.random_generator.beta(self.alphas, self.betas)))

        return chosen_slot, self.play(chosen_slot)

    def play(self, slot_index: int) -> int:
        """Play one round of one of a given slot machine.
        Updates the posterior probability using the new information given by the slot.

        Args:
            slot_index: index of which slot to play.

        Returns:
            reward of the slot.
        """

        reward = self.slots[slot_index].play()
        self.alphas[slot_index] += reward
        self.betas[slot_index] += 1 - reward
        return reward

    def get_means(self) -> np.ndarray:
        """Compute the means of all the beta distributions.

        Returns:
            array of distribution's means.
        """

        # Means computed using analytical formula for beta distribution
        return self.alphas / (self.alphas + self.betas)

    def get_stds(self) -> np.ndarray:
        """Compute the standard deviations of all the beta distributions.

        Returns:
            array of distribution's stds.
        """

        # Standard deviations computed using analytical formula for beta distribution
        return np.sqrt(self.alphas * self.betas / (np.square(self.alphas + self.betas) * (self.alphas + self.betas + 1)))
