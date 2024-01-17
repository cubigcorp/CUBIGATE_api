import logging
from typing import List
from abc import ABC, abstractclassmethod
import argparse



class DegreeScheduler(ABC):

    def __init__(self,
                 args=None):
        self._last_deg = -1
        self._step_count = 0
        self._verbose = True
        self._T = -1
        self.args = args

    @staticmethod
    def command_line_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--scheduler_help',
            action='help')
        return parser


    @classmethod
    def from_command_line_args(cls, args, T, verbose = True):
        """
        Creating the API from command line arguments.

        Args:
            args: (List[str]):
            The command line arguments
        Returns:
            DegreeScheduler:
                The scheduler object.
        """
        args = cls.command_line_parser().parse_args(args)
        scheduler = cls(**vars(args), args=args)
        scheduler._T = T
        scheduler._verbose = verbose
        return scheduler


    def get_last_deg(self) -> float:
        return self._last_deg


    @abstractclassmethod
    def _get_next_deg(self) -> float:
        pass


    def step(self) -> float:
        self._step_count += 1
        if self._step_count > self. _T:
            raise Exception(f"Exceeded the total number of variation")
        self._next_deg = self._get_next_deg()
        if self._verbose:
            logging.info(f"Scheduled degree for {self._step_count}th variation: {self._next_deg}")
        self._last_deg = self._next_deg
        return self._next_deg



class StepDeg(DegreeScheduler):
    def __init__(self,
                 scheduler_base_deg: float,
                 scheduler_gamma: float,
                 scheduler_step_size: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gamma = scheduler_gamma
        self._step_size = scheduler_step_size
        self._last_deg = scheduler_base_deg

    def _get_next_deg(self) -> float:
        if self._step_count % self._step_size == 0:
            return self._last_deg * self._gamma
        else:
            return self._last_deg

    @staticmethod
    def command_line_parser():
        parser = super(
            StepDeg, StepDeg).command_line_parser()
        parser.add_argument(
            '--scheduler_gamma',
            type=float
        )
        parser.add_argument(
            '--scheduler_step_size',
            type=int
        )
        parser.add_argument(
            '--scheduler_base_deg',
            type=float
        )
        return parser



class ExponentialDeg(DegreeScheduler):
    def __init__(self,
                 scheduler_base_deg: float,
                 scheduler_gamma: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gamma = scheduler_gamma
        self._last_deg = scheduler_base_deg

    def _get_next_deg(self) -> float:
        return self._last_deg * self._gamma

    @staticmethod
    def command_line_parser():
        parser = super(
            ExponentialDeg, ExponentialDeg).command_line_parser()
        parser.add_argument(
            '--scheduler_gamma',
            type=float
        )
        parser.add_argument(
            '--scheduler_base_deg',
            type=float
        )
        return parser




class LinearDeg(DegreeScheduler):
    def __init__(self,
                 scheduler_base_deg: float,
                 scheduler_min_deg: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_deg = scheduler_base_deg
        self._step_size = (scheduler_base_deg - scheduler_min_deg) / self._T

    def _get_next_deg(self) -> float:
        return self._last_deg  - self._step_size

    @staticmethod
    def command_line_parser():
        parser = super(
            LinearDeg, LinearDeg).command_line_parser()
        parser.add_argument(
            '--scheduler_min_deg',
            type=float
        )
        parser.add_argument(
            '--scheduler_base_deg',
            type=float
        )
        return parser




class ConstantDeg(DegreeScheduler):
    def __init__(self,
                 scheduler_base_deg: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_deg = scheduler_base_deg
    

    @staticmethod
    def command_line_parser():
        parser = super(
            ExponentialDeg, ExponentialDeg).command_line_parser()
        parser.add_argument(
            '--scheduler_base_deg',
            type=float
        )
        return parser

    def _get_next_deg(self) -> float:
        return self._last_deg


def get_scheduler_class_from_name(name: str):
    if name == 'step':
        return StepDeg
    elif name == 'exponential':
        return ExponentialDeg
    elif name == 'linear':
        return LinearDeg
    elif name == 'constant':
        return ConstantDeg
    else:
        raise ValueError(f'Unknown scheduler name {name}')

