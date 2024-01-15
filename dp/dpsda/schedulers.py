import logging
from typing import List
from abc import ABC, abstractclassmethod
import argparse



class DegreeScheduler(ABC):

    def __init__(self, T: int, base_deg: float, verbose: bool):
        self._T = T
        self._last_deg = base_deg
        self._step_count = 0
        self._verbose = verbose

    @staticmethod
    def command_line_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--scheduler_help',
            action='help')
        return parser


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
                 T: int,
                 scheduler_base_deg: float,
                 scheduler_gamma: float,
                 scheduler_step_size: int,
                 verbose: bool = True):
        super().__init__(T, scheduler_base_deg, verbose)
        self._gamma = scheduler_gamma
        self._step_size = scheduler_step_size

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
                 T: int,
                 scheduler_base_deg: float,
                 scheduler_gamma: float,
                 verbose: bool = True):
        super().__init__(T, scheduler_base_deg, verbose)
        self._gamma = scheduler_gamma

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
                 T: int,
                 scheduler_base_deg: float,
                 scheduler_min_deg: float,
                 verbose: bool = True):
        super().__init__(T, scheduler_base_deg, verbose)
        self._step_size = (scheduler_base_deg - scheduler_min_deg) / T

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
                 T: int,
                 scheduler_base_deg: float,
                 verbose: bool = True):
        super().__init__(T, scheduler_base_deg, verbose)
    

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



def get_scheduler(name: str, args: List, T: int) -> DegreeScheduler:
    scheduler_class = get_scheduler_class_from_name(name)
    args = scheduler_class.command_line_parser().parse_args(args)
    scheduler = scheduler_class(**vars(args), T=T)
    return scheduler
