from abc import ABC, abstractclassmethod
import argparse


class Scheduler(ABC):

    def __init__(self,
                 args=None):
        self._last = -1
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
        scheduler.setting()
        return scheduler

    @abstractclassmethod
    def setting(self):
        pass

    def get_last(self) -> float:
        return self._last


    @abstractclassmethod
    def _get_next(self) -> float:
        pass


    def step(self) -> float:
        self._step_count += 1
        if self._step_count > self. _T:
            raise Exception(f"Exceeded the total number of variation")
        self._next = self._get_next()
        self._last = self._next
        return self._next


def get_scheduler_class_from_name(name: str, scheduler: str):
    if scheduler == 'weight':
        from dpsda.schedulers import weight_schedulers
        if name == 'step':
            return weight_schedulers.StepWeight
        elif name == 'exponential':
            return weight_schedulers.ExponentialWeight
        elif name == 'linear':
            return weight_schedulers.LinearWeight
        elif name == 'constant':
            return weight_schedulers.ConstantWeight
        else:
            raise ValueError(f'Unknown scheduler name {name}')
    elif scheduler == 'degree':
        from dpsda.schedulers import degree_schedulers
        if name == 'step':
            return degree_schedulers.StepDeg
        elif name == 'exponential':
            return degree_schedulers.ExponentialDeg
        elif name == 'linear':
            return degree_schedulers.LinearDeg
        elif name == 'constant':
            return degree_schedulers.ConstantDeg
        else:
            raise ValueError(f'Unknown scheduler name {name}')
    else:
        raise ValueError(f'Unknown scheduler name {name}')