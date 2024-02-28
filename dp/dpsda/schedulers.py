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
            SchedulerreeScheduler:
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


    @abstractclassmethod
    def _get_next(self) -> float:
        pass



    def set_from_t(self, t: int) -> None:
        while self._step_count <= t:
            self.step()


    def step(self) -> float:
        if self._step_count > self. _T:
            raise Exception(f"Exceeded the number of variation")
        if self._step_count == 0:
            self._next = self._last
        else:
            self._next = self._get_next()
        self._last = self._next
        self._step_count += 1
        return self._next


def get_scheduler_class_from_name(name: str):
    if name == 'step':
        return StepScheduler
    elif name == 'exponential':
        return ExponentialScheduler
    elif name == 'linear':
        return LinearScheduler
    elif name == 'constant':
        return ConstantScheduler
    elif name == 'wlinear':
        return WLinearScheduler
    else:
        raise ValueError(f'Unknown scheduler name {name}')




class StepScheduler(Scheduler):
    def __init__(self,
                 base: float,
                 gamma: float,
                 step_size: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gamma = gamma
        self._step_size = step_size
        self._base = base

    def setting(self):
        self._last = self._base

    def _get_next(self) -> float:
        if (self._step_count % self._step_size == 0) and (self._step_count != 0):
            return self._last * self._gamma
        else:
            return self._last

    @staticmethod
    def command_line_parser():
        parser = super(
            StepScheduler, StepScheduler).command_line_parser()
        parser.add_argument(
            '--gamma',
            type=float
        )
        parser.add_argument(
            '--step_size',
            type=int
        )
        parser.add_argument(
            '--base',
            type=float
        )
        return parser



class ExponentialScheduler(Scheduler):
    def __init__(self,
                 base: float,
                 gamma: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gamma = gamma
        self._base = base


    def setting(self):
        self._last = self._base

    def _get_next(self) -> float:
        return self._last * self._gamma

    @staticmethod
    def command_line_parser():
        parser = super(
            ExponentialScheduler, ExponentialScheduler).command_line_parser()
        parser.add_argument(
            '--gamma',
            type=float
        )
        parser.add_argument(
            '--base',
            type=float
        )
        return parser




class LinearScheduler(Scheduler):
    def __init__(self,
                 base: float,
                 min: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last = base
        self._base = base
        self._min = min

    def _get_next(self) -> float:
        test = self._last  - self._step_size
        return test if test >= 0 else 0


    def setting(self):
        self._step_size = (self._base - self._min) / self._T


    @staticmethod
    def command_line_parser():
        parser = super(
            LinearScheduler, LinearScheduler).command_line_parser()
        parser.add_argument(
            '--min',
            type=float
        )
        parser.add_argument(
            '--base',
            type=float
        )
        return parser




class ConstantScheduler(Scheduler):
    def __init__(self,
                 base: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._base = base


    def setting(self):
        self._last = self._base

    @staticmethod
    def command_line_parser():
        parser = super(
            ExponentialScheduler, ExponentialScheduler).command_line_parser()
        parser.add_argument(
            '--base',
            type=float
        )
        return parser

    def _get_next(self) -> float:
        return self._last



class WLinearScheduler(Scheduler):
    def __init__(self,
                 base: float,
                 warmup: int,
                 min: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last = base
        self._base = base
        self._min = min
        self._warmup = warmup

    def _get_next(self) -> float:
        if self._step_count < self._warmup:
            return self._last
        test = self._last  - self._step_size
        return test if test >= 0 else 0


    def setting(self):
        self._step_size = (self._base - self._min) / self._T


    @staticmethod
    def command_line_parser():
        parser = super(
            LinearScheduler, LinearScheduler).command_line_parser()
        parser.add_argument(
            '--min',
            type=float
        )
        parser.add_argument(
            '--base',
            type=float
        )
        parser.add_argument(
            '--warmup',
            type=int
        )
        return parser
