from dpsda.schedulers.scheduler import Scheduler


class StepDeg(Scheduler):
    def __init__(self,
                 degree_scheduler_base_deg: float,
                 degree_scheduler_gamma: float,
                 degree_scheduler_step_size: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gamma = degree_scheduler_gamma
        self._step_size = degree_scheduler_step_size
        self._base = degree_scheduler_base_deg

    def setting(self):
        self._last = self._base

    def _get_next(self) -> float:
        if self._step_count % self._step_size == 0:
            return self._last * self._gamma
        else:
            return self._last

    @staticmethod
    def command_line_parser():
        parser = super(
            StepDeg, StepDeg).command_line_parser()
        parser.add_argument(
            '--degree_scheduler_gamma',
            type=float
        )
        parser.add_argument(
            '--degree_scheduler_step_size',
            type=int
        )
        parser.add_argument(
            '--degree_scheduler_base_deg',
            type=float
        )
        return parser



class ExponentialDeg(Scheduler):
    def __init__(self,
                 degree_scheduler_base_deg: float,
                 degree_scheduler_gamma: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gamma = degree_scheduler_gamma
        self._base = degree_scheduler_base_deg


    def setting(self):
        self._last = self._base

    def _get_next(self) -> float:
        return self._last * self._gamma

    @staticmethod
    def command_line_parser():
        parser = super(
            ExponentialDeg, ExponentialDeg).command_line_parser()
        parser.add_argument(
            '--degree_scheduler_gamma',
            type=float
        )
        parser.add_argument(
            '--degree_scheduler_base_deg',
            type=float
        )
        return parser




class LinearDeg(Scheduler):
    def __init__(self,
                 degree_scheduler_base_deg: float,
                 degree_scheduler_min_deg: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last = degree_scheduler_base_deg
        self._base = degree_scheduler_base_deg
        self._min = degree_scheduler_min_deg

    def _get_next(self) -> float:
        test = self._last  - self._step_size
        return test if test >= 0 else 0


    def setting(self):
        self._step_size = (self._base - self._min) / self._T


    @staticmethod
    def command_line_parser():
        parser = super(
            LinearDeg, LinearDeg).command_line_parser()
        parser.add_argument(
            '--degree_scheduler_min_deg',
            type=float
        )
        parser.add_argument(
            '--degree_scheduler_base_deg',
            type=float
        )
        return parser




class ConstantDeg(Scheduler):
    def __init__(self,
                 degree_scheduler_base_deg: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._base = degree_scheduler_base_deg


    def setting(self):
        self._last = self._base

    @staticmethod
    def command_line_parser():
        parser = super(
            ExponentialDeg, ExponentialDeg).command_line_parser()
        parser.add_argument(
            '--degree_scheduler_base_deg',
            type=float
        )
        return parser

    def _get_next(self) -> float:
        return self._last

