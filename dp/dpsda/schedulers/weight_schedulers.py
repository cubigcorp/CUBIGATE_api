from dpsda.schedulers.scheduler import Scheduler

class StepWeight(Scheduler):
    def __init__(self,
                 weight_scheduler_base_w: float,
                 weight_scheduler_gamma: float,
                 weight_scheduler_step_size: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gamma = weight_scheduler_gamma
        self._step_size = weight_scheduler_step_size
        self._base = weight_scheduler_base_w

    def _get_next(self) -> float:
        if self._step_count % self._step_size == 0:
            return self._last * self._gamma
        else:
            return self._last


    def setting(self):
        self._last = self._base


    @staticmethod
    def command_line_parser():
        parser = super(
            StepWeight, StepWeight).command_line_parser()
        parser.add_argument(
            '--weight_scheduler_gamma',
            type=float
        )
        parser.add_argument(
            '--weight_scheduler_step_size',
            type=int
        )
        parser.add_argument(
            '--weight_scheduler_base_w',
            type=float
        )
        return parser




class ExponentialWeight(Scheduler):
    def __init__(self,
                 weight_scheduler_base_w: float,
                 weight_scheduler_gamma: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gamma = weight_scheduler_gamma
        self._base = weight_scheduler_base_w


    def setting(self):
        self._last = self._base

    def _get_next(self) -> float:
        return self._last * self._gamma

    @staticmethod
    def command_line_parser():
        parser = super(
            ExponentialWeight, ExponentialWeight).command_line_parser()
        parser.add_argument(
            '--weight_scheduler_gamma',
            type=float
        )
        parser.add_argument(
            '--weight_scheduler_base_w',
            type=float
        )
        return parser




class LinearWeight(Scheduler):
    def __init__(self,
                 weight_scheduler_base_w: float,
                 weight_scheduler_min_w: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last = weight_scheduler_base_w
        self._base = weight_scheduler_base_w
        self._min = weight_scheduler_min_w

    def _get_next(self) -> float:
        test = self._last + self._step_size
        return test if test >= 0 else 0


    def setting(self):
        self._step_size = (self._base - self._min) / self._T


    @staticmethod
    def command_line_parser():
        parser = super(
            LinearWeight, LinearWeight).command_line_parser()
        parser.add_argument(
            '--weight_scheduler_min_w',
            type=float
        )
        parser.add_argument(
            '--weight_scheduler_base_w',
            type=float
        )
        return parser




class ConstantWeight(Scheduler):
    def __init__(self,
                 weight_scheduler_base_w: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._base = weight_scheduler_base_w


    def setting(self):
        self._last = self._base

    @staticmethod
    def command_line_parser():
        parser = super(
            ExponentialWeight, ExponentialWeight).command_line_parser()
        parser.add_argument(
            '--weight_scheduler_base_w',
            type=float
        )
        return parser

    def _get_next(self) -> float:
        return self._last




