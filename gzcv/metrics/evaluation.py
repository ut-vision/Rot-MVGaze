import abc

from gzcv.utils.math import angular_error


class AbstructEvaluator(metaclass=abc.ABCMeta):
    @property
    def name(self):
        if hasattr(self, "_name") and self._name is not None:
            return self._name
        else:
            return self.__class__.__name__

    @abc.abstractmethod
    def __call__(self, data):
        pass


class AngularError(AbstructEvaluator):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        """
        [Args]   : a, b: [torch.Tensor], shaped [B, 2/3]
        [Returns]: error_degree: [torch.Tensor], shaped [B, ]
        """
        a = data["pred_gaze"]
        b = data["gt_gaze"]
        error_degree = angular_error(a, b)
        return error_degree
