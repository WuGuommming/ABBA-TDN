
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops.logger import setup_logger
from ops.lr_scheduler import get_scheduler
from ops.utils import reduce_tensor
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from tensorboardX import SummaryWriter
# from torch.utils.data import *
import torch.utils.data
import torchvision
import numpy as np
import torch.autograd
from torch.autograd import Variable
from linf_sgd import Linf_SGD
from PIL import Image
from torchvision import transforms

import abc
from abc import abstractmethod
import functools
import inspect
from numbers import Number
import itertools
import warnings
import logging
from poolnet import build_model, weights_init
import torch.nn.functional as F
import cv2 as cv

best_prec1 = 0
zeroinput = 0
oneinput = 1


def softmax(logits):
    """Transforms predictions into probability values.

    Parameters
    ----------
    logits : array_like
        The logits predicted by the model.

    Returns
    -------
    `numpy.ndarray`
        Probability values corresponding to the logits.
    """
    if logits.ndim!=1:
        i=None

    assert logits.ndim == 1

    # for numerical reasons we subtract the max logit
    # (mathematically it doesn't matter!)
    # otherwise exp(logits) might become too large or too small
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / np.sum(e)


def crossentropy(label, logits):
    """Calculates the cross-entropy.

    Parameters
    ----------
    logits : array_like
        The logits predicted by the model.
    label : int
        The label describing the target distribution.

    Returns
    -------
    float
        The cross-entropy between softmax(logits) and onehot(label).

    """

    assert logits.ndim == 1

    # for numerical reasons we subtract the max logit
    # (mathematically it doesn't matter!)
    # otherwise exp(logits) might become too large or too small

    if torch.is_tensor(logits):
        logits = np.array(logits)
    logits = logits - np.max(logits)
    e = np.exp(logits)
    s = np.sum(e)
    ce = np.log(s) - logits[label]
    return ce


class Distance(abc.ABC):
    """Base class for distances.

    This class should be subclassed when implementing
    new distances. Subclasses must implement _calculate.

    """

    def __init__(self, reference=None, other=None, bounds=None, value=None):

        if value is not None:
            # alternative constructor
            assert isinstance(value, Number)
            assert reference is None
            assert other is None
            assert bounds is None
            self.reference = None
            self.other = None
            self._bounds = None
            self._value = value
            self._gradient = None
        else:
            # standard constructor
            self.reference = reference
            self.other = other
            self._bounds = bounds
            self._value, self._gradient = self._calculate()

        assert self._value is not None

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient

    @abstractmethod
    def _calculate(self):
        """Returns distance and gradient of distance w.r.t. to self.other"""
        raise NotImplementedError

    def name(self):
        return self.__class__.__name__

    def __str__(self):
        return "{} = {:.6e}".format(self.name(), self._value)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if other.__class__ != self.__class__:
            raise TypeError(
                "Comparisons are only possible between the same distance types."
            )
        return self.value == other.value

    def __lt__(self, other):
        if other.__class__ != self.__class__:
            raise TypeError(
                "Comparisons are only possible between the same distance types."
            )
        return self.value < other.value


class MeanSquaredDistance(Distance):
    """Calculates the mean squared error between two inputs.

    """

    def _calculate(self):
        min_, max_ = self._bounds
        n = self.reference.size
        f = n * (max_ - min_) ** 2

        diff = self.other - self.reference
        value = np.vdot(diff, diff) / f

        # calculate the gradient only when needed
        self._g_diff = diff
        self._g_f = f
        gradient = None
        return value, gradient

    @property
    def gradient(self):
        if self._gradient is None:
            self._gradient = self._g_diff / (self._g_f / 2)
        return self._gradient

    def __str__(self):
        return "normalized MSE = {:.2e}".format(self._value)


MSE = MeanSquaredDistance


class Linfinity(Distance):
    """Calculates the L-infinity norm of the difference between two inputs.

    """

    def _calculate(self):
        min_, max_ = self._bounds
        diff = (self.other - self.reference) / (max_ - min_)
        value = np.max(np.abs(diff)).astype(np.float64)
        gradient = None
        return value, gradient

    @property
    def gradient(self):
        raise NotImplementedError

    def __str__(self):
        return "normalized Linf distance = {:.2e}".format(self._value)


class Criterion(abc.ABC):
    """Base class for criteria that define what is adversarial.

    The :class:`Criterion` class represents a criterion used to
    determine if predictions for an image are adversarial given
    a reference label. It should be subclassed when implementing
    new criteria. Subclasses must implement is_adversarial.

    """

    def name(self):
        """Returns a human readable name that uniquely identifies
        the criterion with its hyperparameters.

        Returns
        -------
        str
            Human readable name that uniquely identifies the criterion
            with its hyperparameters.

        Notes
        -----
        Defaults to the class name but subclasses can provide more
        descriptive names and must take hyperparameters into account.

        """
        return self.__class__.__name__

    @abstractmethod
    def is_adversarial(self, predictions, label):
        """Decides if predictions for an image are adversarial given
        a reference label.

        Parameters
        ----------
        predictions : :class:`numpy.ndarray`
            A vector with the pre-softmax predictions for some image.
        label : int
            The label of the unperturbed reference image.

        Returns
        -------
        bool
            True if an image with the given predictions is an adversarial
            example when the ground-truth class is given by label, False
            otherwise.

        """
        raise NotImplementedError

    def __and__(self, other):
        return CombinedCriteria(self, other)


class Misclassification(Criterion):
    """Defines adversarials as inputs for which the predicted class
    is not the original class.

    See Also
    --------
    :class:`TopKMisclassification`

    Notes
    -----
    Uses `numpy.argmax` to break ties.

    """

    def name(self):
        return "Top1Misclassification"

    def is_adversarial(self, predictions, label):
        if torch.is_tensor(predictions):
            predictions = np.array(predictions)
        top1 = np.argmax(predictions)
        return top1 != label


class GDOptimizer:
    """Basic gradient descent optimizer implementation that can minimize w.r.t.
    a single variable.

    Parameters
    ----------
    shape : tuple
        shape of the variable w.r.t. which the loss should be minimized

    """

    def __init__(self, learning_rate):
        self._learning_rate = learning_rate

    def __call__(self, gradient):
        """Updates internal parameters of the optimizer and returns
        the change that should be applied to the variable.

        Parameters
        ----------
        gradient : `np.ndarray`
            the gradient of the loss w.r.t. to the variable
        """

        return -self._learning_rate * gradient


class AdamOptimizer:
    """Basic Adam optimizer implementation that can minimize w.r.t.
    a single variable.

    Parameters
    ----------
    shape : tuple
        shape of the variable w.r.t. which the loss should be minimized

    """

    def __init__(
        self, shape, dtype, learning_rate, beta1=0.9, beta2=0.999, epsilon=10e-8
    ):
        """Updates internal parameters of the optimizer and returns
        the change that should be applied to the variable.

        Parameters
        ----------
        shape : tuple
            the shape of the input
        dtype : data-type
            the data-type of the input
        learning_rate: float
            the learning rate in the current iteration
        beta1: float
            decay rate for calculating the exponentially
            decaying average of past gradients
        beta2: float
            decay rate for calculating the exponentially
            decaying average of past squared gradients
        epsilon: float
            small value to avoid division by zero
        """

        self.m = np.zeros(shape, dtype=dtype)
        self.v = np.zeros(shape, dtype=dtype)
        self.t = 0

        self._beta1 = beta1
        self._beta2 = beta2
        self._learning_rate = learning_rate
        self._epsilon = epsilon

    def __call__(self, gradient):
        """Updates internal parameters of the optimizer and returns
        the change that should be applied to the variable.

        Parameters
        ----------
        gradient : `np.ndarray`
            the gradient of the loss w.r.t. to the variable
        """

        self.t += 1

        self.m = self._beta1 * self.m + (1 - self._beta1) * gradient
        self.v = self._beta2 * self.v + (1 - self._beta2) * gradient ** 2

        bias_correction_1 = 1 - self._beta1 ** self.t
        bias_correction_2 = 1 - self._beta2 ** self.t

        m_hat = self.m / bias_correction_1
        v_hat = self.v / bias_correction_2

        return -self._learning_rate * m_hat / (np.sqrt(v_hat) + self._epsilon)


class StopAttack(Exception):
    """Exception thrown to request early stopping of an attack
    if a given (optional!) threshold is reached."""

    pass


class BaseAdversarial(object):
    """Defines an adversarial that should be found and stores the result.

    The :class:`Adversarial` class represents a single adversarial example
    for a given model, criterion and reference input. It can be passed to
    an adversarial attack to find the actual adversarial perturbation.

    Parameters
    ----------
    model : a :class:`Model` instance
        The model that should be fooled by the adversarial.
    criterion : a :class:`Criterion` instance
        The criterion that determines which inputs are adversarial.
    unperturbed : a :class:`numpy.ndarray`
        The unperturbed input to which the adversarial input should be as close as possible.
    original_class : int
        The ground-truth label of the unperturbed input.
    distance : a :class:`Distance` class
        The measure used to quantify how close inputs are.
    threshold : float or :class:`Distance`
        If not None, the attack will stop as soon as the adversarial
        perturbation has a size smaller than this threshold. Can be
        an instance of the :class:`Distance` class passed to the distance
        argument, or a float assumed to have the same unit as the
        the given distance. If None, the attack will simply minimize
        the distance as good as possible. Note that the threshold only
        influences early stopping of the attack; the returned adversarial
        does not necessarily have smaller perturbation size than this
        threshold; the `reached_threshold()` method can be used to check
        if the threshold has been reached.

    """

    def __init__(
            self,
            model,
            criterion,
            unperturbed,
            original_class,
            distance=MSE,
            threshold=None,
            verbose=False,
    ):
        if inspect.isclass(criterion):
            raise ValueError("criterion should be an instance, not a class")

        self.__model = model
        self.__criterion = criterion
        self.__unperturbed = unperturbed
        self.__unperturbed_for_distance = unperturbed
        self.__original_class = original_class
        self.__distance = distance

        if threshold is not None and not isinstance(threshold, Distance):
            threshold = distance(value=threshold)
        self.__threshold = threshold

        self.verbose = verbose

        self.__best_adversarial = None
        self.__best_distance = distance(value=np.inf)
        self.__best_adversarial_output = None

        self._total_prediction_calls = 0
        self._total_gradient_calls = 0

        self._best_prediction_calls = 0
        self._best_gradient_calls = 0

        # check if the original input is already adversarial
        self._check_unperturbed()

    def _check_unperturbed(self):
        try:
            self.forward_one(self.__unperturbed)
        except StopAttack:
            # if a threshold is specified and the unperturbed input is
            # misclassified, this can already cause a StopAttack
            # exception
            assert self.distance.value == 0.0

    def _reset(self):
        self.__best_adversarial = None
        self.__best_distance = self.__distance(value=np.inf)
        self.__best_adversarial_output = None

        self._best_prediction_calls = 0
        self._best_gradient_calls = 0

        self._check_unperturbed()

    @property
    def perturbed(self):
        """The best adversarial example found so far."""
        return self.__best_adversarial

    @property
    def output(self):
        """The model predictions for the best adversarial found so far.

        None if no adversarial has been found.
        """
        return self.__best_adversarial_output

    @property
    def adversarial_class(self):
        """The argmax of the model predictions for the best adversarial found so far.

        None if no adversarial has been found.
        """
        if self.output is None:
            return None
        return np.argmax(self.output)

    @property
    def distance(self):
        """The distance of the adversarial input to the original input."""
        return self.__best_distance

    @property
    def unperturbed(self):
        """The original input."""
        return self.__unperturbed

    @property
    def original_class(self):
        """The class of the original input (ground-truth, not model prediction)."""
        return self.__original_class

    @property
    def _model(self):  # pragma: no cover
        """Should not be used."""
        return self.__model

    @property
    def _criterion(self):  # pragma: no cover
        """Should not be used."""
        return self.__criterion

    @property
    def _distance(self):  # pragma: no cover
        """Should not be used."""
        return self.__distance

    def set_distance_dtype(self, dtype):
        assert dtype >= self.__unperturbed.dtype
        self.__unperturbed_for_distance = self.__unperturbed.astype(dtype, copy=False)

    def reset_distance_dtype(self):
        self.__unperturbed_for_distance = self.__unperturbed

    def normalized_distance(self, x):
        """Calculates the distance of a given input x to the original input.

        Parameters
        ----------
        x : `numpy.ndarray`
            The input x that should be compared to the original input.

        Returns
        -------
        :class:`Distance`
            The distance between the given input and the original input.

        """
        return self.__distance(self.__unperturbed_for_distance, x, bounds=self.bounds())

    def reached_threshold(self):
        """Returns True if a threshold is given and the currently
        best adversarial distance is smaller than the threshold."""
        return self.__threshold is not None and self.__best_distance <= self.__threshold

    def __new_adversarial(self, x, predictions, in_bounds):
        x = x.copy()  # to prevent accidental inplace changes
        distance = self.normalized_distance(x)
        if in_bounds and self.__best_distance > distance:
            # new best adversarial
            if self.verbose:
                print("new best adversarial: {}".format(distance))

            self.__best_adversarial = x
            self.__best_distance = distance
            self.__best_adversarial_output = predictions

            self._best_prediction_calls = self._total_prediction_calls
            self._best_gradient_calls = self._total_gradient_calls

            if self.reached_threshold():
                raise StopAttack

            return True, distance
        return False, distance

    def __is_adversarial(self, x, predictions, in_bounds):
        """Interface to criterion.is_adverarial that calls
        __new_adversarial if necessary.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            The input that should be checked.
        predictions : :class:`numpy.ndarray`
            A vector with the pre-softmax predictions for some input x.
        label : int
            The label of the unperturbed reference input.

        """
        is_adversarial = self.__criterion.is_adversarial(
            predictions, self.__original_class
        )
        assert isinstance(is_adversarial, bool) or isinstance(is_adversarial, np.bool_)
        if is_adversarial:
            is_best, distance = self.__new_adversarial(x, predictions, in_bounds)
        else:
            is_best = False
            distance = None
        return is_adversarial, is_best, distance

    @property
    def target_class(self):
        """Interface to criterion.target_class for attacks.

        """
        try:
            target_class = self.__criterion.target_class()
        except AttributeError:
            target_class = None
        return target_class

    def num_classes(self):
        n = self.__model.num_classes()
        assert isinstance(n, numbers.Number)
        return n

    def bounds(self):
        min_, max_ = self.__model.bounds()
        assert isinstance(min_, numbers.Number)
        assert isinstance(max_, numbers.Number)
        assert min_ < max_
        return min_, max_

    def in_bounds(self, input_):
        min_, max_ = self.bounds()
        return min_ <= input_.min() and input_.max() <= max_

    def channel_axis(self, batch):
        """Interface to model.channel_axis for attacks.

        Parameters
        ----------
        batch : bool
            Controls whether the index of the axis for a batch of inputs
            (4 dimensions) or a single input (3 dimensions) should be returned.

        """
        axis = self.__model.channel_axis()
        if not batch:
            axis = axis - 1
        return axis

    def has_gradient(self):
        """Returns true if _backward and _forward_backward can be called
        by an attack, False otherwise.

        """
        try:
            self.__model.gradient
            self.__model.gradient_one
            self.__model.forward_and_gradient_one
            self.__model.backward
            self.__model.backward_one
        except AttributeError:
            return False
        else:
            return True

    def forward_one(self, x, strict=True, return_details=False):
        """Interface to model.forward_one for attacks.

        Parameters
        ----------
        x : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """

        in_bounds = self.in_bounds(x)
        assert not strict or in_bounds

        self._total_prediction_calls += 1
        predictions = self.__model.forward_one(x)
        is_adversarial, is_best, distance = self.__is_adversarial(
            x, predictions, in_bounds
        )

        assert predictions.ndim == 1
        if return_details:
            return predictions, is_adversarial, is_best, distance
        else:
            return predictions, is_adversarial

    def forward(self, inputs, greedy=False, strict=True, return_details=False):
        """Interface to model.forward for attacks.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the model.
        greedy : bool
            Whether the first adversarial should be returned.
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        if strict:
            in_bounds = self.in_bounds(inputs)
            assert in_bounds

        self._total_prediction_calls += len(inputs)
        predictions = self.__model.forward(inputs)

        assert predictions.ndim == 2
        assert predictions.shape[0] == inputs.shape[0]

        if return_details:
            assert greedy

        adversarials = []
        for i in range(len(predictions)):
            if strict:
                in_bounds_i = True
            else:
                in_bounds_i = self.in_bounds(inputs[i])
            is_adversarial, is_best, distance = self.__is_adversarial(
                inputs[i], predictions[i], in_bounds_i
            )
            if is_adversarial and greedy:
                if return_details:
                    return predictions, is_adversarial, i, is_best, distance
                else:
                    return predictions, is_adversarial, i
            adversarials.append(is_adversarial)

        if greedy:  # pragma: no cover
            # no adversarial found
            if return_details:
                return predictions, False, None, False, None
            else:
                return predictions, False, None

        is_adversarial = np.array(adversarials)
        assert is_adversarial.ndim == 1
        assert is_adversarial.shape[0] == inputs.shape[0]
        return predictions, is_adversarial

    def gradient_one(self, x=None, label=None, strict=True):
        """Interface to model.gradient_one for attacks.

        Parameters
        ----------
        x : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
            Defaults to the original input.
        label : int
            Label used to calculate the loss that is differentiated.
            Defaults to the original label.
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        assert self.has_gradient()

        if x is None:
            x = self.__unperturbed
        if label is None:
            label = self.__original_class

        assert not strict or self.in_bounds(x)

        self._total_gradient_calls += 1
        gradient = self.__model.gradient_one(x, label)

        assert gradient.shape == x.shape
        return gradient

    def forward_and_gradient_one(
            self, x=None, label=None, strict=True, return_details=False
    ):
        """Interface to model.forward_and_gradient_one for attacks.

        Parameters
        ----------
        x : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
            Defaults to the original input.
        label : int
            Label used to calculate the loss that is differentiated.
            Defaults to the original label.
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        assert self.has_gradient()

        if x is None:
            x = self.__unperturbed
        if label is None:
            label = self.__original_class

        in_bounds = self.in_bounds(x)
        assert not strict or in_bounds

        self._total_prediction_calls += 1
        self._total_gradient_calls += 1
        predictions, gradient = self.__model.forward_and_gradient_one(x, label)
        is_adversarial, is_best, distance = self.__is_adversarial(
            x, predictions, in_bounds
        )

        assert predictions.ndim == 1
        assert gradient.shape == x.shape
        if return_details:
            return predictions, gradient, is_adversarial, is_best, distance
        else:
            return predictions, gradient, is_adversarial

    def forward_and_gradient(self, x, label=None, strict=True, return_details=False):
        """Interface to model.forward_and_gradient_one for attacks.

        Parameters
        ----------
        x : `numpy.ndarray`
            Multiple input with shape as expected by the model
            (with the batch dimension).
        label : `numpy.ndarray`
            Labels used to calculate the loss that is differentiated.
            Defaults to the original label.
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        assert self.has_gradient()

        if label is None:
            label = np.ones(len(x), dtype=np.int) * self.__original_class

        in_bounds = self.in_bounds(x)
        assert not strict or in_bounds

        self._total_prediction_calls += len(x)
        self._total_gradient_calls += len(x)
        predictions, gradients = self.__model.forward_and_gradient(x, label)

        assert predictions.ndim == 2
        assert gradients.shape == x.shape

        is_adversarials, is_bests, distances = [], [], []
        for single_x, prediction in zip(x, predictions):
            is_adversarial, is_best, distance = self.__is_adversarial(
                single_x, prediction, in_bounds
            )
            is_adversarials.append(is_adversarial)
            is_bests.append(is_best)
            distances.append(distance)

        is_adversarials = np.array(is_adversarials)
        is_bests = np.array(is_bests)
        distances = np.array(distances)

        if return_details:
            return predictions, gradients, is_adversarials, is_bests, distances
        else:
            return predictions, gradients, is_adversarials

    def backward_one(self, gradient, x=None, strict=True):
        """Interface to model.backward_one for attacks.

        Parameters
        ----------
        gradient : `numpy.ndarray`
            Gradient of some loss w.r.t. the logits.
        x : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).

        Returns
        -------
        gradient : `numpy.ndarray`
            The gradient w.r.t the input.

        See Also
        --------
        :meth:`gradient`

        """
        assert self.has_gradient()
        assert gradient.ndim == 1

        if x is None:
            x = self.__unperturbed

        assert not strict or self.in_bounds(x)

        self._total_gradient_calls += 1
        gradient = self.__model.backward_one(gradient, x)

        assert gradient.shape == x.shape
        return gradient


class Adversarial(BaseAdversarial):
    def _check_unperturbed(self):
        try:
            # for now, we use the non-yielding implementation in the super-class
            # TODO: add support for batching this first call as well
            super(Adversarial, self).forward_one(self._BaseAdversarial__unperturbed)
        except StopAttack:
            # if a threshold is specified and the unperturbed input is
            # misclassified, this can already cause a StopAttack
            # exception
            assert self.distance.value == 0.0

    def forward_one(self, x, strict=True, return_details=False):
        """Interface to model.forward_one for attacks.

        Parameters
        ----------
        x : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        in_bounds = self.in_bounds(x)
        assert not strict or in_bounds

        self._total_prediction_calls += 1
        predictions = yield ("forward_one", x)
        predictions = predictions.detach().cpu()

        assert predictions is not None, (
            "Predictions is None; this happens if"
            " you forget the `yield from` "
            "preceding the forward() call."
        )

        is_adversarial, is_best, distance = self._BaseAdversarial__is_adversarial(
            x, predictions, in_bounds
        )

        assert predictions.ndim == 1
        if return_details:
            return predictions, is_adversarial, is_best, distance
        else:
            return predictions, is_adversarial

    def forward(self, inputs, greedy=False, strict=True, return_details=False):
        """Interface to model.forward for attacks.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the model.
        greedy : bool
            Whether the first adversarial should be returned.
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        if strict:
            in_bounds = self.in_bounds(inputs)
            assert in_bounds

        self._total_prediction_calls += len(inputs)
        predictions = yield ("forward", inputs)

        assert predictions is not None, (
            "Prediction is None; this happens if "
            "you forget the `yield from` "
            "preceding the forward() call."
        )

        assert predictions.ndim == 2
        assert predictions.shape[0] == inputs.shape[0]

        if return_details:
            assert greedy

        adversarials = []
        for i in range(len(predictions)):
            if strict:
                in_bounds_i = True
            else:
                in_bounds_i = self.in_bounds(inputs[i])
            is_adversarial, is_best, distance = self._Adversarial__is_adversarial(
                inputs[i], predictions[i], in_bounds_i
            )
            if is_adversarial and greedy:
                if return_details:
                    return predictions, is_adversarial, i, is_best, distance
                else:
                    return predictions, is_adversarial, i
            adversarials.append(is_adversarial)

        if greedy:  # pragma: no cover
            # no adversarial found
            if return_details:
                return predictions, False, None, False, None
            else:
                return predictions, False, None

        is_adversarial = np.array(adversarials)
        assert is_adversarial.ndim == 1
        assert is_adversarial.shape[0] == inputs.shape[0]
        return predictions, is_adversarial

    def gradient_one(self, x=None, label=None, strict=True):
        """Interface to model.gradient_one for attacks.

        Parameters
        ----------
        x : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
            Defaults to the original input.
        label : int
            Label used to calculate the loss that is differentiated.
            Defaults to the original label.
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        assert self.has_gradient()

        if x is None:
            x = self._Adversarial__unperturbed
        if label is None:
            label = self._Adversarial__original_class

        assert not strict or self.in_bounds(x)

        self._total_gradient_calls += 1
        gradient = yield ("gradient_one", x, label)

        assert gradient is not None, (
            "gradient is None; this happens if "
            "you forget the `yield from` "
            "preceding the forward() call."
        )

        assert gradient.shape == x.shape
        return gradient

    def forward_and_gradient_one(
            self, x=None, label=None, strict=True, return_details=False
    ):
        """Interface to model.forward_and_gradient_one for attacks.

        Parameters
        ----------
        x : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
            Defaults to the original input.
        label : int
            Label used to calculate the loss that is differentiated.
            Defaults to the original label.
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        assert self.has_gradient()

        if x is None:
            x = self._Adversarial__unperturbed
        if label is None:
            label = self._Adversarial__original_class

        in_bounds = self.in_bounds(x)
        assert not strict or in_bounds

        self._total_prediction_calls += 1
        self._total_gradient_calls += 1
        output = yield ("forward_and_gradient_one", x, label)

        assert output is not None, (
            "Prediction is None; this happens if "
            "you forget the `yield from` "
            "preceding the forward() call."
        )
        predictions, gradient = output

        is_adversarial, is_best, distance = self._Adversarial__is_adversarial(
            x, predictions, in_bounds
        )
        assert predictions.ndim == 1
        assert gradient.shape == x.shape
        if return_details:
            return predictions, gradient, is_adversarial, is_best, distance
        else:
            return predictions, gradient, is_adversarial

    def backward_one(self, gradient, x=None, strict=True):
        """Interface to model.backward_one for attacks.

        Parameters
        ----------
        gradient : `numpy.ndarray`
            Gradient of some loss w.r.t. the logits.
        x : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).

        Returns
        -------
        gradient : `numpy.ndarray`
            The gradient w.r.t the input.

        See Also
        --------
        :meth:`gradient`

        """
        assert self.has_gradient()
        assert gradient.ndim == 1

        if x is None:
            x = self._Adversarial__unperturbed

        assert not strict or self.in_bounds(x)

        self._total_gradient_calls += 1
        gradient = yield ("backward_one", gradient, x)

        assert gradient is not None, (
            "gradient is None; this happens if "
            "you forget the `yield from` "
            "preceding the forward() call."
        )

        assert gradient.shape == x.shape
        return gradient


def run_parallel(  # noqa: C901
        batch,
        create_attack_fn,
        model,
        criterion,
        inputs,
        labels,
        distance,
        threshold=None,
        verbose=False,
        individual_kwargs=None,
        **kwargs
):
    """
    Runs the same type of attack vor multiple inputs in parallel by
    batching them.

    Parameters
    ----------
    create_attack_fn : a function returning an :class:`Attack` instance
        The attack to use.
    model : a :class:`Model` instance
        The model that should be fooled by the adversarial.
    criterion : a :class:`Criterion` class or list of :class:`Criterion` classes
        The criterion/criteria that determine(s) which inputs are adversarial.
    inputs :  a :class:`numpy.ndarray`
        The unperturbed inputs to which the adversarial input should be as close
        as possible.
    labels :  a :class:`numpy.ndarray`
        The ground-truth labels of the unperturbed inputs.
    distance : a :class:`Distance` class or list of :class:`Distance` classes
        The measure(s) used to quantify how close inputs are.
    threshold : float or :class:`Distance`
        If not None, the attack will stop as soon as the adversarial
        perturbation has a size smaller than this threshold. Can be
        an instance of the :class:`Distance` class passed to the distance
        argument, or a float assumed to have the same unit as the
        the given distance. If None, the attack will simply minimize
        the distance as good as possible. Note that the threshold only
        influences early stopping of the attack; the returned adversarial
        does not necessarily have smaller perturbation size than this
        threshold; the :class:`Adversarial`.`reached_threshold()` method can
         be used to check
        if the threshold has been reached.
    verbose : bool
        Whether the adversarial examples should be created in verbose mode.
    individual_kwargs : list of dict
         The optional keywords passed to create_attack_fn that should be
         different for each of the input samples. For each input a different
         set of arguments will be used.
    kwargs : dict
        The optional keywords passed to create_attack_fn that are common for
        every element in the batch.

    Returns
    -------
    The list of generated adversarial examples.
    """

    assert len(inputs) == len(
        labels
    ), "The number of inputs must match the number of labels."  # noqa: E501

    # if only one criterion has been passed use the same one for all inputs
    if not isinstance(criterion, (list, tuple)):
        criterion = [criterion] * len(inputs)
    else:
        assert len(criterion) == len(
            inputs
        ), "The number of criteria must match the number of inputs."  # noqa: E501

    # if only one distance has been passed use the same one for all inputs
    if not isinstance(distance, (list, tuple)):
        distance = [distance] * len(inputs)
    else:
        assert len(distance) == len(
            inputs
        ), "The number of distances must match the number of inputs."  # noqa: E501

    if individual_kwargs is None:
        individual_kwargs = [kwargs] * len(inputs)
    else:
        assert isinstance(
            individual_kwargs, (list, tuple)
        ), "Individual_kwargs must be a list or None."  # noqa: E501
        assert len(individual_kwargs) == len(
            inputs
        ), "The number of individual_kwargs must match the number of inputs."  # noqa: E501

        for i in range(len(individual_kwargs)):
            assert isinstance(individual_kwargs[i], dict)
            individual_kwargs[i] = {**kwargs, **individual_kwargs[i]}
    
    advs = [
        Adversarial(
            model,
            _criterion,
            x,
            label,
            distance=_distance,
            threshold=threshold,
            verbose=verbose,
        )
        for _criterion, _distance, x, label in zip(criterion, distance, inputs, labels)
    ]

    attacks = [
        create_attack_fn(model, distance).as_generator(batch, adv, **kwargs)
        for adv, kwargs in zip(advs, individual_kwargs)
    ]

    predictions = [None for _ in attacks]
    gradients = []
    backwards = []
    prediction_gradients = []

    batched_predictions = []
    results = itertools.chain(
        predictions, gradients, backwards, prediction_gradients, batched_predictions
    )

    while True:
        attacks_requesting_predictions = []
        predictions_args = []
        attacks_requesting_gradients = []
        gradients_args = []
        attacks_requesting_backwards = []
        backwards_args = []
        attacks_requesting_prediction_gradients = []
        predictions_gradients_args = []
        attacks_requesting_batched_predictions = []
        batched_predictions_args = []
        for attack, result in zip(attacks, results):
            try:
                x = attack.send(result)
            except StopIteration:
                # print("StopIteration")
                continue
            method, args = x[0], x[1:]

            if method == "forward_one":
                attacks_requesting_predictions.append(attack)
                predictions_args.append(args)
            elif method == "gradient_one":
                attacks_requesting_gradients.append(attack)
                gradients_args.append(args)
            elif method == "backward_one":
                attacks_requesting_backwards.append(attack)
                backwards_args.append(args)
            elif method == "forward_and_gradient_one":
                attacks_requesting_prediction_gradients.append(attack)
                predictions_gradients_args.append(args)
            elif method == "forward":
                attacks_requesting_batched_predictions.append(attack)
                batched_predictions_args.append(args)
            else:
                assert False

        n_active_attacks = (
                len(attacks_requesting_predictions)
                + len(attacks_requesting_gradients)
                + len(attacks_requesting_backwards)
                + len(attacks_requesting_prediction_gradients)
                + len(attacks_requesting_batched_predictions)
        )
        if n_active_attacks < len(predictions) + len(gradients) + len(backwards) + len(
                prediction_gradients
        ) + len(
            batched_predictions
        ):  # noqa: E501
            # an attack completed in this iteration
            logging.info(
                "{} of {} attacks completed".format(
                    len(advs) - n_active_attacks, len(advs)
                )
            )  # noqa: E501
        if n_active_attacks == 0:
            break

        if len(attacks_requesting_predictions) > 0:
            logging.debug(
                "calling forward with {}".format(len(attacks_requesting_predictions))
            )  # noqa: E501
            predictions_args = map(np.stack, zip(*predictions_args))
            predictions = model.forward(*predictions_args)
        else:
            predictions = []

        if len(attacks_requesting_batched_predictions) > 0:
            logging.debug(
                "calling native forward with {}".format(
                    len(attacks_requesting_batched_predictions)
                )
            )  # noqa: E501

            # we are only interested in the first argument
            inputs = [x[0] for x in batched_predictions_args]

            # merge individual batches into one larger super-batch
            batch_lengths = [len(x) for x in inputs]
            batch_splits = np.cumsum(batch_lengths)
            inputs = np.concatenate([x for x in inputs])

            # split super-batch back into individual batches
            batched_predictions = model.forward(inputs)
            batched_predictions = np.split(batched_predictions, batch_splits, axis=0)

        else:
            batched_predictions = []

        if len(attacks_requesting_gradients) > 0:
            logging.debug(
                "calling gradient with {}".format(len(attacks_requesting_gradients))
            )  # noqa: E501
            gradients_args = map(np.stack, zip(*gradients_args))
            gradients = model.gradient(*gradients_args)
        else:
            gradients = []

        if len(attacks_requesting_backwards) > 0:
            logging.debug(
                "calling backward with {}".format(len(attacks_requesting_backwards))
            )  # noqa: E501
            backwards_args = map(np.stack, zip(*backwards_args))
            backwards = model.backward(*backwards_args)
        else:
            backwards = []
        if len(attacks_requesting_prediction_gradients) > 0:
            logging.debug(
                "calling forward_and_gradient_one with {}".format(
                    len(attacks_requesting_prediction_gradients)
                )
            )  # noqa: E501

            predictions_gradients_args = map(np.stack, zip(*predictions_gradients_args))

            prediction_gradients = model.forward_and_gradient(
                *predictions_gradients_args
            )

            prediction_gradients = list(zip(*prediction_gradients))
        else:
            prediction_gradients = []

        attacks = itertools.chain(
            attacks_requesting_predictions,
            attacks_requesting_gradients,
            attacks_requesting_backwards,
            attacks_requesting_prediction_gradients,
            attacks_requesting_batched_predictions,
        )
        results = itertools.chain(
            predictions, gradients, backwards, prediction_gradients, batched_predictions
        )
    return advs


class BaseAttack(abc.ABC):
    """Abstract base class for adversarial attacks.

    The :class:`Attack` class represents an adversarial attack that searches
    for adversarial examples. It should be subclassed when implementing new
    attacks.

    Parameters
    ----------
    model : a :class:`Model` instance
        The model that should be fooled by the adversarial.
        Ignored if the attack is called with an :class:`Adversarial` instance.
    criterion : a :class:`Criterion` instance
        The criterion that determines which inputs are adversarial.
        Ignored if the attack is called with an :class:`Adversarial` instance.
    distance : a :class:`Distance` class
        The measure used to quantify similarity between inputs.
        Ignored if the attack is called with an :class:`Adversarial` instance.
    threshold : float or :class:`Distance`
        If not None, the attack will stop as soon as the adversarial
        perturbation has a size smaller than this threshold. Can be
        an instance of the :class:`Distance` class passed to the distance
        argument, or a float assumed to have the same unit as the
        the given distance. If None, the attack will simply minimize
        the distance as good as possible. Note that the threshold only
        influences early stopping of the attack; the returned adversarial
        does not necessarily have smaller perturbation size than this
        threshold; the `reached_threshold()` method can be used to check
        if the threshold has been reached.
        Ignored if the attack is called with an :class:`Adversarial` instance.

    Notes
    -----
    If a subclass overwrites the constructor, it should call the super
    constructor with *args and **kwargs.

    """

    def __init__(
            self, model, distance, criterion=Misclassification(), threshold=None
    ):
        self._default_model = model
        self._default_criterion = criterion
        self._default_distance = distance
        self._default_threshold = threshold

        # to customize the initialization in subclasses, please
        # try to overwrite _initialize instead of __init__ if
        # possible
        self._initialize()

    def _initialize(self):
        """Additional initializer that can be overwritten by
        subclasses without redefining the full __init__ method
        including all arguments and documentation."""
        pass

    @abstractmethod
    def __call__(self, inputs, labels, unpack=True, **kwargs):
        raise NotImplementedError

    def name(self):
        """Returns a human readable name that uniquely identifies
        the attack with its hyperparameters.

        Returns
        -------
        str
            Human readable name that uniquely identifies the attack
            with its hyperparameters.

        Notes
        -----
        Defaults to the class name but subclasses can provide more
        descriptive names and must take hyperparameters into account.

        """
        return self.__class__.__name__


class Attack(BaseAttack):
    def __init__(self, model, distance, criterion=Misclassification(), threshold=None):
        super(Attack, self).__init__(
            model=model, distance=distance, criterion=criterion, threshold=threshold
        )


    def __call__(self, batch, inputs, labels, unpack=True, individual_kwargs=None, **kwargs):
        assert isinstance(inputs, np.ndarray)
        assert isinstance(labels, np.ndarray)

        if len(inputs) != len(labels):
            raise ValueError("The number of inputs and labels needs to be equal")

        model = self._default_model
        criterion = self._default_criterion
        distance = self._default_distance
        threshold = self._default_threshold

        if model is None:
            raise ValueError("The attack needs to be initialized with a model")
        if criterion is None:
            raise ValueError("The attack needs to be initialized with a criterion")
        if distance is None:
            raise ValueError("The attack needs to be initialized with a distance")

        create_attack_fn = self.__class__
        advs = run_parallel(batch, 
            create_attack_fn,
            model,
            criterion,
            inputs,
            labels,
            distance=distance,
            threshold=threshold,
            individual_kwargs=individual_kwargs,
            **kwargs,
        )

        if unpack:
            advs = [a.perturbed for a in advs]
            advs = [
                p if p is not None else np.full_like(u, np.nan)
                for p, u in zip(advs, inputs)
            ]
            advs = np.stack(advs)
        return advs


def generator_decorator(generator):
    @functools.wraps(generator)
    def wrapper(self, batch, a, **kwargs):

        assert isinstance(a, Adversarial)

        if a.distance.value == 0.0:
            logging.info(
                "Not running the attack because the original input"
                " is already misclassified and the adversarial thus"
                " has a distance of 0."
            )
        elif a.reached_threshold():
            logging.info(
                "Not running the attack because the given treshold"
                " is already reached"
            )
        else:
            try:
                _ = yield from generator(self, batch, a, **kwargs)
                assert _ is None, "decorated __call__ method must return None"
            except StopAttack:
                # if a threshold is specified, StopAttack will be thrown
                # when the threshold is reached; thus we can do early
                # stopping of the attack
                logging.info("threshold reached, stopping attack")

        if a.perturbed is None:
            warnings.warn(
                "{} did not find an adversarial, maybe the model"
                " or the criterion is not supported by this"
                " attack.".format(self.name())
            )
        return a

    return wrapper


class IterativeProjectedGradientBaseAttack(Attack):
    """Base class for iterative (projected) gradient attacks.

    Concrete subclasbses should implement as_generator, _gradient
    and _clip_perturbation.

    TODO: add support for other loss-functions, e.g. the CW loss function,
    see https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
    """

    @abstractmethod
    def _gradient(self, a, x, class_, strict=True, gradient_args={}):
        raise NotImplementedError

    @abstractmethod
    def _clip_perturbation(self, a, noise, epsilon):
        raise NotImplementedError

    @abstractmethod
    def _create_optimizer(self, a, stepsize):
        raise NotImplementedError

    @abstractmethod
    def _check_distance(self, a):
        raise NotImplementedError

    def _get_mode_and_class(self, a):
        # determine if the attack is targeted or not
        target_class = a.target_class
        targeted = target_class is not None

        if targeted:
            class_ = target_class
        else:
            class_ = a.original_class
        return targeted, class_

    def _run(
            self,
            batch,
            a,
            epsilon,
            stepsize,
            iterations,
            random_start,
            return_early,
            gradient_args={},
            pert_type="Add",
            blur_model="joint",
            numSP=-1,
            mask_att_l1=2.0,
            direction=None,
            imgname=None
    ):

        self.imgname = imgname
        self.blur_model = blur_model
        self.kernel_size = 17
        self.numSP = numSP
        self.mask_att_l1 = mask_att_l1
        self.direction = direction
        if not a.has_gradient():
            warnings.warn(
                "applied gradient-based attack to model that"
                " does not provide gradients"
            )
            return

        self._check_distance(a)

        targeted, class_ = self._get_mode_and_class(a)

        optimizer = self._create_optimizer(a, stepsize)

        success = yield from self._run_one(
            batch,
            a,
            epsilon,
            optimizer,
            iterations,
            random_start,
            targeted,
            class_,
            return_early,
            gradient_args,
            pert_type
        )
        return success


    def _run_one(
            self,
            batch,
            a,
            epsilon,
            optimizer,
            iterations,
            random_start,
            targeted,
            class_,
            return_early,
            gradient_args,
            pert_type="Add"
    ):
        """ Modified the _run_one() to add the motion-blur-aware attack.
            pert_type = "Add" means to use the trainditional additional noise
                      = "Blur" means to use the novel motion-blur-aware attack
            kernel_size : define the size of linear kernel for motion blur
            blur_model: we define four attacking models:
                        image-level motion blur: whole
                        object-aware motion blur: obj
                        background-aware motion blur: backg
                        joint-object-background motion blur: joint
            """

        self._momentum_history = 0
        min_, max_ = a.bounds()
        s = max_ - min_

        original = a._BaseAdversarial__unperturbed.copy()
        original = original.reshape(40, 3, 224, 224)

        self.batch = batch
        self.disp = False

        if pert_type == "Blur":

            kernel_sz = self.kernel_size

            self.alpha_len = 1

            theta_f, theta_b, alpha, mask = self.init_flow_alpha(original, kernel_sz)
            x = self.add_flow_alpha_pert(original, kernel_sz, theta_f, theta_b, alpha, mask)
            x = np.clip(x, min_, max_)
            strict = True

            success = False

            # calulate the mask directly
            if self.numSP == -3:
                self.blur_model == "bg_obj_att"
                mask_att = self.adapt_mask_gaussblur(original, mask, class_, a._model)
                # make sure the 1 number of mask_att is smaller than that of mask
                # 40, 224, 224, 3
                mask_diff = mask - mask_att
                mask_att[mask_diff < 0] = 0
                # np.save(self.imgname + "_att{}.npy".format(self.mask_att_l1), mask_att.detach().cpu().numpy())
                print("batch " + str(self.batch) + ": mask_att. calculated!")
            
            # import imageio
            # ori = torch.tensor(original).view(40, 3, 224, 224).cpu().permute(0, 2, 3, 1).numpy() * 255.
            # os.makedirs("./ABBAimg/batch_" + str(self.batch) + "/", exist_ok=True)
            # for idx in range(40):
            #     imageio.imwrite("./ABBAimg/batch_" + str(self.batch) + "/ori_" + "_" + str(idx)+".jpg", ori[idx].astype(np.uint8))  

            for i in range(iterations):
                x = for_normalize(x)
                gradient = yield from self._gradient(
                    a, x, class_, strict=strict, gradient_args=gradient_args
                )

                theta_f = theta_f.detach()
                theta_b = theta_b.detach()
                alpha = alpha.detach()
                mask = mask.detach()

                grad_theta_f, grad_theta_b, grad_alpha, grad_mask = self.grad_flow_alpha_pert(original, kernel_sz,
                                                                                              theta_f,
                                                                                              theta_b, alpha, mask,
                                                                                              gradient)

                # non-strict only for the first call and
                # only if random_start is True
                strict = True

                with torch.no_grad():
                    grad_theta_f = -grad_theta_f
                    grad_theta_b = -grad_theta_b
                    grad_alpha = -grad_alpha

                    # updating theta_f and theta_b
                    theta_f[:, :, 2] = theta_f[:, :, 2] + 1e-2 * optimizer(grad_theta_f[:, :, 2])
                    theta_b[:, :, 2] = theta_b[:, :, 2] + 1e-2 * optimizer(grad_theta_b[:, :, 2])

                    # updating alpha:
                    pred = mask.permute(0, 3, 1, 2)
                    pred_att = mask_att.permute(0, 3, 1, 2)
                    grad_alpha_f_att = (pred_att) * grad_alpha
                    grad_alpha_f = (pred - pred_att) * grad_alpha
                    grad_alpha_b = (1 - pred) * grad_alpha

                    alpha_f_att = pred_att * alpha
                    alpha_f = (pred - pred_att) * alpha
                    alpha_b = (1 - pred) * alpha

                    alpha_f = alpha_f + optimizer(grad_alpha_f.view(kernel_sz, 40, 1, 1, -1).mean(dim=4).unsqueeze(-1))
                    alpha_f_att = alpha_f_att + optimizer(grad_alpha_f_att)
                    alpha_b = alpha_b + optimizer(grad_alpha_b.view(kernel_sz, 40, 1, 1, -1).mean(dim=4).unsqueeze(-1))
                    alpha = alpha_f + alpha_b + alpha_f_att

                    '''
                    if self.numSP != -2 and not targeted:
                        # to constraint the 0-norm of alpha
                        # alpha[torch.topk(alpha, int(kernel_sz-epsilon[1]), dim=0, largest=False, sorted=False, out=None)[1]] = 0
                        min_alpha = alpha.min(dim=0).values.unsqueeze(0).repeat(kernel_sz, 1, 1, 1)
                        max_alpha = alpha.max(dim=0).values.unsqueeze(0).repeat(kernel_sz, 1, 1, 1)
                        alpha = (alpha - min_alpha) / (max_alpha - min_alpha + 1e-25)
                        alpha[int(epsilon[1]):, :, :, :] = 0
                        # to enforce the largest result at the first channel
                        alpha[0, :, :, :] = alpha.max(dim=0).values
                        alpha = alpha / (1e-25 + alpha.sum(dim=0))
    
                    '''

                    # to constraint the shift values epsilon[0] range from -1 to 1
                    theta_f[:, :, 2] = torch.clamp(theta_f[:, :, 2], -epsilon[0], epsilon[0])
                    theta_b[:, :, 2] = torch.clamp(theta_b[:, :, 2], -epsilon[0], epsilon[0])
                    
                    x = self.add_flow_alpha_pert(original, kernel_sz, theta_f, theta_b, alpha, mask)
                    x = np.clip(x, min_, max_)

                    x = for_normalize(x)
                    logits, is_adversarial = yield from a.forward_one(x)
                    x = back_normalize(x)

                if self.batch == 494 or self.batch == 941:
                    print("saving adv image .........................................................")
                    import imageio
                    adv = x.reshape(40, 3, 224, 224).transpose(0, 2, 3, 1) * 255.
                    ori = original.reshape(40, 3, 224, 224).transpose(0, 2, 3, 1) * 255.
                    os.makedirs("./ABBAimg/batch_" + str(self.batch) + "/", exist_ok=True)
                    for idx in range(40):
                        imageio.imwrite("./ABBAimg/batch_" + str(self.batch) + "/" + str(i) + "_" + str(idx) + "_adv.jpg", adv[idx].astype(np.uint8))
                        imageio.imwrite("./ABBAimg/batch_" + str(self.batch) + "/" + str(i) + "_" + str(idx) + "_diff.jpg", abs(adv[idx] - ori[idx]).astype(np.uint8) * 10)

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                if targeted:
                    ce = crossentropy(a.original_class, logits)
                    logging.debug(
                        "crossentropy to {} is {}".format(a.original_class, ce)
                    )
                ce = crossentropy(class_, logits)
                logging.debug("crossentropy to {} is {}".format(class_, ce))

            st_iter = time.time()
            if is_adversarial:
                print("Adv theta params: theta_f[20]: {} thetha_b[20]: {}!".format(theta_f[20], theta_b[20]))

                if pert_type == "Blur":

                    max_iterations = 500
                    model = a._model
                    category = a.original_class
                    tv_beta = 3
                    learning_rate = 0.05
                    l1_coeff = 0.02  # 0.05
                    tv_coeff = 0.05
                    mask = torch.zeros([40, 1, 28, 28]).cuda()
                    mask.requires_grad_()
                    optimizer = torch.optim.Adam([mask], lr=learning_rate)
                    blurred = x

                    def tv_norm(input, tv_beta):
                        img = input
                        row_grad = torch.mean(torch.abs((img[:, :, :-1, :] - img[:, :, 1:, :])).pow(tv_beta))
                        col_grad = torch.mean(torch.abs((img[:, :, :, :-1] - img[:, :, :, 1:])).pow(tv_beta))
                        return row_grad + col_grad

                    for _ in range(max_iterations):
                        mask.requires_grad_()
                        optimizer.zero_grad()
                        model.zero_grad()

                        mask_up = F.upsample(mask, (224, 224), mode='bilinear')
                        mask_up = mask_up.repeat(1, 3, 1, 1)

                        if not torch.is_tensor(original):
                            original = torch.from_numpy(original).cuda()
                        if not torch.is_tensor(blurred):
                            blurred = torch.from_numpy(blurred).cuda()

                        perturbated_input = original.mul(1 - mask_up) + blurred.mul(mask_up)
                        perturbated_input.data.clamp_(0, 1)

                        perturbated_input = for_normalize(perturbated_input.clone(), model.input_mean, model.input_std)
                        outputs = torch.nn.Softmax()(model(perturbated_input))

                        loss = l1_coeff * torch.mean(torch.abs(mask_up)) + \
                               tv_coeff * tv_norm(mask_up, tv_beta) + outputs[0, category]

                        loss.backward()
                        optimizer.step()

                        # Optional: clamping seems to give better results
                        mask.data.clamp_(0, 1)
                        with torch.no_grad():
                            if _ % 100 == 0:
                                print("iter: " + str(_) + ", loss: ", loss.data)

                    if is_adversarial:
                        if return_early:
                            return True
                        else:
                            success = True
                else:

                    if return_early:
                        return True
                    else:
                        success = True
            print("iter time: ", time.time() - st_iter)
        return success

    def add_kernel(self, original, kernel, mask):

        original = torch.from_numpy(original).unsqueeze(0).unsqueeze(0).cuda()

        kernel_ = kernel.mul(mask.repeat(1, kernel.size()[1], 1, 1, 1)) + self.kernel_gt.mul(
            1 - mask.repeat(1, kernel.size()[1], 1, 1, 1))

        kernel_ = kernel_.div(1e-25 + kernel_.sum(dim=1).unsqueeze(1).expand(1, self.kernel_size ** 2, 3, 224, 224))

        _, denoise = self.kernel_pred(original, kernel_.unsqueeze(0), 1.0)

        return denoise.squeeze(0).detach().cpu().numpy()

    def grad_kernel(self, original, kernel, mask, x_gradient):

        kernel.requires_grad_()
        mask.requires_grad_()

        x_gradient = torch.from_numpy(x_gradient)
        if x_gradient.is_contiguous() is False:
            x_gradient = x_gradient.contiguous()
        x_gradient = x_gradient.unsqueeze(0).cuda()

        original = torch.from_numpy(original).unsqueeze(0).unsqueeze(0).cuda()

        kernel_ = kernel.mul(mask.repeat(1, kernel.size()[1], 1, 1, 1)) + self.kernel_gt.mul(
            1 - mask.repeat(1, kernel.size()[1], 1, 1, 1))

        kernel_ = kernel_.div(1e-25 + kernel_.sum(dim=1).unsqueeze(1).expand(1, self.kernel_size ** 2, 3, 224, 224))

        _, denoise = self.kernel_pred(original, kernel_, 1.0)

        loss_fn = torch.nn.L1Loss(reduction='sum')

        l1loss_mask = self.mask_reg * loss_fn(mask, torch.zeros_like(mask))

        denoise.backward(x_gradient)
        l1loss_mask.backward()

        kernel_grad = kernel.grad
        mask_grad = mask.grad

        return kernel_grad, mask_grad


    def adapt_mask_gaussblur(self, original, mask_org, category, model):

        max_iterations = 150
        tv_beta = 3
        learning_rate = 0.05
        l1_coeff = self.mask_att_l1  # 2.0
        tv_coeff = 0.1
        mask_org = mask_org.permute(0, 3, 1, 2)

        mask_ = F.upsample(mask_org, (28, 28), mode='bilinear').detach().cpu()
        mask_ = mask_.cuda()
        mask_.requires_grad_()
        optimizer = torch.optim.Adam([mask_], lr=learning_rate)
        blurred = cv.GaussianBlur(original.reshape(-1, 224, 224).transpose(1, 2, 0), (11, 11), 5).transpose(2, 0, 1)

        original = torch.from_numpy(original).cuda().view(40, 3, 224, 224)
        blurred = torch.from_numpy(blurred).cuda().view(40, 3, 224, 224)

        def tv_norm(input, tv_beta):
            img = input
            row_grad = torch.mean(torch.abs((img[:, :, :-1, :] - img[:, :, 1:, :])).pow(tv_beta))
            col_grad = torch.mean(torch.abs((img[:, :, :, :-1] - img[:, :, :, 1:])).pow(tv_beta))
            return row_grad + col_grad

        for i in range(max_iterations):
            mask = F.upsample(mask_, (224, 224), mode='bilinear')
            # The single channel mask is used with an RGB image,
            # so the mask is duplicated to have 3 channel,
            mask = mask.repeat(1, 3, 1, 1)
            # 40, 3, 224, 224

            # Use the mask to perturbated the input image.
            perturbated_input = original.mul(1 - mask) + \
                                blurred.mul(mask)
            
            perturbated_input = for_normalize(perturbated_input.clone(), model.input_mean, model.input_std)
            outputs = torch.nn.Softmax()(model(perturbated_input))

            loss = l1_coeff * torch.mean(torch.abs(mask)) + tv_coeff * tv_norm(mask, tv_beta) + outputs[0, category]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if self.disp:
            #     vis = visdom.Visdom(env='Adversarial Example Showing')
            #     vis.images(mask.data.clamp_(0, 1), win='mask_')

            # Optional: clamping seems to give better results
            mask.data.clamp_(0., 1.)

        mask = F.upsample(mask_, (224, 224), mode='bilinear').permute(0, 2, 3, 1)
        mask.data.clamp_(0., 1.)
        mask[mask >= 0.5] = 1.
        mask[mask < 0.5] = 0.

        # if self.disp:
        #     vis.images(mask.permute(0,3,1,2), win='mask')
        print("\t\t mask_att max: ", torch.max(mask))
        return mask

    def add_flow_alpha_pert(self, original_, kernel_sz, theta_f, theta_b, alpha, mask):

        original = torch.from_numpy(original_).cuda()

        blur_steps = kernel_sz
        theta_org = torch.tensor([[
            [1., 0, 0],
            [0, 1., 0]
        ]]*40, dtype=torch.float).cuda()
        
        seg = mask.permute(0, 3, 1, 2).repeat(blur_steps, 1, 3, 1, 1) # bulr_steps, 40, 3, 224, 224
        original = original.view(40, 3, 224, 224).repeat(blur_steps, 1, 1, 1, 1)
        theta_f_diff = (theta_f - theta_org) / (blur_steps - 1)
        theta_b_diff = (theta_b - theta_org) / (blur_steps - 1)

        theta_f_diff = theta_f_diff.repeat(blur_steps, 1, 1, 1)
        theta_b_diff = theta_b_diff.repeat(blur_steps, 1, 1, 1)
        theta_range = torch.arange(0, blur_steps).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 40, 1, 1).cuda()
        theta_f_ = theta_f_diff * theta_range + theta_org
        theta_b_ = theta_b_diff * theta_range + theta_org

        flow_f = F.affine_grid(theta_f_.view(-1, 2, 3), original.view(-1, 3, 224, 224).size(), align_corners=True)
        flow_b = F.affine_grid(theta_b_.view(-1, 2, 3), original.view(-1, 3, 224, 224).size(), align_corners=True)

        seg_f = F.grid_sample(input=seg.view(-1, 3, 224, 224), grid=(flow_f), mode='bilinear', align_corners=True)
        seg_b = F.grid_sample(input=seg.view(-1, 3, 224, 224), grid=(flow_b), mode='bilinear', align_corners=True)

        seg_f = seg_f.view(blur_steps, 40, 3, 224, 224).permute(0, 1, 3, 4, 2)[:, :, :, :, 0].unsqueeze(-1)
        seg_b = seg_b.view(blur_steps, 40, 3, 224, 224).permute(0, 1, 3, 4, 2)[:, :, :, :, 0].unsqueeze(-1)

        u_seg_fb = seg_f + seg_b
        u_seg_fb[u_seg_fb > 1] = 1
        u_seg_fb[u_seg_fb < 0] = 0

        flow = flow_f.view(blur_steps, 40, 224, 224, 2) * u_seg_fb + flow_b.view(blur_steps, 40, 224, 224, 2) * (1 - u_seg_fb)

        tensorFlow = flow
        warped = F.grid_sample(input=original.view(-1, 3, 224, 224), grid=tensorFlow.view(-1, 224, 224, 2), mode='bilinear', align_corners=True)
        warped = warped.view(blur_steps, 40, 3, 224, 224)

        # regularize the warped to make weights out of image domain to be zero
        min_alpha = alpha.min(dim=0).values.unsqueeze(0).repeat(kernel_sz, 1, 1, 1, 1)
        max_alpha = alpha.max(dim=0).values.unsqueeze(0).repeat(kernel_sz, 1, 1, 1, 1)
        alpha = (alpha - min_alpha) / (max_alpha - min_alpha + 1e-25)
        regbyflow = flow.permute(0, 1, 4, 2, 3)
        for ri in range(2):
            alpha[regbyflow[:, :, ri, :, :].unsqueeze(2) > 1.] = 0
            alpha[regbyflow[:, :, ri, :, :].unsqueeze(2) < -1.] = 0
        alpha = alpha / (1e-25 + alpha.sum(dim=0))

        perted = warped.view(blur_steps, 40, 3, 224, 224) * alpha.view(blur_steps, 40, 1, 224, 224)  # [17,3,h,w]*[17,1,h,w]
        perted = perted.view(blur_steps, 40, 3, 224, 224)

        perted = perted.sum(dim=0)

        '''import imageio
        warped = warped.clone().view(-1, 3, 224, 224)
        warped = warped.detach().view(kernel_sz, 40, 3, 224, 224).cpu().permute(0, 1, 3, 4, 2).numpy() * 255.
        perted_img = perted.clone().view(-1, 3, 224, 224)
        perted_img = perted_img.detach().view(40, 3, 224, 224).cpu().permute(0, 2, 3, 1).numpy() * 255.
        os.makedirs("./ABBAwarped/batch_" + str(self.batch) + "/", exist_ok=True)
        for idx in range(0, 40, 8):
            imageio.imwrite("./ABBAwarped/batch_" + str(self.batch) + "/" + str(idx) + "_traned_img2.jpg",
                            warped[2, idx].astype(np.uint8))
            imageio.imwrite("./ABBAwarped/batch_" + str(self.batch) + "/" + str(idx) + "_traned_img6.jpg",
                            warped[6, idx].astype(np.uint8))
            imageio.imwrite("./ABBAwarped/batch_" + str(self.batch) + "/" + str(idx) + "_traned_img10.jpg",
                            warped[10, idx].astype(np.uint8))
            imageio.imwrite("./ABBAwarped/batch_" + str(self.batch) + "/" + str(idx) + "_traned_img15.jpg",
                            warped[15, idx].astype(np.uint8))
            imageio.imwrite("./ABBAwarped/batch_" + str(self.batch) + "/" + str(idx) + "_traned_img_mean.jpg",
                            np.mean(warped[:, idx, :, :, :], axis=0).astype(np.uint8))
            imageio.imwrite("./ABBAwarped/batch_" + str(self.batch) + "/" + str(idx) + "_perted.jpg",
                            perted_img[idx].astype(np.uint8))'''


        return perted.detach().cpu().numpy()

    def grad_flow_alpha_pert(self, original_, kernel_sz, theta_f, theta_b, alpha, mask, x_gradient_):

        original = torch.from_numpy(original_).cuda().view(40, 3, 224, 224)

        x_gradient = torch.from_numpy(x_gradient_)
        if x_gradient.is_contiguous() is False:
            x_gradient = x_gradient.contiguous()
        x_gradient = x_gradient.view(40, 3, 224, 224)

        blur_steps = kernel_sz

        theta_f.requires_grad_()
        theta_b.requires_grad_()
        alpha.requires_grad_()
        mask.requires_grad_()

        theta_org = torch.tensor([[
            [1., 0, 0],
            [0, 1., 0]
        ]]*40, dtype=torch.float).cuda()

        seg = mask.permute(0, 3, 1, 2).repeat(blur_steps, 1, 3, 1, 1)  # bulr_steps, 40, 3, 224, 224
        original = original.view(40, 3, 224, 224).repeat(blur_steps, 1, 1, 1, 1)
        theta_f_diff = (theta_f - theta_org) / (blur_steps - 1)
        theta_b_diff = (theta_b - theta_org) / (blur_steps - 1)

        theta_f_diff = theta_f_diff.repeat(blur_steps, 1, 1, 1)
        theta_b_diff = theta_b_diff.repeat(blur_steps, 1, 1, 1)
        theta_range = torch.arange(0, blur_steps).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 40, 1, 1).cuda()
        theta_f_ = theta_f_diff * theta_range + theta_org
        theta_b_ = theta_b_diff * theta_range + theta_org

        flow_f = F.affine_grid(theta_f_.view(-1, 2, 3), original.view(-1, 3, 224, 224).size(), align_corners=True)
        flow_b = F.affine_grid(theta_b_.view(-1, 2, 3), original.view(-1, 3, 224, 224).size(), align_corners=True)

        seg_f = F.grid_sample(input=seg.view(-1, 3, 224, 224), grid=(flow_f), mode='bilinear', align_corners=True)
        seg_b = F.grid_sample(input=seg.view(-1, 3, 224, 224), grid=(flow_b), mode='bilinear', align_corners=True)

        seg_f = seg_f.view(blur_steps, 40, 3, 224, 224).permute(0, 1, 3, 4, 2)[:, :, :, :, 0].unsqueeze(-1)
        seg_b = seg_b.view(blur_steps, 40, 3, 224, 224).permute(0, 1, 3, 4, 2)[:, :, :, :, 0].unsqueeze(-1)

        u_seg_fb = seg_f + seg_b
        u_seg_fb[u_seg_fb > 1] = 1
        u_seg_fb[u_seg_fb < 0] = 0

        flow = flow_f.view(blur_steps, 40, 224, 224, 2) * u_seg_fb + flow_b.view(blur_steps, 40, 224, 224, 2) * (
                    1 - u_seg_fb)

        tensorFlow = flow
        warped = F.grid_sample(input=original.view(-1, 3, 224, 224), grid=tensorFlow.view(-1, 224, 224, 2),
                               mode='bilinear', align_corners=True)
        warped = warped.view(blur_steps, 40, 3, 224, 224)
        perted = warped.view(blur_steps, 40, 3, 224, 224) * alpha.view(blur_steps, 40, 1, 224, 224)  # [17,3,h,w]*[17,1,h,w]
        perted = perted.view(blur_steps, 40, 3, 224, 224).sum(dim=0)

        perted.backward(x_gradient.cuda())

        alpha_grad_reg = alpha.grad

        theta_f_grad = theta_f.grad
        theta_b_grad = theta_b.grad

        theta_grad_mean = (theta_f_grad + theta_b_grad) / 2
        theta_f_grad = theta_grad_mean
        theta_b_grad = theta_grad_mean

        return theta_f_grad, theta_b_grad, alpha_grad_reg, mask.grad

    def init_flow_alpha(self, original, kernel_sz, init_mode="default"):

        alpha = torch.zeros(kernel_sz, 40, 1, 224, 224)

        alpha[:int(self.alpha_len), :, :, :, :] = 1 / self.alpha_len
        # print("alpha_len:{}".format(self.alpha_len))
        theta_f, theta_b = self.flow_estimate_saliency(original)

        alpha = alpha.cuda()

        return theta_f, theta_b, alpha, self.pred

    def flow_estimate_saliency(self, x_):

        if not hasattr(self, 'pred') and not os.path.exists(self.imgname):
            if not hasattr(self, 'net_saliecny'):
                self.net_saliecny = build_model("resnet").cuda()
                self.net_saliecny.eval()  # use_global_stats = True
                self.net_saliecny.apply(weights_init)
                self.net_saliecny.base.load_pretrained_model(
                    torch.load("./resnet50-run0-final.pth"))
                self.net_saliecny.load_state_dict(torch.load("./resnet50-run0-final.pth"))
                self.net_saliecny.eval()  # use_global_stats = True
                net_saliecny = self.net_saliecny

            else:
                net_saliecny = self.net_saliecny

            # forward pass
            x_ = x_ * 255.
            x_ = x_ - np.tile(np.array((122.67892, 116.66877, 104.00699)), 40)[:, np.newaxis, np.newaxis].reshape(40, 3, 1, 1)
            x = torch.Tensor(x_).cuda()
            x = x.unsqueeze(0)

            pred_ = net_saliecny(x.view(40, 3, 224, 224)).view(40, 1, 224, 224)
            pred_ = torch.sigmoid(pred_)
            pred = pred_
            pred[pred_ > 3e-1] = 1
            pred[pred_ <= 3e-1] = 0

            # sementic segmentation regularized flow
            self.pred = pred.permute(0, 2, 3, 1).detach()

            '''import imageio
            save_pred = pred_ * 255.
            for idx in range(40):
                one_pred = save_pred[idx, 0, :, :].cpu().detach().numpy().astype(np.uint8)
                os.makedirs("./ABBAtest_saliency/batch_" + str(self.batch) + "/", exist_ok=True)
                imageio.imwrite("./ABBAtest_saliency/batch_" + str(self.batch) + "/" + str(idx)+".jpg", one_pred)'''

            print("batch " + str(self.batch) + ": saliency detection succeeds!")


        theta_b = torch.tensor([[
            [1.0, 0., 0.2],
            [0., 1.0, 0.]
        ]]*40, dtype=torch.float).cuda()

        theta_f = torch.tensor([[
            [1.0, 0., 0.2],
            [0., 1.0, 0.]
        ]]*40, dtype=torch.float).cuda()

        return theta_f, theta_b


class LinfinityClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        min_, max_ = a.bounds()
        s = max_ - min_
        clipped = np.clip(perturbation, -epsilon * s, epsilon * s)
        return clipped


class LinfinityDistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, Linfinity):
            logging.warning(
                "Running an attack that tries to minimize the"
                " Linfinity norm of the perturbation without"
                " specifying foolbox.distances.Linfinity as"
                " the distance metric might lead to suboptimal"
                " results."
            )


class GDOptimizerMixin(object):
    def _create_optimizer(self, a, stepsize):
        return GDOptimizer(stepsize)


class MomentumIterativeAttack(
    LinfinityClippingMixin,
    LinfinityDistanceCheckMixin,
    GDOptimizerMixin,
    IterativeProjectedGradientBaseAttack,
):
    """The Momentum Iterative Method attack
    introduced in [1]_. It's like the Basic
    Iterative Method or Projected Gradient
    Descent except that it uses momentum.

    References
    ----------
    .. [1] Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su,
           Jun Zhu, Xiaolin Hu, Jianguo Li, "Boosting Adversarial
           Attacks with Momentum",
           https://arxiv.org/abs/1710.06081

    """

    def _gradient(self, a, x, class_, strict=True, gradient_args={}):
        # get current gradient
        gradient = yield from a.gradient_one(x, class_, strict=strict)
        gradient = gradient / max(1e-12, np.mean(np.abs(gradient)))

        # combine with history of gradient as new history
        self._momentum_history = self._decay_factor * self._momentum_history + gradient

        # use history
        gradient = self._momentum_history
        gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient

    @generator_decorator
    def as_generator(
            self,
            batch,
            a,
            epsilon=0.3,
            stepsize=0.06,
            iterations=10,
            decay_factor=1.0,
            random_start=False,
            return_early=True,
            pert_type="Add",
            blur_model="blur_model",
            numSP=-1,
            mask_att_l1=2.0,
            direction=None,
            imgname=None
    ):

        """Momentum-based iterative gradient attack known as
        Momentum Iterative Method.

        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        binary_search : bool
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        decay_factor : float
            Decay factor used by the momentum term.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        if isinstance(epsilon, np.ndarray):
            for x in epsilon.tolist():
                assert x >= 0
        else:
            assert epsilon > 0

        self._decay_factor = decay_factor

        yield from self._run(
            batch, a, epsilon, stepsize, iterations, random_start, return_early, pert_type=pert_type, \
            blur_model=blur_model, numSP=numSP, mask_att_l1=mask_att_l1, direction=direction, imgname=imgname
        )



def run_attack_mifgsm(batch, model, image, label, pert_type, imgname, eplison, blur_model, step_size=5, numSP=-1,
                      mask_att_l1=2.0, direction=None):
    # apply the attack
    distance = Linfinity
    attack = MomentumIterativeAttack(model=model, distance=distance)
    adversarial = attack(batch, image, label,
                         epsilon=eplison,
                         stepsize=step_size,
                         iterations=10,
                         random_start=False,
                         return_early=True,
                         unpack=False,
                         pert_type=pert_type,
                         blur_model=blur_model,
                         numSP=numSP,
                         mask_att_l1=mask_att_l1,
                         direction=direction,
                         imgname=imgname)

    advs = [a.perturbed for a in adversarial]
    advs = [
        p if p is not None else np.full_like(u, np.nan)
        for p, u in zip(advs, image)
    ]
    perturbed_image = np.stack(advs)

    diff = np.linalg.norm(perturbed_image - image)
    if diff == 0:
        status = 0
    else:
        if adversarial[0].adversarial_class is not None:
            status = 1
        else:
            status = -1

    return perturbed_image, status


def main():
    global args, best_prec1
    args = parser.parse_args()

    print('dataset: ' + str(args.dataset))
    print('attack type: ' + str(args.attack_type))
    if not args.attack_type == "0":
        if args.attack_type == "999":
            print('pgd times: ' + str(args.pgd_times))
            print('pgd weight: ' + str(args.pgd_weight))
            print('pgd L2: ' + str(args.pgdL2))
        elif args.attack_type == "888":
            print('multi times: ' + str(args.pgd_times))
            print('multi weight: ' + str(args.pgd_weight))
            print('multi L2: ' + str(args.pgdL2))
        elif args.attack_type == "777":
            print("Gaussian Blur")
        elif args.attack_type == "666":
            print("fgsm weight: " + str(args.pgd_weight))
        elif args.attack_type == "555":
            print("ABBA")
        elif args.attack_type == "2":
            print('motion len: ' + str(args.motion_len))
            print('motion val: ' + str(args.motion_val))
            print('val type: ' + args.gw_type)
        else:
            print('motion len: ' + str(args.motion_len))
            print('motion val: ' + str(args.motion_val))
    else:
        args.motion_len = 0
    print("=============================")
    if "4" in args.attack_type:
        args.print_freq = 10
        print('frame times: ' + str(args.frame_times))
        print('frame weight: ' + str(args.frame_weight))
    print("=============================")
    if "3" in args.attack_type:
        print('step times: ' + str(args.step_times))
        print('step weight: ' + str(args.step_weight))
        print('step start: ' + str(args.step_start))
    print("=============================")

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)
    full_arch_name = args.arch
    if not args.attack_type == "0":
        if args.attack_type == "555":
            args.store_name += "ABBA" + args.attack_type + "="
        elif args.attack_type == "999":
            args.store_name += "att" + args.attack_type + "=" + str(args.pgd_times) + "-" + str(
                args.pgd_weight).replace(".", "")[:3]
            if args.pgdL2:
                args.store_name += "=L2"
        elif args.attack_type == "888":
            args.store_name += "multi" + args.attack_type + "=" + str(args.pgd_times) + "-" + str(
                args.pgd_weight).replace(".", "")[:3]
            if args.pgdL2:
                args.store_name += "=L2"
        elif args.attack_type == "2":
            args.store_name += "att2=" + args.gw_type
        else:
            args.store_name += "att" + args.attack_type + "m" + str(args.motion_len) + "=v" + str(
                args.motion_val).replace(".", "")[:3] + "="
        if "4" in args.attack_type:
            args.store_name += str(args.frame_times) + "-" + str(args.frame_weight).replace(".", "")[:2] + "="
        if "3" in args.attack_type:
            args.store_name += str(args.step_start).replace(".", "")[:2] + "-" + str(
                args.step_start + args.step_times * args.step_weight).replace(".", "")[:2] + "="
    args.store_name = '_'.join(
        [args.store_name, str(args.attack_type), '_TDN_', args.dataset, args.modality, full_arch_name,
         args.consensus_type, 'segment%d' % args.num_segments, 'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)

    check_rootfolders()

    logger = setup_logger(output=os.path.join(args.root_model, args.store_name, "log"),
                          distributed_rank=0,
                          name=f'TDN')
    logger.info('storing name: ' + args.store_name)

    model = TSN(num_class,
                args.num_segments,
                args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                fc_lr5=(args.tune_from and args.dataset in args.tune_from))

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    for group in policies:
        logger.info(
            ('[TDN-{}]group: {} has {} params, lr_mult: {}, decay_mult: {}'.
             format(args.arch, group['name'], len(group['params']),
                    group['lr_mult'], group['decay_mult'])))

    train_augmentation = model.get_augmentation(
        flip=False if 'something' in args.dataset else True)

    cudnn.benchmark = True

    # Data loading code
    normalize = GroupNormalize(input_mean, input_std)

    train_dataset = TSNDataSet(
        args.dataset,
        args.root_path,
        args.train_list,
        num_segments=args.num_segments,
        modality=args.modality,
        image_tmpl=prefix,
        transform=torchvision.transforms.Compose([train_augmentation,
                                                  Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                                                  ToTorchFormatTensor(
                                                      div=(args.arch not in ['BNInception', 'InceptionV3'])),
                                                  normalize, ]),
        dense_sample=args.dense_sample)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # train_sampler = torch.utils.data.SequentialSampler(train_dataset)
    # for i in train_sampler:
    #    print(i, end=',')
    # print("\n")
    # train_sampler = torch.utils.data.RandomSampler(train_dataset)
    # for i in train_sampler:
    #    print(i, end=',')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, num_workers=args.workers,
                                               pin_memory=True, shuffle=True, drop_last=True)

    val_dataset = TSNDataSet(
        args.dataset,
        args.root_path,
        args.val_list,
        num_segments=args.num_segments,
        modality=args.modality,
        image_tmpl=prefix,
        random_shift=False,
        transform=torchvision.transforms.Compose([
            GroupScale(int(scale_size)), GroupCenterCrop(crop_size),
            Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
            normalize, ]),
        dense_sample=args.dense_sample,
        new_length=args.new_length, addition_length=args.motion_len)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size, num_workers=args.workers,
                                             pin_memory=True, sampler=None, shuffle=False, drop_last=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    optimizer = torch.optim.SGD(policies, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = get_scheduler(optimizer, len(train_loader), args)

    model = model.cuda()

    global zeroinput, oneinput
    zeroinput = torch.zeros(3, 224, 224).cuda()
    oneinput = torch.ones(3, 224, 224).cuda()
    for z, o, m, s in zip(zeroinput, oneinput, model.input_mean, model.input_std):
        z.sub_(m).div_(s)
        o.sub_(m).div_(s)
    zeroinput = zeroinput.repeat(args.batch_size, 40, 1, 1)
    oneinput = oneinput.repeat(args.batch_size, 40, 1, 1)
    # print(zeroinput)
    # print(oneinput)
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info(("=> loaded checkpoint '{}' (epoch {})".format(
                args.evaluate, checkpoint['epoch'])))
        else:
            logger.info(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        logger.info(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        if (args.dataset == 'hmdb51' or args.dataset == 'ucf101') and (
                'v1' in args.tune_from or 'v2' in args.tune_from):
            sd = {k.replace('module.base_model.', 'base_model.'): v for k, v in sd.items()}
            sd = {k.replace('module.new_fc.', 'new_fc.'): v for k, v in sd.items()}
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                logger.info('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                logger.info('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        logger.info('#### Notice: keys loaded but not in models: {}'.format(keys1 - keys2))
        logger.info('#### Notice: keys required but not in pre-models: {}'.format(keys2 - keys1))
        if args.dataset not in args.tune_from:  # new dataset
            logger.info('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    with open(os.path.join(args.root_model, args.store_name, "log", 'args.txt'), 'w') as f:
        f.write(str(args))

    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_model, args.store_name, "log"))

    if args.evaluate:
        logger.info(("===========evaluate==========="))

        prec1, prec5, val_loss = validate(val_loader, model, criterion, logger)

        is_best = prec1 > best_prec1
        best_prec1 = prec1
        logger.info(("Best Prec@1: '{}'".format(best_prec1)))
        save_epoch = args.start_epoch + 1
        save_checkpoint(
            {
                'epoch': args.start_epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'prec1': prec1,
                'best_prec1': best_prec1,
            }, save_epoch, is_best)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train_loader.sampler.set_epoch(epoch)
        train_loss, train_top1, train_top5 = train(train_loader, model, criterion, optimizer, epoch=epoch,
                                                   logger=logger, scheduler=scheduler)

        tf_writer.add_scalar('loss/train', train_loss, epoch)
        tf_writer.add_scalar('acc/train_top1', train_top1, epoch)
        tf_writer.add_scalar('acc/train_top5', train_top5, epoch)
        tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1, prec5, val_loss = validate(val_loader, model, criterion, logger, epoch=epoch)

            tf_writer.add_scalar('loss/test', val_loss, epoch)
            tf_writer.add_scalar('acc/test_top1', prec1, epoch)
            tf_writer.add_scalar('acc/test_top5', prec5, epoch)

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

            logger.info(("Best Prec@1: '{}'".format(best_prec1)))
            tf_writer.flush()
            save_epoch = epoch + 1
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'prec1': prec1,
                    'best_prec1': best_prec1,
                }, save_epoch, is_best)


def train(train_loader, model, criterion, optimizer, epoch, logger=None, scheduler=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.partialBN(False)
    else:
        model.partialBN(True)

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        output = model(input_var)
        loss = criterion(output, target_var)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))  # TODO

    logger.info(('Training Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                 .format(top1=top1, top5=top5, loss=losses)))

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, logger=None, epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_att = AverageMeter()
    top5_att = AverageMeter()
    losses_att = AverageMeter()

    model.eval()
    # [bs, 8*5*3*3, H, W] 正常是8段*连续5帧 需要往前补10帧 所以是8段*15帧
    indices = [(k * (args.motion_len + args.new_length) + j) * 3 + i for k in range(0, args.num_segments) for j in
               range(args.motion_len, args.motion_len + args.new_length) for i in range(0, 3)]
    indices = torch.tensor(indices).cuda()
    end = time.time()

    aa = torch.zeros(1, 1)
    temp = 0
    cnt = 0
    tot_status = [0, 0, 0]
    for i, (addition_input, target) in enumerate(val_loader):
        if i == 0:
            print("total input size: ", addition_input.size())
            print("frame indices:", indices)

        if args.attack_type == "0":
            with torch.no_grad():
                target = target.cuda()
                addition_input = addition_input.cuda()
                input = torch.index_select(addition_input, dim=1, index=indices)

                output = model(input)
                loss = criterion(output, target).cuda()
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    logger.info(
                        ('Test: [{0}/{1}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5)))

        elif args.attack_type == "999":
            target = target.cuda()
            addition_input = addition_input.cuda()
            input = torch.index_select(addition_input, dim=1, index=indices)
            input.requires_grad = True
            output = model(input)
            loss = criterion(output, target).cuda()
            loss.backward()
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                logger.info(
                    ('Test: [{0}/{1}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5)))

            pertu = torch.zeros(input.size(0), 120, 224, 224).cuda()
            for pgd_e in range(args.pgd_times):
                pertu = Variable(pertu.clone().detach(), requires_grad=True).cuda()
                if not args.pgdL2:
                    opt_adv = Linf_SGD([pertu], lr=args.pgd_weight)
                    opt_adv.zero_grad()

                model.zero_grad()

                adv_output = model(input + pertu)
                loss = (- criterion(adv_output, target)).cuda()

                loss.backward()
                if not args.pgdL2:
                    opt_adv.step()
                else:
                    g = pertu.grad.data
                    with torch.no_grad():
                        norm2 = torch.norm(g, 2, dim=(-2, -1), keepdim=True)
                        norm2 = norm2 + 1e-16
                        step = 1. / norm2 * g
                        pertu = pertu - args.pgd_weight * step

                # print(adv_input.grad)
                if i == 0 or i == 200:
                    print(loss)

                model.zero_grad()
                if not args.pgdL2:
                    opt_adv.zero_grad()

                if not args.pgdL2:
                    pertu = torch.clamp(pertu, min=-(oneinput - zeroinput) * 15. / 255,
                                        max=(oneinput - zeroinput) * 15. / 255)
                else:
                    now_norm = torch.norm(pertu, 2, dim=(-2, -1), keepdim=True)
                    now_norm = now_norm.repeat(1, 1, 224, 224)
                    pertu = torch.where(now_norm > 10., pertu / now_norm * 10., pertu)

            adv_output = model(input + pertu)
            prec1_att, prec5_att = accuracy(adv_output.data, target, topk=(1, 5))
            loss = (- criterion(adv_output, target)).cuda()

            losses_att.update(loss.item(), input.size(0))
            top1_att.update(prec1_att.item(), input.size(0))
            top5_att.update(prec5_att.item(), input.size(0))

            if i % args.print_freq == 0:
                logger.info(
                    ('Attack Test: [{0}/{1}]\t'
                     'Prec_att@1 {top1_att.val:.3f} ({top1_att.avg:.3f})\t'
                     'Prec_att@5 {top5_att.val:.3f} ({top5_att.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        i, len(val_loader), top1_att=top1_att, top5_att=top5_att, loss=losses_att)))

            saveImg = "/media/hd0/wuguoming/att-mb/" + args.dataset + "/pgd=" + str(args.pgd_times) + "-" + str(
                args.pgd_weight).replace(".", "")[:3]
            if args.pgdL2:
                saveImg = saveImg + "=L2"
            saveImg = saveImg + "/"
            if args.full_saveimg:
                img_adv = back_normalize((input.clone().detach() + pertu.clone().detach()).view(-1, 3, 224, 224),
                                         model.input_mean, model.input_std).cuda()
                img_adv = img_adv.to(torch.device('cpu'))
                frame_tot = img_adv.size(0) // args.batch_size
                for bs in range(args.batch_size):
                    imgpath = saveImg + str(i * args.batch_size + bs) + "/"
                    for frame_idx in range(frame_tot):
                        if not os.path.exists(imgpath):
                            os.makedirs(imgpath)
                        torchvision.utils.save_image(img_adv[bs * frame_tot + frame_idx],
                                                     imgpath + str(frame_idx) + ".jpg")

        elif args.attack_type == "888":
            target = target.cuda()
            addition_input = addition_input.cuda()
            input = torch.index_select(addition_input, dim=1, index=indices)
            input.requires_grad = True
            output = model(input)
            loss = criterion(output, target).cuda()
            loss.backward()
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                logger.info(
                    ('Test: [{0}/{1}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5)))

            if args.pgdL2:
                pertu = torch.ones(input.size(0), 120, 224, 224).cuda()
            else:
                pertu = torch.zeros(input.size(0), 120, 224, 224).cuda()
            for pgd_e in range(args.pgd_times):
                pertu = Variable(pertu.clone().detach(), requires_grad=True).cuda()
                if not args.pgdL2:
                    opt_adv = Linf_SGD([pertu], lr=args.pgd_weight)
                    opt_adv.zero_grad()

                model.zero_grad()

                adv_output = model(torch.mul(input, pertu))
                loss = (- criterion(adv_output, target)).cuda()

                loss.backward()
                if not args.pgdL2:
                    g = pertu.grad.data
                    g = torch.sign(g)
                    with torch.no_grad():
                        pertu = pertu.mul_(torch.pow(1 / args.pgd_weight, g))
                        tmp = torch.mul(input, pertu).data
                        tmp = torch.clamp(tmp, zeroinput, oneinput)
                        pertu = torch.div(tmp, input)
                        pertu = torch.where(torch.isinf(pertu), torch.full_like(pertu, 1), pertu)

                else:
                    g = pertu.grad.data
                    with torch.no_grad():
                        norm2 = torch.norm(g, 2, dim=(-2, -1), keepdim=True)
                        norm2 = norm2 + 1e-16
                        step = 1. / norm2 * g
                        pertu = pertu.mul_(torch.pow(1 / args.pgd_weight, step))

                # print(adv_input.grad)
                if i == 0 or i == 200:
                    print(loss)

                model.zero_grad()

                if not args.pgdL2:
                    pertu = torch.clamp(pertu, min=0.9, max=1.1)
                else:
                    now_norm = torch.norm(pertu, 2, dim=(-2, -1), keepdim=True)
                    now_norm = now_norm.repeat(1, 1, 224, 224)
                    pertu = torch.where(now_norm > 226., pertu / now_norm * 226., pertu)

            adv_output = model(torch.mul(input, pertu))
            prec1_att, prec5_att = accuracy(adv_output.data, target, topk=(1, 5))
            loss = (- criterion(adv_output, target)).cuda()

            losses_att.update(loss.item(), input.size(0))
            top1_att.update(prec1_att.item(), input.size(0))
            top5_att.update(prec5_att.item(), input.size(0))

            if i % args.print_freq == 0:
                logger.info(
                    ('Attack Test: [{0}/{1}]\t'
                     'Prec_att@1 {top1_att.val:.3f} ({top1_att.avg:.3f})\t'
                     'Prec_att@5 {top5_att.val:.3f} ({top5_att.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        i, len(val_loader), top1_att=top1_att, top5_att=top5_att, loss=losses_att)))

            saveImg = "/media/hd0/wuguoming/att-mb/" + args.dataset + "/multi=" + str(args.pgd_times) + "-" + str(
                args.pgd_weight).replace(".", "")[:3]
            if args.pgdL2:
                saveImg = saveImg + "=L2"
            saveImg = saveImg + "/"
            if args.full_saveimg:
                img_adv = back_normalize((torch.mul(input, pertu).clone().detach()).view(-1, 3, 224, 224),
                                         model.input_mean, model.input_std).cuda()
                img_adv = img_adv.to(torch.device('cpu'))
                frame_tot = img_adv.size(0) // args.batch_size
                for bs in range(args.batch_size):
                    imgpath = saveImg + str(i * args.batch_size + bs) + "/"
                    for frame_idx in range(frame_tot):
                        if not os.path.exists(imgpath):
                            os.makedirs(imgpath)
                        torchvision.utils.save_image(img_adv[bs * frame_tot + frame_idx],
                                                     imgpath + str(frame_idx) + ".jpg")

        elif args.attack_type == "777":
            target = target.cuda()
            addition_input = addition_input.cuda()
            input = torch.index_select(addition_input, dim=1, index=indices)
            input.requires_grad = True
            output = model(input)
            loss = criterion(output, target).cuda()
            loss.backward()
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                logger.info(
                    ('Test: [{0}/{1}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5)))

            with torch.no_grad():
                gaussian = torchvision.transforms.GaussianBlur(kernel_size=11, sigma=(0.8, 1.5))
                pertu = gaussian(input) - input

                adv_output = model(input + pertu)
                prec1_att, prec5_att = accuracy(adv_output.data, target, topk=(1, 5))
                loss = (- criterion(adv_output, target)).cuda()

                losses_att.update(loss.item(), input.size(0))
                top1_att.update(prec1_att.item(), input.size(0))
                top5_att.update(prec5_att.item(), input.size(0))

            if i % args.print_freq == 0:
                logger.info(
                    ('Attack Test: [{0}/{1}]\t'
                     'Prec_att@1 {top1_att.val:.3f} ({top1_att.avg:.3f})\t'
                     'Prec_att@5 {top5_att.val:.3f} ({top5_att.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        i, len(val_loader), top1_att=top1_att, top5_att=top5_att, loss=losses_att)))

            saveImg = "/media/hd0/wuguoming/att-mb/" + args.dataset + "/gaussian="

            saveImg = saveImg + "/"
            if args.full_saveimg:
                img_adv = back_normalize((input + pertu).clone().detach().view(-1, 3, 224, 224),
                                         model.input_mean, model.input_std).cuda()
                img_adv = img_adv.to(torch.device('cpu'))
                frame_tot = img_adv.size(0) // args.batch_size
                for bs in range(args.batch_size):
                    imgpath = saveImg + str(i * args.batch_size + bs) + "/"
                    for frame_idx in range(frame_tot):
                        if not os.path.exists(imgpath):
                            os.makedirs(imgpath)
                        torchvision.utils.save_image(img_adv[bs * frame_tot + frame_idx],
                                                     imgpath + str(frame_idx) + ".jpg")

        elif args.attack_type == "666":
            target = target.cuda()
            addition_input = addition_input.cuda()
            input = torch.index_select(addition_input, dim=1, index=indices)
            input.requires_grad = True
            output = model(input)
            loss = criterion(output, target).cuda()
            loss.backward()
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                logger.info(
                    ('Test: [{0}/{1}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5)))

            pertu = torch.zeros(input.size(0), 120, 224, 224).cuda()

            pertu = Variable(pertu.clone().detach(), requires_grad=True).cuda()

            opt_adv = Linf_SGD([pertu], lr=args.pgd_weight)
            opt_adv.zero_grad()

            model.zero_grad()

            adv_output = model(input + pertu)
            loss = (- criterion(adv_output, target)).cuda()

            loss.backward()
            opt_adv.step()
            model.zero_grad()
            opt_adv.zero_grad()
            pertu = torch.clamp(pertu, min=-(oneinput - zeroinput) * 4. / 255,
                                max=(oneinput - zeroinput) * 4. / 255)

            adv_output = model(input + pertu)
            prec1_att, prec5_att = accuracy(adv_output.data, target, topk=(1, 5))
            loss = (- criterion(adv_output, target)).cuda()

            losses_att.update(loss.item(), input.size(0))
            top1_att.update(prec1_att.item(), input.size(0))
            top5_att.update(prec5_att.item(), input.size(0))

            if i % args.print_freq == 0:
                logger.info(
                    ('Attack Test: [{0}/{1}]\t'
                     'Prec_att@1 {top1_att.val:.3f} ({top1_att.avg:.3f})\t'
                     'Prec_att@5 {top5_att.val:.3f} ({top5_att.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        i, len(val_loader), top1_att=top1_att, top5_att=top5_att, loss=losses_att)))

            saveImg = "/media/hd0/wuguoming/att-mb/" + args.dataset + "/fgsm=" + str(args.pgd_weight).replace(".", "")[
                                                                                 :3]
            saveImg = saveImg + "/"
            if args.full_saveimg:
                img_adv = back_normalize((input.clone().detach() + pertu.clone().detach()).view(-1, 3, 224, 224),
                                         model.input_mean, model.input_std).cuda()
                img_adv = img_adv.to(torch.device('cpu'))
                frame_tot = img_adv.size(0) // args.batch_size
                for bs in range(args.batch_size):
                    imgpath = saveImg + str(i * args.batch_size + bs) + "/"
                    for frame_idx in range(frame_tot):
                        if not os.path.exists(imgpath):
                            os.makedirs(imgpath)
                        torchvision.utils.save_image(img_adv[bs * frame_tot + frame_idx],
                                                     imgpath + str(frame_idx) + ".jpg")

        elif args.attack_type == "555":

            target = target.cuda()
            addition_input = addition_input.cuda()
            input = torch.index_select(addition_input, dim=1, index=indices)
            # print(input.size())

            # [bs, 8*5*3channel, 224, 224]
            # input.requires_grad = True

            model.zero_grad()
            ###############################

            input = back_normalize(input.clone().detach(), model.input_mean, model.input_std).view(1, 40, 3, 224, 224)
            input = input.detach().cpu().numpy()
            target = target.detach().cpu().numpy()

            st_mb = time.time()
            adversarial, status = run_attack_mifgsm(i, model, input, target, "Blur",
                                                    os.path.join("./ABBAtest" + "_saliency",
                                                                 "image_name" + "_saliency.jpg"),
                                                    eplison=np.array([0.1, 15]),  # np.array([0.5,10])
                                                    blur_model="bg_obj_att",
                                                    step_size=10,  # 20
                                                    numSP=-3,
                                                    mask_att_l1=2.0,
                                                    direction=None)  # np.array([2., 2.]) None np.array([0., 0.])
            print("one time: ", time.time() - st_mb)
            tot_status[status] += 1
            if i % 100 == 0:
                logger.info(
                    ('ABBA: [{}/{}]\t'
                     'status: {}\t'
                     'FR: {}, not attack: {}, fail: {}'.format(
                        i, len(val_loader), status,
                        tot_status[0]/(i+1), tot_status[1]/(i+1), tot_status[2]/(i+1))))

            ########################################################
           
        else:
            with torch.no_grad():
                target = target.cuda()
                addition_input = addition_input.cuda()
                input = torch.index_select(addition_input, dim=1, index=indices)

                frameval = torch.ones([input.size()[0], args.motion_len]).cuda()  # [farther ... closer]
                stdval = torch.div(frameval * args.motion_val, torch.sum(frameval, dim=1).view(-1, 1))
                motioninput = input.clone() * (1 - args.motion_val)
                input_roll = addition_input.clone()
                for step in range(1, args.motion_len + 1):
                    input_roll = torch.roll(input_roll, 3, 1)
                    motioninput += torch.index_select(input_roll, dim=1, index=indices) * stdval[:,
                                                                                          args.motion_len - step].view(
                        input.size()[0], 1, 1, 1)  # dim=1 frames*3的维度 移动3位（3通道）
                motionpertu = motioninput - input.clone()

            input.requires_grad = True
            output = model(input)
            loss = criterion(output, target).cuda()
            loss.backward()
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                logger.info(
                    ('Test: [{0}/{1}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5)))

            attgrad = input.grad.data * 1e6
            model.zero_grad()
            pertu = motionpertu
            if args.attack_type == "1":  # 只取同向
                temp = torch.mul(torch.sign(attgrad), motionpertu)  # 注意 这里有偏移，因为标准化后不是以0为对称
                temp = torch.clamp(input=temp, min=0)
                pertu = torch.where(temp > args.zero_eps, motionpertu, temp)
            elif "2" in args.attack_type:
                if "4" in args.attack_type:
                    st = time.time()
                    model.zero_grad()
                    frameval = torch.ones([input.size()[0], args.motion_len]).cuda()
                    frameval.requires_grad = True
                    stdval = torch.div(frameval * args.motion_val, torch.sum(frameval, dim=1).view(-1, 1))
                    motioninput = input.clone() * (1 - args.motion_val)
                    input_roll = addition_input.clone().detach()
                    for step in range(1, args.motion_len + 1):
                        input_roll = torch.roll(input_roll, 3, 1)
                        motioninput = motioninput + torch.index_select(input_roll, dim=1,
                                                                       index=indices) * stdval[:,
                                                                                        args.motion_len - step].view(
                            input.size()[0], 1, 1, 1)
                    motionpertu = motioninput - input.clone().detach()

                    temp = torch.mul(torch.sign(attgrad), motionpertu.clone().detach())
                    temp = torch.clamp(input=temp, min=0)
                    temp = torch.where(temp > 0, 1.2, 0)

                    attgrad = attgrad.view(attgrad.size()[0], attgrad.size()[1], 224 * 224)
                    va = torch.max(input=attgrad, dim=2)[0]
                    vi = torch.min(input=attgrad, dim=2)[0]
                    attgrad = attgrad.view(attgrad.size()[0], attgrad.size()[1], 224, 224)
                    va = va.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                    vi = vi.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                    # normgrad = torch.maximum(torch.abs(va), torch.abs(vi))
                    normgrad = (torch.abs(va) + torch.abs(vi)) / 2
                    gw = torch.div(attgrad, normgrad)

                    if "3" in args.attack_type:
                        gw = gw.view(attgrad.size()[0], attgrad.size()[1], 224 * 224)
                        ave = torch.mean(gw, dim=2)
                        va = torch.max(gw, dim=2)[0]
                        vi = torch.min(gw, dim=2)[0]
                        ave = ave.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                        va = va.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                        vi = vi.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                        gw = gw.view(attgrad.size()[0], attgrad.size()[1], 224, 224)
                        val3 = 1 + torch.div(torch.abs((va + vi) / 2 - ave), va - vi)
                        gw = torch.mul(gw, val3)

                    gw = torch.clamp(gw, min=-1, max=1)
                    gw = 1 - torch.abs(gw)

                    attgrad = attgrad.view(attgrad.size()[0], attgrad.size()[1], 224 * 224)
                    motionpertu = motionpertu.view(attgrad.size()[0], attgrad.size()[1], 224 * 224)
                    if "1" in args.gw_type:
                        cntmp = torch.sign(motionpertu.clone().detach())
                        cntmp = cntmp.view(attgrad.size()[0], attgrad.size()[1], 224 * 224)
                        cntmp0 = torch.clamp(cntmp, min=-1, max=0)
                        cnt0 = torch.sum(torch.abs(cntmp0), dim=2)
                        cnt0 = cnt0.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                        cntmp1 = torch.clamp(cntmp, min=0, max=1)
                        cnt1 = torch.sum(cntmp1, dim=2)
                        cnt1 = cnt1.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                        val1 = torch.minimum(cnt0, cnt1) / torch.maximum(cnt0, cnt1)
                        gw = torch.mul(gw, val1)
                    if "2" in args.gw_type:
                        cntmp = torch.sign(attgrad)
                        cntmp = cntmp.view(attgrad.size()[0], attgrad.size()[1], 224 * 224)
                        cntmp0 = torch.clamp(cntmp, min=-1, max=0)
                        cnt0 = torch.sum(torch.abs(cntmp0), dim=2)
                        cnt0 = cnt0.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                        cntmp1 = torch.clamp(cntmp, min=0, max=1)
                        cnt1 = torch.sum(cntmp1, dim=2)
                        cnt1 = cnt1.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                        val2 = torch.maximum(cnt0, cnt1) / torch.minimum(cnt0, cnt1)
                        gw = torch.mul(gw, val2)

                    gw = torch.clamp(gw, max=1)
                    gw = gw * 0.8
                    motionpertu = motionpertu.view(attgrad.size()[0], attgrad.size()[1], 224, 224)
                    attgrad = attgrad.view(attgrad.size()[0], attgrad.size()[1], 224, 224)

                    # if i % 100 == 0:
                    # print(gw)
                    gw = torch.maximum(gw, temp)
                    pertu = torch.mul(motionpertu, gw)

                    output = model(input + pertu)
                    loss = criterion(output, target).cuda()
                    loss.backward()

                    fg = frameval.grad.data

                    model.zero_grad()
                    for e in range(args.frame_times):
                        with torch.no_grad():
                            framevalgrad = torch.sign(fg) * args.frame_weight
                            frameval = frameval + framevalgrad / args.step_times
                            frameval = torch.clamp(frameval, min=0)

                        frameval.requires_grad = True
                        stdval = torch.div(frameval * args.motion_val, torch.sum(frameval, dim=1).view(-1, 1))
                        motioninput = input.clone() * (1 - args.motion_val)
                        input_roll = addition_input.clone().detach()
                        for step in range(1, args.motion_len + 1):
                            input_roll = torch.roll(input_roll, 3, 1)
                            motioninput = motioninput + torch.index_select(input_roll, dim=1,
                                                                           index=indices) * stdval[:,
                                                                                            args.motion_len - step].view(
                                input.size()[0], 1, 1, 1)
                        motionpertu = motioninput - input.clone().detach()

                        temp = torch.mul(torch.sign(attgrad), motionpertu.clone().detach())
                        temp = torch.clamp(input=temp, min=0)
                        temp = torch.where(temp > 0, 1.2, 0)

                        attgrad = attgrad.view(attgrad.size()[0], attgrad.size()[1], 224 * 224)
                        va = torch.max(input=attgrad, dim=2)[0]
                        vi = torch.min(input=attgrad, dim=2)[0]
                        attgrad = attgrad.view(attgrad.size()[0], attgrad.size()[1], 224, 224)
                        va = va.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                        vi = vi.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                        # normgrad = torch.maximum(torch.abs(va), torch.abs(vi))
                        normgrad = (torch.abs(va) + torch.abs(vi)) / 2
                        gw = torch.div(attgrad, normgrad)

                        if "3" in args.gw_type:
                            gw = gw.view(attgrad.size()[0], attgrad.size()[1], 224 * 224)
                            ave = torch.mean(gw, dim=2)
                            va = torch.max(gw, dim=2)[0]
                            vi = torch.min(gw, dim=2)[0]
                            ave = ave.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                            va = va.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                            vi = vi.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                            gw = gw.view(attgrad.size()[0], attgrad.size()[1], 224, 224)
                            val3 = 1 + torch.div(torch.abs((va + vi) / 2 - ave), va - vi)
                            gw = torch.mul(gw, val3)

                        gw = torch.clamp(gw, min=-1, max=1)
                        gw = 1 - torch.abs(gw)

                        attgrad = attgrad.view(attgrad.size()[0], attgrad.size()[1], 224 * 224)
                        motionpertu = motionpertu.view(attgrad.size()[0], attgrad.size()[1], 224 * 224)
                        if "1" in args.gw_type:
                            cntmp = torch.sign(motionpertu.clone().detach())
                            cntmp = cntmp.view(attgrad.size()[0], attgrad.size()[1], 224 * 224)
                            cntmp0 = torch.clamp(cntmp, min=-1, max=0)
                            cnt0 = torch.sum(torch.abs(cntmp0), dim=2)
                            cnt0 = cnt0.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                            cntmp1 = torch.clamp(cntmp, min=0, max=1)
                            cnt1 = torch.sum(cntmp1, dim=2)
                            cnt1 = cnt1.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                            val1 = torch.minimum(cnt0, cnt1) / torch.maximum(cnt0, cnt1)
                            gw = torch.mul(gw, val1)
                        if "2" in args.gw_type:
                            cntmp = torch.sign(attgrad)
                            cntmp = cntmp.view(attgrad.size()[0], attgrad.size()[1], 224 * 224)
                            cntmp0 = torch.clamp(cntmp, min=-1, max=0)
                            cnt0 = torch.sum(torch.abs(cntmp0), dim=2)
                            cnt0 = cnt0.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                            cntmp1 = torch.clamp(cntmp, min=0, max=1)
                            cnt1 = torch.sum(cntmp1, dim=2)
                            cnt1 = cnt1.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                            val2 = torch.maximum(cnt0, cnt1) / torch.minimum(cnt0, cnt1)
                            gw = torch.mul(gw, val2)

                        gw = torch.clamp(gw, max=1)
                        gw = gw * 0.8
                        motionpertu = motionpertu.view(attgrad.size()[0], attgrad.size()[1], 224, 224)
                        attgrad = attgrad.view(attgrad.size()[0], attgrad.size()[1], 224, 224)

                        # if i % 100 == 0:
                        # print(gw)
                        gw = torch.maximum(gw, temp)
                        pertu = torch.mul(motionpertu, gw)

                        output = model(input + pertu)
                        loss = criterion(output, target).cuda()
                        loss.backward()
                        if i % 50 == 0:
                            print(loss)
                        fg = frameval.grad.data

                        model.zero_grad()
                    print("one time: ", time.time()-st)
                    if i % 50 == 0:
                        print(loss)
                        print("===========")

                elif args.attack_type == "2":
                    with torch.no_grad():
                        temp = torch.mul(torch.sign(attgrad), motionpertu)
                        temp = torch.clamp(input=temp, min=0)
                        temp = torch.where(temp > 0, 1.2, 0)

                        attgrad = attgrad.view(attgrad.size()[0], attgrad.size()[1], 224 * 224)
                        va = torch.max(input=attgrad, dim=2)[0]
                        vi = torch.min(input=attgrad, dim=2)[0]
                        attgrad = attgrad.view(attgrad.size()[0], attgrad.size()[1], 224, 224)
                        va = va.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                        vi = vi.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                        # normgrad = torch.maximum(torch.abs(va), torch.abs(vi))
                        normgrad = (torch.abs(va) + torch.abs(vi)) / 2
                        gw = torch.div(attgrad, normgrad)

                        if "3" in args.gw_type:
                            gw = gw.view(attgrad.size()[0], attgrad.size()[1], 224 * 224)
                            ave = torch.mean(gw, dim=2)
                            va = torch.max(gw, dim=2)[0]
                            vi = torch.min(gw, dim=2)[0]
                            ave = ave.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                            va = va.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                            vi = vi.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                            gw = gw.view(attgrad.size()[0], attgrad.size()[1], 224, 224)
                            val3 = 1 + torch.div(torch.abs((va + vi) / 2 - ave), va - vi)
                            gw = torch.mul(gw, val3)

                        gw = torch.clamp(gw, min=-1, max=1)
                        gw = 1 - torch.abs(gw)

                        attgrad = attgrad.view(attgrad.size()[0], attgrad.size()[1], 224 * 224)
                        motionpertu = motionpertu.view(attgrad.size()[0], attgrad.size()[1], 224 * 224)
                        if "1" in args.gw_type:
                            cntmp = torch.sign(motionpertu)
                            cntmp = cntmp.view(attgrad.size()[0], attgrad.size()[1], 224 * 224)
                            cntmp0 = torch.clamp(cntmp, min=-1, max=0)
                            cnt0 = torch.sum(torch.abs(cntmp0), dim=2)
                            cnt0 = cnt0.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                            cntmp1 = torch.clamp(cntmp, min=0, max=1)
                            cnt1 = torch.sum(cntmp1, dim=2)
                            cnt1 = cnt1.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                            val1 = torch.minimum(cnt0, cnt1) / torch.maximum(cnt0, cnt1)
                            gw = torch.mul(gw, val1)
                        if "2" in args.gw_type:
                            cntmp = torch.sign(attgrad)
                            cntmp = cntmp.view(attgrad.size()[0], attgrad.size()[1], 224 * 224)
                            cntmp0 = torch.clamp(cntmp, min=-1, max=0)
                            cnt0 = torch.sum(torch.abs(cntmp0), dim=2)
                            cnt0 = cnt0.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                            cntmp1 = torch.clamp(cntmp, min=0, max=1)
                            cnt1 = torch.sum(cntmp1, dim=2)
                            cnt1 = cnt1.view(attgrad.size()[0], attgrad.size()[1], 1, 1)
                            val2 = torch.maximum(cnt0, cnt1) / torch.minimum(cnt0, cnt1)
                            gw = torch.mul(gw, val2)

                        gw = torch.clamp(gw, max=1)
                        gw = gw * 0.8
                        motionpertu = motionpertu.view(attgrad.size()[0], attgrad.size()[1], 224, 224)
                        attgrad = attgrad.view(attgrad.size()[0], attgrad.size()[1], 224, 224)

                        # if i % 100 == 0:
                        # print(gw)
                        gw = torch.maximum(gw, temp)
                        pertu = torch.mul(motionpertu, gw)

            elif args.attack_type == "3":
                model.zero_grad()
                adv_input = (input + motionpertu * args.step_start).clone().detach()
                adv_input.requires_grad = True
                output = model(adv_input)
                loss = criterion(output, target).cuda()
                loss.backward()
                attgrad = adv_input.grad.data * 1e6
                attgrad.requires_grad = False
                model.zero_grad()
                # 先得到起步对应梯度

                for step in range(args.step_times):
                    temp = torch.mul(torch.sign(attgrad), motionpertu)
                    temp = torch.clamp(input=temp, min=0)
                    pertu = torch.where(temp > 0, motionpertu * args.step_weight, temp)

                    adv_input = (adv_input + pertu).detach()
                    adv_input = torch.clamp(adv_input, min=zeroinput, max=oneinput)
                    model.zero_grad()
                    adv_input.requires_grad = True
                    output = model(adv_input)
                    loss = criterion(output, target).cuda()
                    loss.backward()
                    attgrad = adv_input.grad.data * 1e6
                    adv_input.grad.data.zero_()
                    model.zero_grad()

                    if i % 50 == 0:
                        print(loss)

                pertu = adv_input - input
            elif "4" in args.attack_type:
                st = time.time()
                frameval = torch.ones([input.size()[0], args.motion_len]).cuda()
                best_loss = 0
                best_adv = input.clone().detach()
                best_adv.requires_grad = False
                for e in range(args.frame_times):
                    model.zero_grad()
                    frameval = frameval.clone().detach()
                    frameval.requires_grad = True
                    if args.attack_type == "4":
                        stdval = torch.div(frameval * args.motion_val, torch.sum(frameval, dim=1).view(-1, 1))
                        motioninput = input.clone() * (1 - args.motion_val)
                        input_roll = addition_input.clone()
                        for step in range(1, args.motion_len + 1):
                            input_roll = torch.roll(input_roll, 3, 1)
                            motioninput = motioninput + torch.index_select(input_roll, dim=1, index=indices) * stdval[:,
                                                                                                               args.motion_len - step].view(
                                input.size()[0], 1, 1, 1)  # dim=1 frames*3的维度 移动3位（3通道）
                        motionpertu = motioninput - input.clone()
                        adv_input = input + motionpertu
                        output = model(adv_input)
                        loss = criterion(output, target).cuda()
                        loss.backward()

                        frameval = frameval + torch.sign(frameval.grad.data) * args.frame_weight
                        frameval = torch.clamp(frameval, min=0)

                        model.zero_grad()
                        if i % 50 == 0:
                            print(loss)

                    elif "3" in args.attack_type:
                        model.zero_grad()
                        adv_input = (input + motionpertu * args.step_start).clone().detach()
                        adv_input.requires_grad = True
                        output = model(adv_input)
                        loss = criterion(output, target).cuda()
                        loss.backward()
                        attgrad = adv_input.grad.data * 1e6
                        attgrad.requires_grad = False
                        model.zero_grad()
                        # 先得到起步对应梯度

                        framevalgrad = torch.zeros(frameval.size())

                        for it in range(args.step_times):
                            frameval = frameval.clone().detach()  # 阻断 frameval是叶子
                            frameval.requires_grad = True
                            stdval = torch.div(frameval * args.motion_val, torch.sum(frameval, dim=1).view(-1, 1))
                            motioninput = input.clone() * (1 - args.motion_val)
                            input_roll = addition_input.clone().detach()
                            for step in range(1, args.motion_len + 1):
                                input_roll = torch.roll(input_roll, 3, 1)
                                motioninput = motioninput + torch.index_select(input_roll, dim=1,
                                                                               index=indices) * stdval[:,
                                                                                                args.motion_len - step].view(
                                    input.size()[0], 1, 1, 1)
                            motionpertu = motioninput - input.clone().detach()

                            temp = torch.mul(torch.sign(attgrad), motionpertu)
                            temp = torch.clamp(input=temp, min=0)
                            pertu = torch.where(temp > 0, motionpertu * args.step_weight, 0)

                            adv_input = adv_input.clone().detach() + pertu
                            adv_input = torch.clamp(adv_input, min=zeroinput, max=oneinput)
                            model.zero_grad()
                            output = model(adv_input)
                            adv_input.retain_grad()

                            loss = criterion(output, target).cuda()
                            loss.backward()
                            if i % 50 == 0:
                                print(loss)

                            attgrad = adv_input.grad.data * 1e6
                            framevalgrad = torch.sign(frameval.grad.data) * args.frame_weight

                            frameval.grad.zero_()
                            adv_input.grad.zero_()
                            model.zero_grad()
                        if i % 50 == 0:
                            print(e, end="========\n")
                        if loss > best_loss:
                            best_adv = adv_input
                            best_loss = loss
                            if i % 50 == 0:
                                print(loss, end=" best-----------------\n")
                        frameval = frameval + framevalgrad / args.step_times
                        frameval = torch.clamp(frameval, min=0)
                        model.zero_grad()
                        if i % 50 == 0:
                            print(loss)
                            print("===========")
                pertu = best_adv - input
                print("one time: ", time.time()-st)

            with torch.no_grad():
                output = model(input + pertu)
                loss = criterion(output, target).cuda()
                prec1_att, prec5_att = accuracy(output.data, target, topk=(1, 5))
                top1_att.update(prec1_att.item(), input.size(0))
                top5_att.update(prec5_att.item(), input.size(0))
                losses_att.update(loss.item(), input.size(0))

                if i % args.print_freq == 0:
                    logger.info(
                        ('Attack Test: [{0}/{1}]\t'
                         'Prec_att@1 {top1_att.val:.3f} ({top1_att.avg:.3f})\t'
                         'Prec_att@5 {top5_att.val:.3f} ({top5_att.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                            i, len(val_loader), top1_att=top1_att, top5_att=top5_att, loss=losses_att)))

                saveImg = os.path.join(args.root_model, args.store_name, str(epoch))
                if not os.path.exists(saveImg):
                    os.makedirs(saveImg)
                if not args.full_saveimg:
                    img_input = back_normalize(input.clone().detach().view(-1, 3, 224, 224), model.input_mean,
                                               model.input_std).cuda()
                    img_adv = back_normalize((input.clone().detach() + pertu.clone().detach()).view(-1, 3, 224, 224),
                                             model.input_mean, model.input_std).cuda()
                    img_pertu = torch.abs(img_adv - img_input)
                    img_moinput = back_normalize(motioninput.clone().detach().view(-1, 3, 224, 224),
                                                 model.input_mean, model.input_std).cuda()

                    img_input = img_input.to(torch.device('cpu'))
                    img_adv = img_adv.to(torch.device('cpu'))
                    img_pertu = img_pertu.to(torch.device('cpu'))
                    img_moinput = img_moinput.to(torch.device('cpu'))

                    torchvision.utils.save_image(img_input, saveImg + "/input" + str(i) + ".jpg")
                    torchvision.utils.save_image(img_adv, saveImg + "/adv" + str(i) + ".jpg")
                    torchvision.utils.save_image(img_pertu, saveImg + "/pertu" + str(i) + ".jpg")
                    torchvision.utils.save_image(img_moinput, saveImg + "/moinput" + str(i) + ".jpg")
                else:
                    img_adv = back_normalize((input.clone().detach() + pertu.clone().detach()).view(-1, 3, 224, 224),
                                             model.input_mean, model.input_std).cuda()
                    img_moinput = back_normalize(motioninput.clone().detach().view(-1, 3, 224, 224),
                                                 model.input_mean, model.input_std).cuda()
                    img_input = back_normalize(input.clone().detach().view(-1, 3, 224, 224), model.input_mean,
                                               model.input_std).cuda()

                    img_adv = img_adv.to(torch.device('cpu')).view(-1, 3, 224, 224)
                    img_moinput = img_moinput.to(torch.device('cpu')).view(-1, 3, 224, 224)
                    img_input = img_input.to(torch.device('cpu')).view(-1, 3, 224, 224)

                    frame_tot = img_adv.size(0) // args.batch_size

                    if not args.full_savepath == "":
                        saveImg = args.full_savepath
                    else:
                        saveImg = "/media/hd0/wuguoming/att-mb/" + args.dataset + "/" + str(
                            args.attack_type) + "=m" + str(args.motion_len) + "=v" + str(args.motion_val).replace(".",
                                                                                                                  "")[
                                                                                     :2]
                        if "4" in args.attack_type:
                            saveImg = saveImg + "=" + str(args.frame_times) + "-" + str(args.frame_weight).replace(".",
                                                                                                                   "")[
                                                                                    :2]
                        if "3" in args.attack_type:
                            saveImg = saveImg + "=" + str(args.step_start).replace(".", "")[:2] + "-" + str(
                                args.step_start + args.step_times * args.step_weight).replace(".", "")[:2]
                        if "2" in args.attack_type:
                            saveImg = saveImg + "=" + args.gw_type
                        saveImg = saveImg + "/"
                    for bs in range(args.batch_size):
                        imgpath = saveImg + str(i * args.batch_size + bs) + "/"
                        # motionpath = "/media/hd0/wuguoming/att-mb/" + args.dataset + "/normal_motion_m" + str(args.motion_len) + "=v" + str(args.motion_val).replace(".", "")[:2] + "/" + str(i * args.batch_size + bs) + "/"
                        if not os.path.exists(imgpath):
                            os.makedirs(imgpath)
                        # if not os.path.exists(motionpath):
                        # os.makedirs(motionpath)
                        for frame_idx in range(frame_tot):
                            torchvision.utils.save_image(img_adv[bs * frame_tot + frame_idx],
                                                         imgpath + str(frame_idx) + ".jpg")
                            # torchvision.utils.save_image(img_moinput[bs * frame_tot + frame_idx], motionpath + str(frame_idx) + ".jpg")

    logger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                 .format(top1=top1, top5=top5, loss=losses)))
    if not args.attack_type == "0":
        logger.info(('Attacking Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                     .format(top1=top1_att, top5=top5_att, loss=losses_att)))

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, epoch, is_best):
    filename = '%s/%s/%d_epoch_ckpt.pth.tar' % (args.root_model, args.store_name, epoch)
    torch.save(state, filename)
    if is_best:
        best_filename = '%s/%s/best.pth.tar' % (args.root_model, args.store_name)
        torch.save(state, best_filename)


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [
        args.root_log, args.root_model,
        os.path.join(args.root_log, args.store_name),
        os.path.join(args.root_model, args.store_name)
    ]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder)


if __name__ == '__main__':
    main()

'''
CUDA_VISIBLE_DEVICES=0 python main.py hmdb51 RGB --arch resnet101 \
--num_segments 8 --dropout 0.5 --consensus_type=avg --eval-freq=1 --npb \
--evaluate --batch-size 8 \
--tune_from /media/hd0/wuguoming/wgm_data/best-hmdb51/kin-101.tar  \
--root_model /media/hd0/wuguoming/wgm_data \
--attack_type 34 --frame_times 20 --frame_weight 0.2 \
--step_start 0.3 --step_times 5 --step_weight 0.28 \
--motion_len 10 > ./hm-att/hm-att-34=20-02=03-17.txt 2>&1
'''

'''
CUDA_VISIBLE_DEVICES=0 python main.py ucf101 RGB --arch resnet101 --num_segments 8 --dropout 0.5 --consensus_type=avg --eval-freq=1 --npb --evaluate --batch-size 8 --tune_from /media/hd0/wuguoming/wgm_data/best-ucf101/kin-101.tar  --root_model /media/hd0/wuguoming/wgm_data --attack_type 3 --step_start 0.3 --step_times 5 --step_weight 0.28 --motion_len 5 > ./ucf-att/ucf-att-3=m5=03-17.txt 2>&1

'''
