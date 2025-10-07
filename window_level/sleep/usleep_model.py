# Authors: Theo Gnassounou <theo.gnassounou@inria.fr>
#          Omar Chehab <l-emir-omar.chehab@inria.fr>
#
# License: BSD (3-clause)

import numpy as np
import torch
from torch import nn
# from .base import EEGModuleMixin, deprecated_args


# Authors: Pierre Guetschel
#          Maciej Sliwowski
#
# License: BSD-3

import warnings
from typing import Dict, Iterable, List, Optional, Tuple

from collections import OrderedDict

import numpy as np
import torch
from docstring_inheritance import NumpyDocstringInheritanceInitMeta
from torchinfo import ModelStatistics, summary


def deprecated_args(obj, *old_new_args):
    out_args = []
    for old_name, new_name, old_val, new_val in old_new_args:
        if old_val is None:
            out_args.append(new_val)
        else:
            warnings.warn(
                f'{obj.__class__.__name__}: {old_name!r} is depreciated. Use {new_name!r} instead.'
            )
            if new_val is not None:
                raise ValueError(
                    f'{obj.__class__.__name__}: Both {old_name!r} and {new_name!r} were specified.'
                )
            out_args.append(old_val)
    return out_args


class EEGModuleMixin(metaclass=NumpyDocstringInheritanceInitMeta):
    """
    Mixin class for all EEG models in braindecode.

    Parameters
    ----------
    n_outputs : int
        Number of outputs of the model. This is the number of classes
        in the case of classification.
    n_chans : int
        Number of EEG channels.
    chs_info : list of dict
        Information about each individual EEG channel. This should be filled with
        ``info["chs"]``. Refer to :class:`mne.Info` for more details.
    n_times : int
        Number of time samples of the input window.
    input_window_seconds : float
        Length of the input window in seconds.
    sfreq : float
        Sampling frequency of the EEG recordings.
    add_log_softmax: bool
        Whether to use log-softmax non-linearity as the output function.
        LogSoftmax final layer will be removed in the future.
        Please adjust your loss function accordingly (e.g. CrossEntropyLoss)!
        Check the documentation of the torch.nn loss functions:
        https://pytorch.org/docs/stable/nn.html#loss-functions.

    Raises
    ------
    ValueError: If some input signal-related parameters are not specified
                and can not be inferred.

    FutureWarning: If add_log_softmax is True, since LogSoftmax final layer
                   will be removed in the future.

    Notes
    -----
    If some input signal-related parameters are not specified,
    there will be an attempt to infer them from the other parameters.
    """

    def __init__(
            self,
            n_outputs: Optional[int] = None,
            n_chans: Optional[int] = None,
            chs_info: Optional[List[Dict]] = None,
            n_times: Optional[int] = None,
            input_window_seconds: Optional[float] = None,
            sfreq: Optional[float] = None,
            add_log_softmax: Optional[bool] = False,
    ):
        if (
                n_chans is not None and
                chs_info is not None and
                len(chs_info) != n_chans
        ):
            raise ValueError(f'{n_chans=} different from {chs_info=} length')
        if (
                n_times is not None and
                input_window_seconds is not None and
                sfreq is not None and
                n_times != int(input_window_seconds * sfreq)
        ):
            raise ValueError(
                f'{n_times=} different from '
                f'{input_window_seconds=} * {sfreq=}'
            )
        self._n_outputs = n_outputs
        self._n_chans = n_chans
        self._chs_info = chs_info
        self._n_times = n_times
        self._input_window_seconds = input_window_seconds
        self._sfreq = sfreq
        self._add_log_softmax = add_log_softmax
        super().__init__()

    @property
    def n_outputs(self):
        if self._n_outputs is None:
            raise ValueError('n_outputs not specified.')
        return self._n_outputs

    @property
    def n_chans(self):
        if self._n_chans is None and self._chs_info is not None:
            return len(self._chs_info)
        elif self._n_chans is None:
            raise ValueError(
                'n_chans could not be inferred. Either specify n_chans or chs_info.'
            )
        return self._n_chans

    @property
    def chs_info(self):
        if self._chs_info is None:
            raise ValueError('chs_info not specified.')
        return self._chs_info

    @property
    def n_times(self):
        if (
                self._n_times is None and
                self._input_window_seconds is not None and
                self._sfreq is not None
        ):
            return int(self._input_window_seconds * self._sfreq)
        elif self._n_times is None:
            raise ValueError(
                'n_times could not be inferred. '
                'Either specify n_times or input_window_seconds and sfreq.'
            )
        return self._n_times

    @property
    def input_window_seconds(self):
        if (
                self._input_window_seconds is None and
                self._n_times is not None and
                self._sfreq is not None
        ):
            return self._n_times / self._sfreq
        elif self._input_window_seconds is None:
            raise ValueError(
                'input_window_seconds could not be inferred. '
                'Either specify input_window_seconds or n_times and sfreq.'
            )
        return self._input_window_seconds

    @property
    def sfreq(self):
        if (
                self._sfreq is None and
                self._input_window_seconds is not None and
                self._n_times is not None
        ):
            return self._n_times / self._input_window_seconds
        elif self._sfreq is None:
            raise ValueError(
                'sfreq could not be inferred. '
                'Either specify sfreq or input_window_seconds and n_times.'
            )
        return self._sfreq

    @property
    def add_log_softmax(self):
        if self._add_log_softmax:
            warnings.warn("LogSoftmax final layer will be removed! " +
                          "Please adjust your loss function accordingly (e.g. CrossEntropyLoss)!")
        return self._add_log_softmax

    @property
    def input_shape(self) -> Tuple[int]:
        """Input data shape."""
        return (1, self.n_chans, self.n_times)

    def get_output_shape(self) -> Tuple[int]:
        """Returns shape of neural network output for batch size equal 1.

        Returns
        -------
        output_shape: Tuple[int]
            shape of the network output for `batch_size==1` (1, ...)
    """
        with torch.inference_mode():
            try:
                return tuple(self.forward(
                    torch.zeros(
                        self.input_shape,
                        dtype=next(self.parameters()).dtype,
                        device=next(self.parameters()).device
                    )).shape)
            except RuntimeError as exc:
                if str(exc).endswith(
                        ("Output size is too small",
                         "Kernel size can't be greater than actual input size")
                ):
                    msg = (
                        "During model prediction RuntimeError was thrown showing that at some "
                        f"layer `{str(exc).split('.')[-1]}` (see above in the stacktrace). This "
                        "could be caused by providing too small `n_times`/`input_window_seconds`. "
                        "Model may require longer chunks of signal in the input than "
                        f"{self.input_shape}."
                    )
                    raise ValueError(msg) from exc
                raise exc

    mapping = None

    def load_state_dict(self, state_dict, *args, **kwargs):

        mapping = self.mapping if self.mapping else {}
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k in mapping:
                new_state_dict[mapping[k]] = v
            else:
                new_state_dict[k] = v

        return super().load_state_dict(new_state_dict, *args, **kwargs)

    def to_dense_prediction_model(self, axis: Tuple[int] = (2, 3)) -> None:
        """
        Transform a sequential model with strides to a model that outputs
        dense predictions by removing the strides and instead inserting dilations.
        Modifies model in-place.

        Parameters
        ----------
        axis: int or (int,int)
            Axis to transform (in terms of intermediate output axes)
            can either be 2, 3, or (2,3).

        Notes
        -----
        Does not yet work correctly for average pooling.
        Prior to version 0.1.7, there had been a bug that could move strides
        backwards one layer.

        """
        if not hasattr(axis, "__len__"):
            axis = [axis]
        assert all([ax in [2, 3] for ax in axis]), "Only 2 and 3 allowed for axis"
        axis = np.array(axis) - 2
        stride_so_far = np.array([1, 1])
        for module in self.modules():
            if hasattr(module, "dilation"):
                assert module.dilation == 1 or (module.dilation == (1, 1)), (
                    "Dilation should equal 1 before conversion, maybe the model is "
                    "already converted?"
                )
                new_dilation = [1, 1]
                for ax in axis:
                    new_dilation[ax] = int(stride_so_far[ax])
                module.dilation = tuple(new_dilation)
            if hasattr(module, "stride"):
                if not hasattr(module.stride, "__len__"):
                    module.stride = (module.stride, module.stride)
                stride_so_far *= np.array(module.stride)
                new_stride = list(module.stride)
                for ax in axis:
                    new_stride[ax] = 1
                module.stride = tuple(new_stride)

    def get_torchinfo_statistics(
            self,
            col_names: Optional[Iterable[str]] = (
                    "input_size",
                    "output_size",
                    "num_params",
                    "kernel_size",
            ),
            row_settings: Optional[Iterable[str]] = ("var_names", "depth"),
    ) -> ModelStatistics:
        """Generate table describing the model using torchinfo.summary.

        Parameters
        ----------
        col_names : tuple, optional
            Specify which columns to show in the output, see torchinfo for details, by default
            ("input_size", "output_size", "num_params", "kernel_size")
        row_settings : tuple, optional
             Specify which features to show in a row, see torchinfo for details, by default
             ("var_names", "depth")

        Returns
        -------
        torchinfo.ModelStatistics
            ModelStatistics generated by torchinfo.summary.
        """
        return summary(
            self,
            input_size=(1, self.n_chans, self.n_times),
            col_names=col_names,
            row_settings=row_settings,
            verbose=0,
        )

    def __str__(self) -> str:
        return str(self.get_torchinfo_statistics())



def _crop_tensors_to_match(x1, x2, axis=-1):
    """Crops two tensors to their lowest-common-dimension along an axis."""
    dim_cropped = min(x1.shape[axis], x2.shape[axis])

    x1_cropped = torch.index_select(
        x1, dim=axis,
        index=torch.arange(dim_cropped).to(device=x1.device)
    )
    x2_cropped = torch.index_select(
        x2, dim=axis,
        index=torch.arange(dim_cropped).to(device=x1.device)
    )
    return x1_cropped, x2_cropped


class _EncoderBlock(nn.Module):
    """Encoding block for a timeseries x of shape (B, C, T)."""

    def __init__(self,
                 in_channels=2,
                 out_channels=2,
                 kernel_size=9,
                 downsample=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.downsample = downsample

        self.block_prepool = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding='same'),
            nn.ELU(),
            nn.BatchNorm1d(num_features=out_channels),
        )

        self.pad = nn.ConstantPad1d(padding=1, value=0)
        self.maxpool = nn.MaxPool1d(
            kernel_size=self.downsample, stride=self.downsample)

    def forward(self, x):
        x = self.block_prepool(x)
        residual = x
        if x.shape[-1] % 2:
            x = self.pad(x)
        x = self.maxpool(x)
        return x, residual


class _DecoderBlock(nn.Module):
    """Decoding block for a timeseries x of shape (B, C, T)."""

    def __init__(self,
                 in_channels=2,
                 out_channels=2,
                 kernel_size=9,
                 upsample=2,
                 with_skip_connection=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.with_skip_connection = with_skip_connection

        self.block_preskip = nn.Sequential(
            nn.Upsample(scale_factor=upsample),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=2,
                      padding='same'),
            nn.ELU(),
            nn.BatchNorm1d(num_features=out_channels),
        )
        self.block_postskip = nn.Sequential(
            nn.Conv1d(
                in_channels=(
                    2 * out_channels if with_skip_connection else out_channels),
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding='same'),
            nn.ELU(),
            nn.BatchNorm1d(num_features=out_channels),
        )

    def forward(self, x, residual):
        x = self.block_preskip(x)
        if self.with_skip_connection:
            x, residual = _crop_tensors_to_match(x, residual, axis=-1)  # in case of mismatch
            x = torch.cat([x, residual], axis=1)  # (B, 2 * C, T)
        x = self.block_postskip(x)
        return x


class USleep(EEGModuleMixin, nn.Module):
    """Sleep staging architecture from Perslev et al 2021.

    U-Net (autoencoder with skip connections) feature-extractor for sleep
    staging described in [1]_.

    For the encoder ('down'):
        -- the temporal dimension shrinks (via maxpooling in the time-domain)
        -- the spatial dimension expands (via more conv1d filters in the
           time-domain)
    For the decoder ('up'):
        -- the temporal dimension expands (via upsampling in the time-domain)
        -- the spatial dimension shrinks (via fewer conv1d filters in the
           time-domain)
    Both do so at exponential rates.

    Parameters
    ----------
    n_chans : int
        Number of EEG or EOG channels. Set to 2 in [1]_ (1 EEG, 1 EOG).
    sfreq : float
        EEG sampling frequency. Set to 128 in [1]_.
    depth : int
        Number of conv blocks in encoding layer (number of 2x2 max pools)
        Note: each block halve the spatial dimensions of the features.
    n_time_filters : int
        Initial number of convolutional filters. Set to 5 in [1]_.
    complexity_factor : float
        Multiplicative factor for number of channels at each layer of the U-Net.
        Set to 2 in [1]_.
    with_skip_connection : bool
        If True, use skip connections in decoder blocks.
    n_outputs : int
        Number of outputs/classes. Set to 5.
    input_window_seconds : float
        Size of the input, in seconds. Set to 30 in [1]_.
    time_conv_size_s : float
        Size of the temporal convolution kernel, in seconds. Set to 9 / 128 in
        [1]_.
    ensure_odd_conv_size : bool
        If True and the size of the convolutional kernel is an even number, one
        will be added to it to ensure it is odd, so that the decoder blocks can
        work. This can ne useful when using different sampling rates from 128
        or 100 Hz.
    in_chans : int
        Alias for n_chans.
    n_classes : int
        Alias for n_outputs.
    input_size_s : float
        Alias for input_window_seconds.

    References
    ----------
    .. [1] Perslev M, Darkner S, Kempfner L, Nikolic M, Jennum PJ, Igel C.
           U-Sleep: resilient high-frequency sleep staging. npj Digit. Med. 4, 72 (2021).
           https://github.com/perslev/U-Time/blob/master/utime/models/usleep.py
    """

    def __init__(
            self,
            n_chans=2,
            sfreq=128,
            depth=12,
            n_time_filters=5,
            complexity_factor=1.67,
            with_skip_connection=True,
            n_outputs=5,
            input_window_seconds=30,
            time_conv_size_s=9 / 128,
            ensure_odd_conv_size=False,
            chs_info=None,
            n_times=None,
            in_chans=None,
            n_classes=None,
            input_size_s=None,
            add_log_softmax=False,
    ):
        n_chans, n_outputs, input_window_seconds = deprecated_args(
            self,
            ("in_chans", "n_chans", in_chans, n_chans),
            ("n_classes", "n_outputs", n_classes, n_outputs),
            ("input_size_s", "input_window_seconds", input_size_s, input_window_seconds),
        )
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            add_log_softmax=add_log_softmax,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del in_chans, n_classes, input_size_s

        self.mapping = {
            'clf.3.weight': 'final_layer.0.weight',
            'clf.3.bias': 'final_layer.0.bias',
            'clf.5.weight': 'final_layer.2.weight',
            'clf.5.bias': 'final_layer.2.bias'
        }

        max_pool_size = 2  # Hardcoded to avoid dimensional errors
        time_conv_size = int(np.round(time_conv_size_s * self.sfreq))
        if time_conv_size % 2 == 0:
            if ensure_odd_conv_size:
                time_conv_size += 1
            else:
                raise ValueError(
                    'time_conv_size must be an odd number to accommodate the '
                    'upsampling step in the decoder blocks.')

        channels = [self.n_chans]
        n_filters = n_time_filters
        for _ in range(depth + 1):
            channels.append(int(n_filters * np.sqrt(complexity_factor)))
            n_filters = int(n_filters * np.sqrt(2))
        self.channels = channels

        # Instantiate encoder
        encoder = list()
        for idx in range(depth):
            encoder += [
                _EncoderBlock(in_channels=channels[idx],
                              out_channels=channels[idx + 1],
                              kernel_size=time_conv_size,
                              downsample=max_pool_size)
            ]
        self.encoder = nn.Sequential(*encoder)

        # Instantiate bottom (channels increase, temporal dim stays the same)
        self.bottom = nn.Sequential(
            nn.Conv1d(in_channels=channels[-2],
                      out_channels=channels[-1],
                      kernel_size=time_conv_size,
                      padding=(time_conv_size - 1) // 2),  # preserves dimension
            nn.ELU(),
            nn.BatchNorm1d(num_features=channels[-1]),
        )

        # Instantiate decoder
        decoder = list()
        channels_reverse = channels[::-1]
        for idx in range(depth):
            decoder += [
                _DecoderBlock(in_channels=channels_reverse[idx],
                              out_channels=channels_reverse[idx + 1],
                              kernel_size=time_conv_size,
                              upsample=max_pool_size,
                              with_skip_connection=with_skip_connection)
            ]
        self.decoder = nn.Sequential(*decoder)

        # The temporal dimension remains unchanged
        # (except through the AvgPooling which collapses it to 1)
        # The spatial dimension is preserved from the end of the UNet, and is mapped to n_classes

        self.clf = nn.Sequential(
            nn.Conv1d(
                in_channels=channels[1],
                out_channels=channels[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),  # output is (B, C, 1, S * T)
            nn.Tanh(),
            nn.AvgPool1d(self.n_times),  # output is (B, C, S)
        )

        self.final_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=channels[1],
                out_channels=self.n_outputs,
                kernel_size=1,
                stride=1,
                padding=0,
            ),  # output is (B, n_classes, S)
            nn.ELU(),
            nn.Conv1d(
                in_channels=self.n_outputs,
                out_channels=self.n_outputs,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.LogSoftmax(dim=1) if self.add_log_softmax else nn.Identity(),
            # output is (B, n_classes, S)
        )

    def forward(self, x):
        """If input x has shape (B, S, C, T), return y_pred of shape (B, n_classes, S).
        If input x has shape (B, C, T), return y_pred of shape (B, n_classes).
        """
        # print('x', x.shape)
        # reshape input
        if x.ndim == 4:  # input x has shape (B, S, C, T)
            x = x.permute(0, 2, 1, 3)  # (B, C, S, T)
            x = x.flatten(start_dim=2)  # (B, C, S * T)

        # print('x', x.shape)
        # encoder
        residuals = []
        for down in self.encoder:
            x, res = down(x)
            residuals.append(res)

        # bottom
        x = self.bottom(x)

        # decoder
        residuals = residuals[::-1]  # flip order
        for up, res in zip(self.decoder, residuals):
            x = up(x, res)

        # classifier
        # print('x before clf', x.shape)
        x = self.clf(x)
        # print('x after clf', x.shape)
        y_pred = self.final_layer(x)  # (B, n_classes, seq_length)
        # print('y_pred before', y_pred.shape)

        if y_pred.shape[-1] == 1:  # seq_length of 1
            y_pred = y_pred[:, :, 0]

        # print('y_pred', y_pred.shape)
        return y_pred
