# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Any, Dict, List

import torch.nn as nn

from torch import Tensor

__all__ = [
    "LatentCodec",
]


class _SetDefaultMixin:
    """Convenience functions for initializing classes with defaults."""

    def _setdefault(self, k, v, f):
        """Initialize attribute ``k`` with value ``v`` or ``f()``."""
        v = v or f()
        setattr(self, k, v)

    # TODO instead of save_direct, override load_state_dict() and state_dict()
    def _set_group_defaults(self, group_key, group_dict, defaults, save_direct=False):
        """Initialize attribute ``group_key`` with items from
        ``group_dict``, using defaults for missing keys.
        Ensures ``nn.Module`` attributes are properly registered.

        Args:
            - group_key:
                Name of attribute.
            - group_dict:
                Dict of items to initialize ``group_key`` with.
            - defaults:
                Dict of defaults for items not in ``group_dict``.
            - save_direct:
                If ``True``, save items directly as attributes of ``self``.
                If ``False``, save items in a ``nn.ModuleDict``.
        """
        group_dict = group_dict if group_dict is not None else {}
        for k, f in defaults.items():
            if k in group_dict:
                continue
            group_dict[k] = f()
        if save_direct:
            for k, v in group_dict.items():
                setattr(self, k, v)
        else:
            group_dict = nn.ModuleDict(group_dict)
        setattr(self, group_key, group_dict)


class LatentCodec(nn.Module, _SetDefaultMixin):
    def forward(self, y: Tensor, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def compress(self, y: Tensor, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def decompress(
        self, strings: List[List[bytes]], shape: Any, *args, **kwargs
    ) -> Dict[str, Any]:
        raise NotImplementedError
