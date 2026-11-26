# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Opt-in, per-step GPU timing that is cheap enough to leave on.

Enable with ``--perf-trace-dir``; see :mod:`megatron.core.perf_trace.tracer` for the cost model
and :mod:`megatron.core.perf_trace.hooks` for what gets instrumented.
"""

from .hooks import (  # noqa: F401
    TIMERS_OF_INTEREST,
    begin_train_step,
    capture_timers,
    end_train_step,
    get_tracer,
    maybe_install,
    phase,
    wrap_data_iterator,
)
from .tracer import PerfTracer, TimedIterator  # noqa: F401

__all__ = [
    'PerfTracer',
    'TimedIterator',
    'TIMERS_OF_INTEREST',
    'begin_train_step',
    'capture_timers',
    'phase',
    'end_train_step',
    'get_tracer',
    'maybe_install',
    'wrap_data_iterator',
]
