# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

l = [i for i in range(int(1e8))]
l = []
import pdb
pdb.set_trace()

l = []

import gc
gc.collect()
# 0

gc.get_referrers(l)
# [{'__builtins__': <module '__builtin__' (built-in)>, 'l': [], '__package__': None, 'i': 99999999, 'gc': <module 'gc' (built-in)>, '__name__': '__main__', '__doc__': None}]

del l
gc.collect()
# 0
