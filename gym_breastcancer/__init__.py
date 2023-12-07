#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shossain
"""
from gym.envs.registration import register


register(
    id = 'breastcancer-v1',
    entry_point = 'gym_breastcancer.envs:BreastCancerDCIS_Ray',
    kwargs = {}
    )

register(
    id = 'breastcancer-v2',
    entry_point = 'gym_breastcancer.envs:BreastCancerDCIS',
    kwargs = {}
    )

register(
    id = 'breastcancer-v3',
    entry_point = 'gym_breastcancer.envs:BreastCancerDCIS_Ray_2_5',
    kwargs = {}
    )


