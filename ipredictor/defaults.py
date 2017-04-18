#: -*- coding: utf-8 -*-
"""
Application default settings file
"""
#: package name
PACKAGE_NAME = 'ipredictor'

#: data datetime format
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

#: resample period identifier
RESAMPLE_PERIOD = 'H'

#: default season period is 24 hours for hourly resampled data
SEASON_PERIOD = 24

#: start coefs for optimization routines
INITIAL_COEF = 0.1

#: default ANN train epochs
TRAIN_EPOCHS = 1