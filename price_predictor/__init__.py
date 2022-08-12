"""Use a customizable LSTM recurrent neural network to quickly predict the future prices of financial instruments."""


__version__ = '1.0.4'

from .price_predictor import Price_Predictor
from .price_predictor import yahoo_finance_csv
from .price_predictor import quick_tomorrow
from .price_predictor import Predict_Iterator
