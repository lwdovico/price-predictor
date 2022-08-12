#!/usr/bin/env python3
import re
import sys
import os
from distutils.core import setup

if sys.version_info < (3, 5):
    sys.exit('Price_predictor requires Python >= 3.5.')
    
abs_path = os.path.abspath(os.path.dirname(__file__))

def get_version():
    with open(os.path.join(abs_path, 'price_predictor/__init__.py')) as f:
        for line in f:
            m = re.match("__version__ = '(.*)'", line)
            if m:
                return m.group(1)
    raise SystemExit("Could not find version string.")

actual_version = str(get_version())
    
setup(
  name = 'price_predictor',
  packages = ['price_predictor'],
  version = actual_version,
  license='MIT',
  description = 'Quickly predict the future prices of financial instruments with a customizable LSTM Recurrent Neural Network',
  long_description = open(os.path.join(abs_path, 'README.rst')).read(),
  author = 'Ludovico Lemma', 
  author_email = 'lwdovico@protonmail.com',
  url = 'https://github.com/ludovicolemma/price-predictor',
  download_url = 'https://github.com/ludovicolemma/price-predictor/archive/refs/tags/v{version}.tar.gz'.format(version = actual_version),
  keywords = ['LSTM',
              'Machine Learning',
              'Market Price Prediction', 
              'Stock Price Prediction', 
              'Financial Forecasting'],
  python_requires='>=3.5',
  install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'sklearn',
          'tensorflow',
      ],
  entry_points={'console_scripts': ['price_predictor=price_predictor.__main__:main']},
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Financial and Insurance Industry',
    'Intended Audience :: Developers',
    'Intended Audience :: End Users/Desktop',
    'Topic :: Software Development :: Build Tools',
    'Topic :: Office/Business :: Financial :: Investment',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)
