|price-predictor PyPI Project Page| |MIT License| |Supported Python
Versions|

Quickly predict the future prices of financial instruments with a
customizable LSTM Recurrent Neural Network

**Price-Predictor**

-  You need only a **valid yahoo finance** symbol like:
   "`EURUSD=X <https://it.finance.yahoo.com/quote/EURUSD=X?p=EURUSD=X>`__",
   "`^DJI <https://it.finance.yahoo.com/quote/^DJI?p=^DJI>`__" or
   "`AMZN <https://it.finance.yahoo.com/quote/AMZN?p=AMZN>`__"

-  Download temporarily the financial data from the optionally specified
   date (if not specified it starts from july 2010)

-  A LSTM RNN will be trained and it will **automatically predict** the
   next **opening price** of the instrument

Installation
------------

With Python and Pip installed, do:

.. code:: sh

   pip3 install price-predictor

Wait for the dependencies to be installed, tensorflow may need a few
minutes.

Command-line usage
-----

Write the following command from your shell:

.. code:: sh

    price_predictor [symbol] [--date (str)] [--ratio (float)] [--layers (int)] [--epochs (int)]

- The **symbol** is just a short code of the financial instrument that must be available on yahoo finance
- The **date** must be provided in the ISO format, for example: 2015-01-25
- The **ratio** is the percentage used to divide training and test set, the higher the closest to the current trends the prediction will turn out, but it may overfit (just ignore if not know how to use)
- The **layers** are the number of layers of the LSTM, the minimum number possible will be always 2, the dafault is 4
- The **epochs** are the number of of epochs the neural network will train, the default is 10 but it can be raised

Documentation
-----

Importing the library

.. code:: sh

   from price_predictor import price_predictor as pp

Then you can use the same function as in the command line and a few more
tools

You can use 2 functions to query and temporarily store the csv and another
one which is used for the command line to get quick results (here it will 
plot them automatically):

.. code:: sh

   pp.yahoo_finance_csv(code, start_from_date = '2010-07-01', end_to_date = today(), interval = 'd')
   
   pp.quick_tomorrow(code, plot = True, start_from_date, training_to_test_ratio, n_layers, n_epochs)
  
- end_to_date is the date where there must stop the download of data
- interval can be 'd' for day, 'wk' for weeks and 'mo' for months
   
Then there are two classes you can work with, the basic one is:

.. code:: sh

   model = pp.Price_Predictor(code, start_from_date = '2010-07-01', end_to_date = today(), interval = 'd', 
                              time_stamps = 30, training_to_test_ratio = 0.7, n_layers = 4, n_epochs = 15, verbose = 0, 
                              load_model = False, path_load = 'model_saved', fit_at_start = False, days_forward = 1)
   
- time_stamps is the length of sequences to consider and use in the training
- verbose is 0 if you do not want any training info output, 1 if you want the progress bar, 2 if you want the description of each training epoch
- load model will load the model stored in the cwd with name = path_load
- fit_at_start is used to avoid manually transforming the data and fitting the model with the method .fit_and_test()
- days_forward is used only if fit_at_start is True, it indicates the how far is the day you want to predict from the last time stamp


This class has a few methods as in the following usage example:

.. code:: sh

   from price_predictor import price_predictor as pp
   import matplotlib.pyplot as plt
   
   fig, axs = plt.subplots(1, 2, figsize=(18,5))
   
   model = pp.Price_Predictor('BTC-EUR', training_to_test_ratio=0.85)
   model.plot_data(ax = axs[0])
   model.fit_and_test(days_forward = 2)
   model.plot_results(ax = axs[1])
   plt.legend()
   plt.show()
   
Output:

|BTC-EUR Example|

- .plot_data() will plot a chart of the training and test data prices with the point of split
- .fit_and_test(days_forward = 2) will scale the data, train the model and test it on the test data, as specified by the parameter it will predict the next price for the day after tomorrow
- .plot_results() will plot a chart of the results of the prediction on the test data

.. code:: sh

   model.predict(input_sequence = None, return_info = True)
   
Output:

   WARNING: No input sequence provided, the records of the data downloaded will be used instead.
   WARNING: The input sequence on which to forecast is longer than 30 which is the input time stamp and the length of array needed in order to get a prediction,the last 30 records will be considered instead.

   In 2 day(s) the price will be: 21021.205
   
   21021.205
   
- return_info = True it will return the warnings and the final print, if False will only return 21021.205
- input_sequence = None it will use the data downloaded, instead if an array or list is specified at least as long as the time_stamps required by the model, the prediction will be based upon that price sequence

Besideds the original parameters it is also possible to access the following relevant attributes of the class in the subsequent way:
- the dataframe used
- the Min Max scaler used
- the training to test split value of the dataframe

.. code:: sh

   model.df
   model.scale
   model.split_val
   
It is possible to access the data and the parameters of class also with the following methods:
   
.. code:: sh

   model.__get_data_frame__()
   model.__get_training_set__()
   model.__get_test_set__()
   model.__get_params__()

It is possible to save and load the trained model as it follows:

.. code:: sh

   model.save_model(dir = 'model_saved')
   model_2 = pp.Price_Predictor('BTC-EUR', 
                                 load_model = True, path_load = 'model_saved', 
                                 fit_at_start = True, days_forward = 2)

As of now the .save_model() method won't store neither the csv nor the scaled data, as such it is necessary to download them again and then scale them either with fit_at_start = True or with the .fit_and_test() method. In both cases the days_forward parameter must be the same as before.

**BETA**: The other class is Predict_Iterator, it inherits all the methods and attributes of the Price_Predictor parent class.

.. code:: sh

   Predict_Iterator(code, start_from_date = '2010-07-01', end_to_date = today(), 
                    effort = 0.5, time_stamps = 30)
                    
The parameter effort is used to manage the computational time, it is best left untouched, it's a coefficient used to concurrently increase or reduce the parameters:

- training_to_test_ratio: from 0.70 to 0.90
- n_layers: from 2 to 4 with a stronger preference towards 2 layers: max(2, 4*effort)
- n_epochs: 2 if effort less than 0.6, 4 if less than 0.75, 10 if less than 0.85 and 15 if greater

The way this class is applied is mainly with the method .get_predictions(), it works like this:

.. code:: sh
   
   iterate_model = Predict_Iterator('FTSEMIB.MI')
   list_of_predictions = iterate_model.get_predictions(days_to_predict = 4)
 
It will predict the next 4 days' prices, the parameter days_to_predict will determin the number of training to perform in a loop with different days_forwards (from 1 to days_to_predict).

The trained models will be accessible through the list attribute stored_models as it follows

.. code:: sh
   
   iterate_model.stored_models
   model_1 = iterate_model.stored_models[0]
   
Each element of the list is a model trained with different future day tergets, once it is accessed it is possible to plot results and manage them as normal Price_Predictor objects.

Notes
----------

It is better to use this tool with financial instruments without a history of substantial price changes, indeed if the price was too high or too low in the past compared to the latest records, the model learnt may be biased towards different levels of prices, it may follow the trend but the amplitude may be completely wrong. 

A possible solution to this problem is changing the starting date from which to gather data, so that the model may not be biased towards past averaage prices, or to increase the training to test split ratio (use with care!).

It is also possible some tensorflow warnings may be given using the command-line with some OS without correct GPU drivers

Disclaimer
----------

I am in no way affiliated with, authorized, maintained or endorsed by
Yahoo Finance or any of its affiliates or subsidiaries. This is an
independent and unofficial project.

It is licensed under an MIT license. Refer to the ``LICENSE`` file for
more information.

.. |price-predictor PyPI Project Page| image:: https://img.shields.io/pypi/v/price-predictor.svg
   :target: https://pypi.org/project/price-predictor/
.. |MIT License| image:: https://img.shields.io/github/license/ludovicolemma/price-predictor.svg
   :target: https://github.com/ludovicolemma/price-predictor/blob/main/LICENSE
.. |Supported Python Versions| image:: https://img.shields.io/pypi/pyversions/price-predictor.svg
.. |BTC-EUR Example| image:: https://raw.githubusercontent.com/ludovicolemma/price-predictor/main/examples/btc-eur.png
