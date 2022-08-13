from .price_predictor import quick_tomorrow
from argparse import ArgumentParser

def main():
    name_library = 'price_predictor'
    parser = ArgumentParser(description="", add_help=True, usage="""
    {0} [symbol] #look for the symbol on yahoo finance
    {2:{1}} [--date (str)]
    {2:{1}} [--target (str)]
    {2:{1}} [--stamps (int)]
    {2:{1}} [--ratio (float)]
    {2:{1}} [--layers (int)]
    {2:{1}} [--epochs (int)]
    {0} --help""".format(name_library, len(name_library), ''))
    
    tomorrow = parser.add_argument_group("Get tomorrow's opening market price",
                                          "Specify a symbol present on yahoo finance, you can set the training to test ratio, the n° of and ephochs of the LSTM. "
                                          "If not specified the parameters will be set as it follows: "
                                          "date = '2010-07-01', \ntarget_value = 'Open', \ntime_stamps = 30, \ntraining_to_test_data_ratio = 0.9, \nn_layers = 4, \nn_epochs = 10")
    
    tomorrow.add_argument('symbol', type=str, nargs='*', help='Yahoo finance symbol to download the CSV with the data')
    tomorrow.add_argument('-d', '--date', type=str, help='Specify the date from which to start gathering data', metavar = '')
    tomorrow.add_argument('-t', '--target', type=str, help="Specify a target between Open, Close, High, Low, 'Adj Close' and Volume", metavar = '')
    tomorrow.add_argument('-s', '--stamps', type=int, help='Specify how many time stamps to have in each record training sequence', metavar = '')
    tomorrow.add_argument('-r', '--ratio', type=float, help='Specify the ratio to split the training and test data', metavar = '')
    tomorrow.add_argument('-l', '--layers', type=int, help='Specify the number of layers of the neural network (if less than 2 is specified it will be 2 by default)', metavar = '')
    tomorrow.add_argument('-e', '--epochs', type=int, help='Specify the number of epochs of the neural network', metavar = '')

    args = parser.parse_args()

    if args.symbol == []:
        parser.print_help()
    
    else:
        #I just need the prediction, no need of the model in the command line, thus [1]
        quick_tomorrow(code = args.symbol[0], plot  = True,
                       start_from_date = args.date,
                       target_value = args.target,
                       time_stamps = args.stamps,
                       training_to_test_ratio = args.ratio, 
                       n_layers = args.layers,
                       n_epochs = args.epochs)[1]

if __name__ == '__main__':
    main()
