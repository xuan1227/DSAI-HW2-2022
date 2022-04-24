from genericpath import exists
from re import A
import numpy as np
import tensorflow as tf
from keras.layers import Dense, LSTM
from keras.models import Sequential
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
import os

register_matplotlib_converters()
RANDOM_SEED = 42
test_size = 15
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def Train(csv_train,csv_test,csv_output):
    global X_test
    global cnt_transformer
    global y_train
    global y_test
    global day
    df = pd.read_csv(csv_train,header=None)
    df2 = pd.read_csv(csv_test,header=None)
    
    df.columns = ['Open', 'High', 'Low','Close']
    df2.columns = ['Open', 'High', 'Low','Close']

    f_columns = ['Open', 'High', 'Low','Close']
    #df.loc[:,f_columns]*=100
    #df2.loc[:,f_columns]*=100

    y = df['Open'].values.reshape(- 1, 1)
    y2 = df2['Open'].values.reshape(- 1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    scaler2 = scaler2.fit(y2)
    y = scaler.transform(y)
    y2 = scaler.transform(y2)
    n_forecast = 1 
    n_lookback = 60
    
    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)
    Y = Y.reshape(Y.shape[0])
    # train the model
    tf.random.set_seed(0)

    if not os.path.exists("model.h5"):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(loss='mse', optimizer='adam')
        model.fit(X, Y, epochs=1000, batch_size=128, validation_split=0.2, verbose=0)
        model.save('model.h5')
    else:
        model = keras.models.load_model("model.h5")
    # generate the multi-step forecasts
    n_future = int(len(df2))
    y_future = []

    x_pred = X[-1:, :, :]  # last observed input sequence
    y_pred = Y[-1]         # last observed target value

    output = []
    for i in range(n_future):

        # feed the last forecast back to the model as an input
        x_pred = np.append(x_pred[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)
        print(x_pred[:, 1:, :])
        # generate the next forecast
        y_pred = model.predict(x_pred)

        # save the forecast
        y_future.append(y_pred.flatten()[0])

        y_pred = np.array(y_pred).reshape(-1, 1)

        y_pred = scaler.inverse_transform(y_pred)

        # read the true price
        y_pred = y2[i]

    
    # transform the forecasts back to the original scale
    y_future = np.array(y_future).reshape(-1, 1)
    y_future = scaler.inverse_transform(y_future)

    # organize the results in a data frame
    df_past = df[['Open']].reset_index()
    df_past.rename(columns={'index': 'Date'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan


    df_future = pd.DataFrame(columns=['Date', 'Open', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_future)
    df_future['Forecast'] = y_future.flatten()
    df_future['Open'] = np.nan

    results = df_past.append(df_future).set_index('Date')
    past, future = results.iloc[0:len(df)], results.iloc[len(df):len(results)]
    past.round(2)
    future.round(2)
    data=future['Forecast']
    dict={'Open':data}
    pd.DataFrame(dict).to_csv('Future.csv',index=False)
    f= open("Future.csv","r")
    lines=f.readlines()
    f= open("Future.csv","w")
    f.writelines(lines[1:])
    # plot the results
    history = past['Open']
    predict = future['Forecast']
    f.close()
    # plt.plot(np.arange(0, len(past)), history, 'g', label="history")
    # plt.plot(np.arange(len(past), len(past) + len(future)), df2['Open'], marker='.', label="true")
    # plt.plot(np.arange(len(past), len(past) + len(future)), predict, 'r', label="prediction")
    # plt.ylabel('Price')
    # plt.xlabel('Day')
    # plt.legend()
    # plt.show()

def Test(file_test,file_input,file_output):
    test = pd.read_csv(file_test,header=None)
    output = open(file_output,"w")
    input = open(file_input,"r")
    test = test.iloc[:,0]
    input_list = []
    test_list = []
    stage = 0
    for i in test:
        test_list.append(float(i))
    for i in input.readlines():
        input_list.append(float(i))
    for i in range(len(input_list)-1):
        if(input_list[i]>test_list[i] and stage == 0):
            output.write("1\n")
            stage=1
        elif(input_list[i]>test_list[i] and stage == 1):
            output.write("0\n")
            stage=1
        elif(input_list[i]>test_list[i] and stage == -1):
            output.write("1\n")
            stage=0
        elif(input_list[i]<test_list[i] and stage == 0):
            output.write("-1\n")
            stage=-1
        elif(input_list[i]<test_list[i] and stage == 1):
            output.write("-1\n")
            stage=0
        elif(input_list[i]<test_list[i] and stage == -1):
            output.write("0\n")
            stage = -1
        else:
            output.write("0\n")
            stage = 0
# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--testing',
                       default='testing_data.csv',
                       help='input testing data file name')

    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    
    #Train(args.training,args.testing,args.output)
    Test(args.testing,"Future.csv",args.output)

    

