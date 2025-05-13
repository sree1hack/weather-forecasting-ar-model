from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR  #SVM Forecasting
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ARDRegression

main = tkinter.Tk()
main.title("Weather Forecasting Using Autoregressive Models") #designing main screen
main.geometry("800x700")

global filename, mse_list, mae_list, scaler, scaler1
global X_train, X_test, y_train, y_test, dataset, model
global X, Y

     
def uploadDataset(): 
    textarea.delete('1.0', END)
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    textarea.insert(END, str(dataset))
    
def processDataset():
    global dataset, X, Y
    textarea.delete('1.0', END)
    dataset.fillna(0, inplace = True)#replace missing values
    dataset['Date/Time'] = dataset['Date/Time'].astype('datetime64[ns]')#converting object to datetime
    #extract time series features from dataset
    dataset['day'] = dataset['Date/Time'].dt.day
    dataset['month'] = dataset['Date/Time'].dt.month
    dataset['year'] = dataset['Date/Time'].dt.year
    dataset['hour'] = dataset['Date/Time'].dt.hour
    dataset['minute'] = dataset['Date/Time'].dt.minute
    dataset['second'] = dataset['Date/Time'].dt.second
    dataset.drop(['Date/Time', 'Weather'], axis = 1,inplace=True)
    textarea.insert(END, str(dataset))

def splitDataset():
    textarea.delete('1.0', END)
    global dataset, X, Y, scaler, scaler1
    global X_train, X_test, y_train, y_test
    data = dataset.values
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler1 = MinMaxScaler(feature_range = (0, 1))
    #extracting X training features and Y target value as temperature
    Y = data[:,0:1]
    X = data[:,1:data.shape[1]-1]
    #normalizing X and Y features
    X = scaler.fit_transform(X)
    Y = scaler1.fit_transform(Y)
    #split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    textarea.insert(END,"Dataset Size : "+str(X.shape[0])+"\n")
    textarea.insert(END,"80% images are used to train DenseNet169 : "+str(X_train.shape[0])+"\n")
    textarea.insert(END,"20% images are used to train DenseNet169 : "+str(X_test.shape[0])+"\n")

def prediction(algorithm, y_test, predict):
    global mse_list, mae_list, scaler1
    #calculating MSE & MAE error
    mse_error = mean_squared_error(y_test,predict)
    mae_error = mean_absolute_error(y_test,predict)
    mae_list.append(mae_error)
    mse_list.append(mse_error)
    textarea.insert(END,algorithm+" MSE : "+str(mse_error)+"\n")
    textarea.insert(END,algorithm+" MAE : "+str(mae_error)+"\n\n")
    predict = predict.reshape(predict.shape[0],1)
    predict = scaler1.inverse_transform(predict)
    predict = predict.ravel()
    labels = scaler1.inverse_transform(y_test)
    labels = labels.ravel()
    labels = labels[0:100]
    predict = predict[0:100]
    for i in range(0, 20):
        textarea.insert(END,"Original Temperature : "+str(labels[i])+" "+algorithm+" Forecasting Temperature : "+str(predict[i])+"\n")
    #plotting comparison graph between original values and predicted values
    plt.plot(labels, color = 'red', label = 'Original Temperature')
    plt.plot(predict, color = 'green', label = algorithm+' Forecasting temperature')
    plt.title(algorithm+" Temperature Forecasting Graph")
    plt.xlabel('Hours')
    plt.ylabel('Temperature Forecasting')
    plt.legend()
    plt.show()    
    
def trainSVM():
    textarea.delete('1.0', END)
    global mse_list, mae_list, X_train, X_test, y_train, y_test, scaler
    mse_list = []
    mae_list = []
    svm_forecast = SVR(kernel="linear", C=5.0, epsilon=0.2)
    #training SVM with X and Y data
    svm_forecast.fit(X_train, y_train.ravel())
    #performing prediction on test data
    predict = svm_forecast.predict(X_test)
    prediction("SVM", y_test, predict)

def trainAR():
    textarea.delete('1.0', END)
    global model, mse_list, mae_list, X_train, X_test, y_train, y_test, scaler

    if 'mae_list' not in globals():
        mae_list = []
    if 'mse_list' not in globals():
        mse_list = []

    model = ARDRegression()
    model.fit(X_train, y_train.ravel())
    predict = model.predict(X_test)
    prediction("AutoRegression", y_test, predict)


def graph():
    global mae_list, mse_list

    data = []

    try:
        if len(mae_list) > 0 and len(mse_list) > 0:
            data.append(['SVM', 'MAE', mae_list[0]])
            data.append(['SVM', 'MSE', mse_list[0]])
        if len(mae_list) > 1 and len(mse_list) > 1:
            data.append(['AutoRegression', 'MAE', mae_list[1]])
            data.append(['AutoRegression', 'MSE', mse_list[1]])
    except Exception as e:
        textarea.insert(END, f"⚠️ Error preparing graph data: {e}\n")
        return

    if not data:
        textarea.insert(END, "⚠️ Please train at least one model before generating the graph.\n")
        return

    try:
        df = pd.DataFrame(data, columns=['Algorithms', 'Performance Output', 'Value'])
        df.pivot(index="Algorithms", columns="Performance Output", values="Value").plot(kind='bar')
        plt.rcParams["figure.figsize"] = [8, 5]
        plt.title("All Algorithms Performance Graph")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        textarea.insert(END, f"⚠️ Failed to plot graph: {e}\n")


def predictWeather():
    textarea.delete('1.0', END)
    global model, scaler, scaler1
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)#replace missing values
    dataset['Date/Time'] = dataset['Date/Time'].astype('datetime64[ns]')#converting object to datetime
    temp = dataset.values
    #extract time series features from dataset
    dataset['day'] = dataset['Date/Time'].dt.day
    dataset['month'] = dataset['Date/Time'].dt.month
    dataset['year'] = dataset['Date/Time'].dt.year
    dataset['hour'] = dataset['Date/Time'].dt.hour
    dataset['minute'] = dataset['Date/Time'].dt.minute
    dataset['second'] = dataset['Date/Time'].dt.second
    dataset.drop(['Date/Time', 'Weather'], axis = 1,inplace=True)
    data = dataset.values
    X = data[:,0:data.shape[1]-1]
    X = scaler.transform(X)
    print(X.shape)
    predict = model.predict(X)
    predict = predict.reshape(predict.shape[0],1)
    predict = scaler1.inverse_transform(predict)
    predict = predict.ravel()
    for i in range(len(predict)):
        textarea.insert(END,"Test Data = "+str(temp[i])+" Predicted Weather Temperature : "+str(predict[i])+"\n\n")
    
    
    
font = ('times', 16, 'bold')
title = Label(main, text='Weather Forecasting Using Autoregressive Models', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 14, 'bold')
uploadDataset = Button(main, text="Upload Weather Dataset", command=uploadDataset)
uploadDataset.place(x=20, y=100)
uploadDataset.config(font=font1)  

processDataset = Button(main, text="Preprocess Dataset", command=processDataset)
processDataset.place(x=300, y=100)
processDataset.config(font=font1)

splitDataset = Button(main, text="Split Dataset Train & Test", command=splitDataset)
splitDataset.place(x=560, y=100)
splitDataset.config(font=font1)

svmButton = Button(main, text="Train SVM Algorithm", command=trainSVM)
svmButton.place(x=850, y=100)
svmButton.config(font=font1)

arButton = Button(main, text="Train Autoregressive Models", command=trainAR)
arButton.place(x=20, y=150)
arButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=300, y=150)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Weather from Test Data", command=predictWeather)
predictButton.place(x=560, y=150)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
textarea=Text(main,height=25,width=110)
scroll=Scrollbar(textarea)
textarea.configure(yscrollcommand=scroll.set)
textarea.place(x=10,y=200)
textarea.config(font=font1)

main.config(bg='light coral')
main.mainloop()
