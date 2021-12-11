# Tesla-Stock-Price-Prediction-using-LSTM
The Dataset has been taken from kaggle using the following link https://www.kaggle.com/faressayah/stock-market-analysis-prediction-using-lstm/data?select=Tesla.csv+-+Tesla.csv.csv

##The model used is
```
Stock_pred = Sequential()

Stock_pred.add(LSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1], 1)))
Stock_pred.add(Dropout(0.1))

Stock_pred.add(LSTM(units = 60, return_sequences = True))
Stock_pred.add(Dropout(0.1))

Stock_pred.add(LSTM(units = 60, return_sequences = True))
Stock_pred.add(Dropout(0.1))

Stock_pred.add(LSTM(units = 60))
Stock_pred.add(Dropout(0.1))

Stock_pred.add(Dense(units = 1))

Stock_pred.compile(optimizer = 'adam', loss = 'mean_squared_error')
##Training the model 
Stock_pred.fit(X_train, y_train, epochs = 25, batch_size = 16)
```

**The predicted stock price is shown using a plot**



![download](https://user-images.githubusercontent.com/67748159/145677539-57929932-c9ca-4978-bb1a-2993295365ad.png)




