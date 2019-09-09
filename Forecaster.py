import pandas as pd


#imports data from CSV file
url = "https://raw.githubusercontent.com/RuchirB/jacobsonEntries/master/Formatted.csv"
df = pd.read_csv(url, header=0, index_col=0)


#plots the first 100 datapoints
import matplotlib.pyplot as plt
df[:100].plot(linewidth=2)
plt.grid(which='both')
plt.show()


#Creates a training dataset from the very first date until April 5th, 2015
from gluonts.dataset.common import ListDataset
training_data = ListDataset(
    [{"start": df.index[0], "target": df.index[1000]}],
    freq = "60min"
)

#The estimator creates a prediction object. Since each prediction is of a 5min frequency interval, this next chunk predicts the next hour
#since prediction_length is 12, it will predict the next 12 lines of data 
#This code essentially just trains on the data for 10 epochs

from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer

estimator = DeepAREstimator(freq="5min", prediction_length=12, trainer=Trainer(epochs=10))
predictor = estimator.train(training_data=training_data)


#Takes the rest of the dataset and sets it to be test data instead of training data

test_data = ListDataset(
    [{"start": df.index[0], "target": df.value[:"2015-04-15 00:00:00"]}],
    freq = "5min"
)

from gluonts.dataset.util import to_pandas

#Create a prediction and plot based on test data
for test_entry, forecast in zip(test_data, predictor.predict(test_data)):
    to_pandas(test_entry)[-60:].plot(linewidth=2)
    forecast.plot(color='g', prediction_intervals=[50.0, 90.0])
plt.grid(which='both')