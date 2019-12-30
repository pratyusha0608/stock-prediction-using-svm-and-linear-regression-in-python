# stock-prediction-using-svm-and-linear-regression-in-python

 download python shoftware 
 http://python.org/download/
 got the data from nsc(nationl stock exchnage) website using python code
from datetime import date
from nsepy import get_history
infy = get_history(symbol='ITNFY',
start=date(1998,1,1),
end=date(2019,1,1))
#converting data into a csv file 
infy.to_csv('infy.csv', mode='a', header="FALSE")

using this data creating svm model and linear regression

