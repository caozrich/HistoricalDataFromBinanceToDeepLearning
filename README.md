# Historical Data from Binance to DeepLearning

This is an example of how to get historical data of a currency pair from Binance API and apply a basic indicator to it and then preprocess the data to be used in deep learning projects (to use the binance API you must have an account).


## libraries required - Python 3.6 or later

- numpy.
- pandas.
- sklearn.
- ta.
- binance.

Clone and Use the Repository
To use this repository and obtain historical data from Binance for deep learning projects, follow these steps:

1. Clone the Repository
First, clone this repository to your local machine using the git clone command in your terminal:

bash
Copy code
```python
git clone https://github.com/caozrich/HistoricalDataFromBinanceToDeepLearning.git
```
2. Install Dependencies
Make sure you have Python 3.7 or later installed on your system. Then, install the required libraries by running the following command in the terminal:

bash
Copy code
```python
pip install numpy pandas scikit-learn ta binance
```
This will install the necessary libraries, including numpy, pandas, scikit-learn, ta, and binance.

3. Configure Binance API Credentials
To use the Binance API, you need to configure your credentials. Open the config.py file in the repository and replace "Your_API_Key_Here" and "Your_API_Secret_Here" with your Binance API key and API secret, respectively:

python
Copy code
API_KEY = "Your_API_Key_Here"
API_SECRET = "Your_API_Secret_Here"
