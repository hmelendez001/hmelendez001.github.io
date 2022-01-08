## Udacity Data Scientist Nanodegree: Project 4 Final Capstone Project

As part of the udacity.com Data Scientist nanodegree, this is project four and the final project of the program, otherwise known as our Capstone Project. This project involves building a stock price predictor that takes daily trading data over a certain date range as input, and outputs projected estimates for given query dates. The inputs will contain multiple metrics, such as opening price (Open), highest price the stock traded at (High), how many stocks were traded (Volume) and closing price adjusted for stock splits and dividends (Adjusted Close); we only need to predict the Adjusted Close price.

Investment firms, hedge funds, and even individuals or day traders have been using financial models to better understand market behavior and make profitable investments and trades. A wealth of information is available in the form of historical stock prices and company performance data, suitable for machine learning algorithms to process. According to an article by Mark Kolakowski ["How Robots Rule the Stock Market (SPX, DJIA)"](https://www.investopedia.com/news/how-robots-rule-stock-market-spx-djia):

> only 10% of trading volume now comes from human discretionary investors, per data from JPMorgan Chase & Co. (JPM) cited by Bloomberg

![Crypto Trading Robot Image](https://insidebitcoins.com/wp-content/uploads/2020/05/automated_forex_robot-1200x900-1-768x576.jpg)
##### Source: "Best Forex Robots 2021" by Kane Pepi at https://insidebitcoins.com/forex-robot

---
### Problem Statement
For this project, I built a stock price predictor that reads historical daily trading data over a certain date range as input from the Yahoo! Finance API, and outputs projected closing price predictions for given future query dates. The inputs contain multiple metrics for any given day:

- Open: stock opening price, 
- High: highest price the stock traded, 
- Low: highest price the stock traded, 
- Volume: how many stocks were bought and sold,
- Close: the closing unadjusted price of the stock, and
- Adjusted Close: the closing price adjusted for stock splits and dividends

For historical test data we only need to predict the Adjusted Close price, however, for future dates we need to predict the inputs as well.
