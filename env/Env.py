import numpy as np
from collections import  defaultdict

from env.priceGenerator import make_stock

costPerShare = 0 # 0.01
class Env:
    '''
    A simple environemnt for our agent,
    the action our agent gives is  weighting over the stocks + cash
    the env calcutes that into stock and figures out the returns
    '''
    def __init__(self,price_fn,num_stocks=2,length=2,starting_value=1000,lookback=10):
        '''

        :param price_fn:  A function that returns a numpy array of prices
        :param num_stocks: How many stocks in our univerese
        :param length: The length of an episode
        '''
        self.num_stocks = num_stocks
        self.lookback = lookback
        self.length = length
        self.oprices= price_fn(num_stocks=num_stocks,length=length)
        self.prices = np.concatenate([self.oprices,np.ones([length+1,1])],axis=1) #attach the value of cash
        self.portfolio = np.zeros([num_stocks+1]) #2k and 2k+1 are te long and short of a stock. portfolio[-1] is cash
        self.portfolio[-1] = 1
        self.time =0
        self.__account_value = starting_value
        self.__shares=np.array([0]*num_stocks +[starting_value])
        self.hist = defaultdict(list)
    @property
    def shares(self):
        return self.__shares
    @property
    def account_value(self):
        return self.__account_value
    @shares.setter
    def shares(self,new_shares):

        self.__shares = new_shares
        self.hist['shares'].append(self.shares)
    @account_value.setter
    def account_value(self,new_act_val):
        self.__account_value = new_act_val
        try:
            act_returns  = self.account_value / self.hist['act_val'][-1]
        except:
            act_returns =1
        self.hist['act_val'].append(self.account_value)
        self.hist['act_returns'].append(act_returns)

    def step(self,new_portfolio):
        '''
        Get the next prices. Then transition the value of the account into the desired portfolio
        :param new_portfolio:
        :return:
        '''
        self.time +=1

        self.update_acount_value(new_portfolio)
        reward = np.log(self.hist['act_returns'][-1]) #already includes transaction costs
        state = {
            "prices":self.prices[self.time-self.lookback+1:self.time+1,:-1], # All prices upto now inclusive but no cash
            "portfolio":self.portfolio,

        }
        done = self.time >=len(self.prices)-2
        return state,reward,done




    def update_acount_value(self,new_portfolio):
        currentShareValues = self.shares * self.prices[self.time]
        currentAccountValue = sum(currentShareValues)

        currentPortfolioProportions = currentShareValues / currentAccountValue
        desiredCashChange = (new_portfolio -currentPortfolioProportions )* currentAccountValue

        desiredChangeInShares = np.floor(desiredCashChange / self.prices[self.time])
        self.shares = self.shares + desiredChangeInShares
        newAccountValue = np.sum(self.shares*self.prices[self.time])
        #becuse we take the floor, sometimes we lose cash for no reason. This is a fix
        missingCash  = currentAccountValue - newAccountValue
        transactionCost = sum(np.abs(desiredChangeInShares[:-1])*costPerShare)

        self.shares[-1] += missingCash - transactionCost

        transactionCost = sum(np.abs(desiredChangeInShares[:-1])*costPerShare)
        self.hist["changeInShares"].append(desiredChangeInShares)
        self.hist["transactionCosts"].append(transactionCost)

        self.account_value =np.sum(self.shares*self.prices[self.time])





