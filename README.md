# Portfolio Optimization Finance

While on vacation I tried to learn reinforcement learning for finance.
This is the code I came up with that accompanies the [blog post](https://medium.com/p/b621c18a69d5)

In a nutshell, I tried to make an agent that picks a portfolio of stocks
and take transaction costs into account. It doesn't really work but I learned
a lot and if you submit some PRs it will get better

## Structure

**main.ipynb** is a notebook that I used to run the code and see what's happening
**SingleSine.ipynb** is the first notebook I did, with experiments on a single sine wave.
**learners** is a directory you can add more learning algorithms to
**learers/a2c** is an actor critic implementation I copy pasted from [Denny Britz](https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient)
**env** has stuff related to the environemt
**env/pricegenerator.py** has code that makes a cool synthetic market, that looks random but has learnable stuff going on
**env/Env.py** is an environemnt for RL agents. It takes actions and gives rewards

## Warning
All of this code is "doodle  code" that I was just fooling around with.
It's probably buggy, and it's a bad idea to copy paste any of it with the
assumption it does what I say.
However, it is a good idea to try for yourself, fix it up and make pull requests.

