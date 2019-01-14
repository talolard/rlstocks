import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
class PolicyEstimator():
    """
    Policy Function approximator. Actions are weightings over a portfolio of n stocks + cash
    """

    def __init__(self, learning_rate=0.01, scope="policy_estimator",num_stocks=2,period=10):
        with tf.variable_scope(scope):
            self.price_history_in = tf.placeholder(tf.float32, [period, num_stocks],
                                                "prices")  # HIST prices, current, and position
            self.price_history = tf.reshape(self.price_history_in, [period * num_stocks])
            self.portfolio= tf.placeholder(tf.float32, [num_stocks+1,], "prices")  # HIST prices, current, and position

            self.target = tf.placeholder(dtype=tf.float32, name="target")  # Position size
            state = tf.concat([self.price_history,self.portfolio],axis=0)

            state = tf.expand_dims(state,0)
            # This is just linear classifier
            l1 = tf.contrib.layers.fully_connected(
                inputs=state,
                num_outputs=32,
                activation_fn=tf.nn.tanh,
                weights_initializer=tf.initializers.glorot_uniform)
            l1 = tf.contrib.layers.layer_norm(l1)

            l2 = tf.contrib.layers.fully_connected(
                inputs=l1,
                num_outputs=32,
                activation_fn=tf.nn.tanh,
                weights_initializer=tf.initializers.glorot_uniform)
            l2 = tf.contrib.layers.layer_norm(l1+l2)

            self.alphas= tf.contrib.layers.fully_connected(
                inputs=l2,
                num_outputs=num_stocks+1,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.initializers.glorot_uniform)

            self.alphas +=1

            self.dirichlet  = tfp.distributions.Dirichlet(self.alphas)
            self.action = self.dirichlet._sample_n(1)
            self.action = tf.squeeze(self.action)

            # Loss and train op
            self.loss = -self.dirichlet.log_prob(self.action) * self.target

            # Add cross entropy cost to encourage exploration
            # self.loss -= 1e-1 * self.dirichlet.entropy()

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state_dict, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action, {
            self.price_history_in: state_dict['prices'],
            self.portfolio:state_dict['portfolio']


        })

    def update(self, state_dict, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.price_history_in: state_dict['prices'],
            self.portfolio:state_dict['portfolio']
            , self.target: target, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

class ValueEstimator():
    """
    Value Function approximator. EE.g. the critic
    """

    def __init__(self, learning_rate=0.01, scope="value_estimator", num_stocks=2, period=10):
        with tf.variable_scope(scope):
            self.price_history_in = tf.placeholder(tf.float32, [ period,num_stocks],
                                                "prices")  # HIST prices, current, and position
            self.price_history = tf.reshape(self.price_history_in,[period*num_stocks])
            self.portfolio = tf.placeholder(tf.float32, [num_stocks + 1, ],
                                            "prices")  # HIST prices, current, and position

            self.target = tf.placeholder(dtype=tf.float32, name="target")  # Position size
            state = tf.concat([self.price_history, self.portfolio],axis=0)
            state = tf.expand_dims(state,0)
            state = tf.contrib.layers.layer_norm(state)
            # This is just linear classifier
            l1 = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state, 0),
                num_outputs=16,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.initializers.glorot_uniform)
            l1 = tf.contrib.layers.layer_norm(l1)

            self.estimate = tf.contrib.layers.fully_connected(
                inputs=l1,
                num_outputs=1,
                activation_fn=tf.nn.tanh,
                weights_initializer=tf.initializers.glorot_uniform)
            self.estimate = tf.squeeze(self.estimate)
            self.loss = tf.squared_difference(self.estimate, self.target)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state_dict, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.estimate, {
            self.price_history_in: state_dict['prices'],
            self.portfolio: state_dict['portfolio']

        })

    def update(self, state_dict, target,  sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.price_history_in: state_dict['prices'],
                     self.portfolio: state_dict['portfolio']
            , self.target: target, }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


import collections
import itertools


def reinforce(estimator_policy, estimator_value,price_func,EnvFactory, num_episodes, discount_factor=1.0, length=1000,num_stocks=3,lookback=10):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    envs = []
    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        env = EnvFactory(price_func,num_stocks,length,starting_value=1000,lookback=lookback)
        env.time = lookback +1 # Make sure enough prices in the buffer
        episode = []
        state, reward, done = env.step(np.array([0]*num_stocks+[1])) # Set portfolio to all cash

        # One step in the environment
        eps = 0#(1-1/(i_episode+1))**2
        for t in itertools.count():

            # Take a step
            nextPortfolio = estimator_policy.predict(state)
            if np.random.binomial(1,eps):
                nextPortfolio = np.random.dirichlet([0.1]*num_stocks+[0.2])
            next_state, reward, done = env.step(nextPortfolio)

            # Keep track of the transition
            episode.append(Transition(
                state=state, action=nextPortfolio, reward=reward, next_state=next_state, done=done))

            # Calculate TD Target
            value_next = estimator_value.predict(next_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - estimator_value.predict(state)

            # Update the value estimator
            val_loss  = estimator_value.update(state, td_target)

            # Update the policy estimator
            # using the td error as our advantage estimate
            pol_loss = estimator_policy.update(state, td_error, nextPortfolio)

            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{}, {} - {} - ".format(
                t, i_episode + 1, num_episodes,val_loss,pol_loss ), end="")

            if t > length or done:
                break

            state = next_state
        print("Val {}".format(env.account_value))
        yield env

    return envs
