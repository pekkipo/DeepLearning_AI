import tensorflow as tf
import numpy as np
import gym

num_inputs = 4
num_hidden = 4
num_outputs = 1  # probability to go left  1-left=prob to go right

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, num_inputs])

hidden_layer_one = tf.layers.dense(X, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer)
hidden_layer_two = tf.layers.dense(hidden_layer_one, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer) # num hidden cauz number of neurons stays the same

output_layer = tf.layers.dense(hidden_layer_two, num_outputs, activation=tf.nn.sigmoid, kernel_initializer=initializer) # basically pass number of inputs from previous layer and number of outputs
# return a single number - probability to go left

probabilities = tf.concat(axis=1, values=[output_layer, 1-output_layer])
# concat so that we get the action using multinomial

action = tf.multinomial(probabilities, num_samples=1)
# not sure what that means exactly, but we don't want just to choose the higher probability every time cauz it might get us stuck in a loop of performing the same actions

init = tf.global_variables_initializer()

# train the session
epi = 50  # episodes of the game, within each episode we can take 500 steps tops
step_limit = 500  # limit os steps in a training session
env = gym.make('CartPole-v0')

avg_steps = []

with tf.Session() as sess:
    init.run()

    for i_episode in range(epi):
        obs = env.reset()

        for step in range(step_limit):
            # action is defined above, based on the NN output
            action_val = action.eval(feed_dict={X:obs.reshape(1, num_inputs)}) # reshape cauz tf needs it in that shape
            obs, reward, done, info = env.step(action_val[0][0])
            # because of the way multinomial returns values

            if done:  # maybe the pole falls over or something
                avg_steps.append(step)
                print("Done after {} steps".format(step))
                break # from the for loop above


print("After {} episodes, average steps per game was {}".format(epi, np.mean(avg_steps)))
env.close()

# here the NN won't perfrom well because we aren't even learning based on the history of previous actions
# we just consider only previous single action
# so must use Policy Gradient theory to fix that