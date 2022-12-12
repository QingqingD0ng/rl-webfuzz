from imp import IMP_HOOK
import imp
from operator import mod
import numpy as np
import random
import mitdeeplearning as mdl
import tensorflow as tf
import os
import time
import math
from collections import namedtuple
import json
from flask import Flask, request, jsonify
import logging
import dill
# set logging
if not os.path.exists('logs'):
    os.mkdir('logs')
if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=time.strftime(
                        'logs/'+'%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))+'.log',
                    filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# add the handler to the root logger
logging.getLogger().addHandler(console)


def create_network(n_actions):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=16, activation='relu'),
        tf.keras.layers.Dense(units=n_actions, activation=None)])
    return model


def get_state(payload):
    index = payload['index']
    size = payload['size']
    state = np.zeros(sum(size))
    offset = 0
    for i, l in zip(index, size):
        state[offset+i] = 1
        offset += l
    return state


class DQN:

    class Memory:
        def __init__(self, capacity, Transition):
            self.capacity = capacity
            self.memory = []
            self.position = 0
            self.Transition = Transition

        def push(self, *args):
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = self.Transition(*args)
            self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size):  # 采样
            return random.sample(self.memory, batch_size)

        def __len__(self):
            return len(self.memory)

    def __init__(self, n_actions, batch_size=32, memory_size=1000, episode=10000, target_update=10, EPSILON=0.1, GAMMA=0.95, learning_rate=0.01, ddqn=False,):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.episode = episode
        self.target_update = target_update
        self.EPSILON = EPSILON
        self.GAMMA = GAMMA
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.Transition = namedtuple(
            'Transition', ('state', 'action', 'reward', 'done', 'next_state'))
        self.target_model = create_network(n_actions)
        self.eval_model = create_network(n_actions)
        self.memory = self.Memory(memory_size, self.Transition)
        self.ddqn = ddqn
        logging.info('DQN MODEL INITIATED SUCCESFULLY!')

    def save(self, save_path):
        if not save_path:
            save_path = self.save_path
        self.target_model.set_weights(self.eval_model.get_weights())
        self.eval_model.save_weights(save_path)
        dict = {}
        properties = ['n_actions', 'batch_size', 'memory_size', 'episode', 'target_update', 'EPSILON', 'GAMMA',
                      'learning_rate', 'ddqn', 'memory', 'smoothed_reward', 'i_episode', 'best_reward', 'total_reward', 'Transition']
        for p in properties:
            dict[p] = self.__dict__[p]
        with open(save_path+".dict", 'wb') as f:
            dill.dump(dict, f)

    def load(self, path):
        if not path:
            return False
        if os.path.isfile(path+".dict"):
            with open(path+".dict", 'rb') as f:
                dict = dill.load(f)
            self.__dict__.update(dict)
            self.eval_model.load_weights(path)
            self.target_model.set_weights(self.eval_model.get_weights())
            if not math.isinf(self.best_reward):
                return True
            else:
                return False

    def choose_action(self, state, single=True):
        state = np.expand_dims(state, axis=0) if single else state
        logits = self.eval_model.predict(state)
        n = random.random()
        if n >= self.EPSILON:
            action = np.argmax(logits, 1)[0]
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def compute_loss(self, states, actions, rewards, dones, next_states):
        q_eval = tf.gather(self.eval_model(states), actions, batch_dims=1)
        if self.ddqn:
            q_next_action = tf.argmax(self.eval_model(next_states), axis=1)
            q_next = tf.gather(self.target_model(
                next_states), q_next_action, batch_dims=1)
        else:
            q_next = tf.reduce_max(self.target_model(next_states), axis=1)
        q_target = rewards + (1-dones) * self.GAMMA * q_next
        loss = tf.keras.losses.MSE(q_eval, q_target)
        return loss

    def train_step(self, model, loss_function, optimizer, custom_fwd_fn=None):
        if self.memory.__len__() < self.batch_size:
            return
        samples = self.memory.sample(self.batch_size)
        batch = self.Transition(*zip(*samples))
        state_batch = np.array(batch.state)
        action_batch = np.array(batch.action)
        reward_batch = np.array(batch.reward)
        done_batch = np.array(batch.done)
        next_state_batch = np.array(batch.next_state)
        with tf.GradientTape() as tape:
            loss = loss_function(state_batch, action_batch,
                                 reward_batch, done_batch, next_state_batch)
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 2)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    def start_training(self, initial_state, save_path):

        logging.info(
            'start training------------------------------------------------')
        self.smoothed_reward = mdl.util.LossHistory(smoothing_factor=0.95)
        # to initiate eval_model weights
        self.choose_action(initial_state)
        # to initiate target_model weights
        self.target_model(np.expand_dims(initial_state, axis=0))
        self.target_model.set_weights(self.eval_model.get_weights())
        self.i_episode = 0
        self.total_reward = 0
        self.best_reward = float('-inf')
        os.mkdir(save_path)
        self.save_path = save_path + 'DDQN' if self.ddqn else save_path + 'DQN'

    def train(self, state, action, next_state, reward, done):
        self.total_reward += reward
        logging.info('train episode:'+str(self.i_episode) +
                     ' total_reward:'+str(self.total_reward))
        t = 1 if done else 0
        self.memory.push(state, action, reward, t, next_state)
        self.train_step(self.eval_model, self.compute_loss, self.optimizer)
        if done:
            logging.info('episode：'+str(self.i_episode) +
                         ' finished '+'total_reward: '+str(self.total_reward))
            self.smoothed_reward.append(self.total_reward)
            if self.best_reward < self.total_reward:
                self.best_reward = self.total_reward
                self.save(self.save_path+"-best")
                logging.info('model saved at :' + str(self.save_path) +
                             ' with reward: '+str(self.total_reward))
            self.total_reward = 0
            self.i_episode += 1
            loss_history = np.array(self.smoothed_reward.get())
            np.save(self.save_path+'_loss_history', loss_history)
            if self.i_episode % self.target_update == 0:
                self.save(self.save_path)


class PolicyGradient:

    class Memory:
        def __init__(self):
            self.clear()

        def clear(self):
            self.states = []
            self.actions = []
            self.rewards = []

        def add_to_memory(self, new_state, new_action, new_reward):
            self.states.append(new_state)
            self.actions.append(new_action)
            self.rewards.append(new_reward)

        def __len__(self):
            return len(self.actions)

    def __init__(self, n_actions, episode=10000, GAMMA=0.95, learning_rate=0.01):
        self.n_actions = n_actions
        self.episode = episode
        self.GAMMA = GAMMA
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.model = create_network(n_actions)
        self.memory = self.Memory()
        logging.info('POLICY GRADIENT MODEL INITIATED SUCCESFULLY!')

    def save(self, save_path):
        if not save_path:
            save_path = self.save_path
        self.model.save_weights(self.save_path)
        with open(save_path+"dict", 'wb') as f:
            dill.dump(self.__dict__, f)

    def choose_action(self, state, single=True):
        state = np.expand_dims(state, axis=0) if single else state
        logits = self.model.predict(state)
        action = tf.random.categorical(logits, num_samples=1)
        action = action.numpy().flatten()
        return action[0] if single else action

    def normalize(self, x):
        x -= np.mean(x)
        x /= np.std(x)
        return x.astype(np.float32)

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        R = 0
        for t in reversed(range(0, len(rewards))):
            R = R * self.GAMMA + rewards[t]
            discounted_rewards[t] = R
        return self.normalize(discounted_rewards)

    def compute_loss(self, logits, actions, rewards):
        neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=actions)
        loss = tf.reduce_mean(neg_logprob * rewards)
        return loss

    def train_step(self, loss_function, optimizer, states, actions, discounted_rewards, custom_fwd_fn=None):
        with tf.GradientTape() as tape:
            if custom_fwd_fn is not None:
                prediction = custom_fwd_fn(states)
            else:
                prediction = self.model(states)

            loss = loss_function(prediction, actions, discounted_rewards)
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 2)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def start_training(self, initial_state, save_path):
        logging.info(
            'start training------------------------------------------------')
        self.smoothed_reward = mdl.util.LossHistory(smoothing_factor=0.95)
        self.i_episode = 0
        self.best_reward = float('-inf')
        os.mkdir(save_path)
        self.save_path = save_path+'PolicyGradient'

    def train(self, state, action, next_state, reward, done):
        logging.info('train episode:', self.i_episode)
        self.memory.add_to_memory(state, action, reward)

        if done:
            self.train_step(self.compute_loss, self.optimizer,
                            np.vstack(self.memory.states),
                            np.array(self.memory.actions),
                            self.discount_rewards(self.memory.rewards))
            total_reward = sum(self.memory.rewards)
            logging.info('train episode:', self.i_episode,
                         'finished. Total reward:', total_reward)

            if self.best_reward < total_reward:
                self.best_reward = total_reward
                self.save(self.save_path+"-best")
                # saveModel(self.save_path)
                logging.info('model saved at :', self.save_path,
                             'with reward:', total_reward)

            self.smoothed_reward.append(total_reward)
            self.i_episode += 1
            self.memory.clear()
            loss_history = np.array(self.smoothed_reward.get())
            np.save(self.save_path+'_loss_history', loss_history)


class QLearning:

    def __init__(self, n_actions, episode=60000, EPSILON=1, GAMMA=0.95, EPSILON_decay=0.995, learning_rate=0.01):
        self.n_actions = n_actions
        self.episode = episode
        self.GAMMA = GAMMA
        self.EPSILON = EPSILON
        self.EPSILON_decay = EPSILON_decay
        self.learning_rate = learning_rate
        self.q_table = np.random.uniform(
            low=0, high=1, size=(n_actions + [n_actions]))
        logging.info('Q-LEARNING MODEL INITIATED SUCCESFULLY!')

    def choose_action(self, state, single=True):
        if np.random.random() > self.EPSILON:
            action = np.argmax(self.q_table[state])
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def save(self, save_path=None):
        if not save_path:
            save_path = self.save_path
        np.save(save_path, self.q_table)
        with open(save_path+"dict", 'wb') as f:
            dill.dump(self.__dict__, f)

    def start_training(self, initial_state, save_path):
        logging.info(
            'start training------------------------------------------------')
        self.smoothed_reward = mdl.util.LossHistory(smoothing_factor=0.95)
        self.i_episode = 0
        self.best_reward = float('-inf')
        self.total_reward = 0
        os.mkdir(save_path)
        self.save_path = save_path + 'Qlearning'

    def train(self, state, action, next_state, reward, done):
        logging.info('train episode:', self.i_episode)
        self.total_reward += reward
        if not done:
            max_future_q = np.max(self.q_table[next_state])
            current_q = self.q_table[state + (action,)]
            new_q = (1 - self.learning_rate) * current_q + \
                self.learning_rate * (reward + self.GAMMA * max_future_q)
            self.q_table[state + (action,)] = new_q

        else:
            logging.info('train episode:', self.i_episode,
                         'finished. Total reward:', self.total_reward)

            if self.best_reward < self.total_reward:
                self.best_reward = self.total_reward
                #np.save(self.save_path, self.q_table)
                self.save(self.save_path)
                logging.info('model saved at :', self.save_path,
                             'with reward:', self.total_reward)

            if self.EPSILON > 0.05 and self.episode > 10000:
                self.EPSILON = math.pow(
                    self.EPSILON_decay_, self.episode - 10000)

            if self.episode % 500 == 0:
                logging.info("Epsilon: " + str(self.EPSILON))

            self.smoothed_reward.append(self.total_reward)
            self.total_reward = 0
            self.i_episode += 1
            loss_history = np.array(self.smoothed_reward.get())
            np.save(self.save_path+'_loss_history', loss_history)


app = Flask(__name__)
model = None

# @app.route("/initQLearning", methods = ["POST"])
# def init_q_learning():
#     global model
#     if model:
#         return 'Error! Model already exists!! Use /deleteModel request if you wish to delete it.',400

#     data = request.get_json()
#     logging.info("/initQLearning with data"+ str(data))

#     if not data:
#         return 'n_actions parameter not specified! Please provide the number of actions to the model!',400
#     n_actions = data.get('n_actions')
#     if n_actions is None:
#         return 'n_actions parameter not specified! Please provide the number of actions to the model!',400
#     n_actions = int(n_actions)
#     if n_actions < 2:
#         return 'The number of actions must be greater than two!',400
#     episode = int(data.get('episode',60000))
#     EPSILON = float(data.get('EPSILON',1))
#     GAMMA = float(data.get('GAMMA',0.95))
#     EPSILON_decay = float(data.get('EPSILON_decay',0.995))
#     learning_rate =float (data.get('learning_rate',0.01))
#     model = QLearning(state_size,n_actions,episode,EPSILON,GAMMA,EPSILON_decay,learning_rate)
#     return 'QLearning model initiated with parameters' +str( model.__dict__),200


@app.route("/initDQN", methods=["POST"])
def init_dqn():
    global model
    if model:
        return jsonify({'code': 400, 'msg': 'Model already exists!! Use /deleteModel request if you wish to delete it.'})

    data = request.get_json()
    logging.info("/initDQN with data" + str(data))

    if not data:
        return jsonify({'code': 400, 'msg': 'n_actions parameter not specified! Please provide the number of actions to the model!'})
    n_actions = data.get('n_actions')
    if n_actions is None:
        return jsonify({'code': 400, 'msg': 'n_actions parameter not specified! Please provide the number of actions to the model!'})
    n_actions = int(n_actions)
    episode = int(data.get('episode', 10000))
    batch_size = int(data.get('batch_size', 32))
    memory_size = int(data.get('memory_size', 1000))
    target_update = int(data.get('target_update', 10))
    EPSILON = float(data.get('EPSILON', 0.1))
    GAMMA = float(data.get('GAMMA', 0.95))
    learning_rate = float(data.get('learning_rate', 0.01))
    ddqn = bool(data.get('ddqn', False))
    model = DQN(n_actions, batch_size, memory_size, episode,
                target_update, EPSILON, GAMMA, learning_rate, ddqn)
    return jsonify({"code": 200, "msg": 'Dqn initiated successfully!', "data": str(model.__dict__)})


@app.route("/initPolicyGradient", methods=["POST"])
def initPolicyGradient():
    global model
    if model:
        return jsonify({'code': 400, 'msg': 'Model already exists!! Use /deleteModel request if you wish to delete it.'})

    data = request.get_json()
    logging.info("/initPolicyGradient with data" + str(data))

    if not data:
        return jsonify({'code': 400, 'msg': 'n_actions parameter not specified! Please provide the number of actions to the model!'})
    logging.info('initPolicyGradient', data)
    n_actions = data.get('n_actions')
    if n_actions is None:
        return jsonify({'code': 400, 'msg': 'n_actions parameter not specified! Please provide the number of actions to the model!'})
    n_actions = int(n_actions)
    episode = int(data.get('episode', 10000))
    GAMMA = float(data.get('GAMMA', 0.95))
    learning_rate = float(data.get('learning_rate', 0.01))
    model = PolicyGradient(n_actions, episode, GAMMA, learning_rate)
    return jsonify({"code": 200, "msg": 'PolicyGradient initiated successfully!', "data": str(model.__dict__)})


@app.route("/action", methods=["POST"])
def get_action():
    global model
    if not model:
        return jsonify({'code': 400, 'msg': 'Error! Model not initiate!! Use /initDQN or /initPolicyGradient to set a model'})
    data = request.get_json()
    logging.info('/action with data' + str(data['state']['payload']))

    if not data:
        return jsonify({'code': 400, 'msg': 'Error! State not provided!! Please add "state" argument'})
    state = data.get('state')
    if state is None:
        return jsonify({'code': 400, 'msg': 'Error! State not provided!! Please add "state" argument'})
    action = model.choose_action(get_state(state))
    return jsonify({"code": 200, "msg": 'Get action successfully!', "data": int(action)})


@app.route("/startTraining", methods=["POST"])
def start_training():
    global model
    if not model:
        return jsonify({'code': 400, 'msg': 'Error! Model not initiate!! Use /initDQN or /initPolicyGradient to set a model'})
    data = request.get_json()
    logging.info('/startTraining with data' + str(data))

    if not data:
        return jsonify({'code': 400, 'msg': 'Error! Initial state not provided!! Please add "initial_state" argument'})
    state = data.get('initial_state')

    if state is None:
        return jsonify({'code': 400, 'msg': 'Error! Initial state not provided!! Please add "initial_state" argument'})
    default_path = './checkpoints/' + \
        time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))+'/'
    save_path = data.get('save_path', default_path)
    model.start_training(get_state(state), save_path)
    return jsonify({"code": 200, "msg": 'Start training successfully!', "data": str(model.__dict__)})


@app.route("/trainStep", methods=["POST"])
def train_step():
    global model
    if not model:
        return jsonify({'code': 400, 'msg': 'Error! Model not initiate!! Use /initDQN or /initPolicyGradient to set a model'})
    if model.i_episode > model.episode:
        return jsonify({'code': 404, 'msg': 'Training Episodes finished! If you want to continue training,please use /addEpisode request!'})

    data = request.get_json()
    logging.info("/trainStep with data" + str(data))
    if not data:
        return jsonify({'code': 400, 'msg': 'Error! Arguments not provided!! Please add all the five arguments.'})
    state = data.get('state')
    if state is None:
        return jsonify({'code': 400, 'msg': 'Error! state not provided!! Please add "state" argument.'})
    state = get_state(state)
    action = data.get('action')
    if action is None:
        return jsonify({'code': 400, 'msg': 'Error! action not provided!! Please add "action" argument.'})
    next_state = data.get('next_state')
    if next_state is None:
        return jsonify({'code': 400, 'msg': 'Error! next_state not provided!! Please add "next_state" argument.'})
    next_state = get_state(next_state)
    reward = data.get('reward')
    if reward is None:
        return jsonify({'code': 400, 'msg': 'Error! reward not provided!! Please add "reward" argument.'})
    done = data.get('done')
    if done is None:
        return jsonify({'code': 400, 'msg': 'Error! done not provided!! Please add "done" argument.'})
    model.train(state, action, next_state, reward, done)
    return jsonify({"code": 200, "msg": "Train step successfully!"})


@app.route("/addEpisode", methods=["POST"])
def addEpisode():
    global model
    if not model:
        return jsonify({'code': 400, 'msg': 'Error! Model not initiate!! Use /initDQN or /initPolicyGradient to set a model'})
    data = request.get_json()
    logging.info("/addEpisode with data")
    logging.info(data)
    if data is None:
        return jsonify({'code': 400, 'msg': 'Error! eoisode not provided!! Please add "episode" argument.'})
    episode = data.get('episode')
    if episode is None:
        return jsonify({'code': 400, 'msg': 'Error! eoisode not provided!! Please add "episode" argument.'})
    model.episode += episode
    return jsonify({"code": 200, "msg": "Add episode successfully!"})


@app.route("/describeModel")
def describeModel():
    global model
    if not model:
        return jsonify({'code': 400, 'msg': 'Error! Model not initiate!! Use /initDQN or /initPolicyGradient to set a model'})
    else:
        return jsonify({"code": 200, "msg": 'Describe model successfully!', "data": str(model.__dict__)})


@app.route("/setParameter", methods=["POST"])
def setParameter():
    global model
    if not model:
        return jsonify({'code': 400, 'msg': 'Error! Model not initiate!! Use /initDQN or /initPolicyGradient to set a model'})
    else:
        data = request.get_json()
        logging.info("/setParameter with data")
        logging.info(data)
        if data is None:
            return jsonify({'code': 400, 'msg': 'Error! parameter not provided!! '})
        for key in data:
            if hasattr(model, key):
                setattr(model, key, data[key])
                logging.info("set parameter", key, ":", data[key])
        return jsonify({"code": 200, "msg": 'Set parameter successfully!', "data": str(model.__dict__)})


@app.route("/deleteModel")
def deleteModel():
    global model
    if model:
        del model
    model = None
    return jsonify({"code": 200, "msg": 'Model deleted successfully!'})


@app.route("/saveModel")
def saveModel(path):
    global model
    if not model:
        return jsonify({'code': 400, 'msg': 'Error! Model not initiate!! Use /initDQN or /initPolicyGradient to set a model'})
    model.save()
    return jsonify({"code": 200, "msg": 'Model saved successfully!', "data": str(model.__dict__)})


@app.route("/loadModel", methods=["POST"])
def load_model():
    global model
    if not model:
        return jsonify({'code': 400, 'msg': 'Model not initiate!! In order to load model, you need to initiate it first!'})
    data = request.get_json()
    logging.info("/loadModel with path")
    logging.info(data)

    if data is None:
        return jsonify({'code': 400, 'msg': 'Error! path not provided!! Please add "path" argument.'})
    path = data.get('path')
    if path is None:
        return jsonify({'code': 400, 'msg': 'Error! path not provided!! Please add "path" argument.'})

    if not os.path.exists(path+".dict"):
        return jsonify({'code': 400, 'msg': 'Model path:'+path+".dict" + "not found!"})
    if not os.path.exists(path+".index"):
        return jsonify({'code': 400, 'msg': 'Model path:'+path+".index" + "not found!"})

    if model.load(path):
        return jsonify({"code": 200, "msg": 'Model loaded successfully!', "data": str(model.__dict__)})
    else:
        return jsonify({'code': 400, 'msg': 'Loading model failed!'})


app.run()
