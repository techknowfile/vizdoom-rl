from random import sample, randint, random
import random
import skimage.color, skimage.transform
import numpy as np


class ReplayMemory:
    def __init__(self, capacity, kframes, resolution, test_memory_size):
        self.resolution = resolution
        state_shape = (capacity, resolution[0], resolution[1] )
        test_state_shape = (test_memory_size, resolution[0], resolution[1])
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.test_buffer = np.zeros(test_state_shape, dtype=np.float32)
        self.test_buff_pos = 0
        self.test_size = 0

        self.capacity = capacity
        self.size = 0
        self.pos = 0

        self.kframes = kframes

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        samples_s1_container = []
        samples_action_container = []
        samples_s2_container = []
        samples_isTerminal_container = []
        samples_reward_container= []
        for i in sample(range(0, self.size), sample_size):#sample size is not kframes, but could be 32 or 64 (batch size)
            frame_indices = range(i-self.kframes+1, i+1)#+1 so that the last data point is considered too.
            #this will wrap around with negative numbers, so -2, -1,0,1,2 :-)

            #todo, this will be bad training. Need zero padding. mimic the test buffer

            s1_data = self.s1[frame_indices]
            action_data = self.a[i]
            s2_data = self.s2[frame_indices]
            isTerminal_data = self.isterminal[i]
            reward_data = self.r[i]

            #IF we have a terminal frame in the PRECEEDING k-frames then we DO NOT take those frames. Rather we
            # #repeat the frame after it
            terminal_offset = 0 #not possible case as you will see
            if True in self.isterminal[range(i-self.kframes+1, i)]: #dont care if the last one is terminal
                for k in range(1, self.kframes):#excludes the last index
                    if self.isterminal[i-k] : terminal_offset = k
            #--end outer if
            if terminal_offset != 0: #then we need to repeat some frames
                #0:-offset  = -offset repeated as many times
                num_repeats = self.kframes - terminal_offset
                tmp_reshaped = np.zeros([1] + list(s1_data[-terminal_offset].shape))  # add a leading dummy dimension
                repeat_frames = np.tile(tmp_reshaped, [num_repeats] + [1] * len(self.resolution))
                s1_data[0:-terminal_offset] = repeat_frames
                s2_data[0:-terminal_offset] = repeat_frames
                #however, the s2 preceeding frame would be s1, so that can be added
                s2_data[-terminal_offset-1] = s1_data[-terminal_offset]

            samples_s1_container.append(s1_data)
            samples_action_container.append(action_data)
            samples_s2_container.append(s2_data)
            samples_isTerminal_container.append(isTerminal_data)
            samples_reward_container.append(reward_data)

        return np.array(samples_s1_container),np.array(samples_action_container), \
               np.array(samples_s2_container),np.array(samples_isTerminal_container),np.array(samples_reward_container)

    def add_to_test_buffer(self, curr_state):
        self.test_buffer[self.test_buff_pos, :, :] = curr_state
        self.test_buff_pos = (self.test_buff_pos + 1) % self.capacity
        self.test_size = min(self.test_size + 1, self.capacity)

    def reset_test_buffer(self):
        self.test_buffer = np.zeros_like(self.test_buffer, dtype=np.float32)
        self.test_buff_pos = 0
        self.test_size = 0

    def get_test_sample(self):
        #get sample of size kframes, or prepend zeros.
        return_state_shape = (self.kframes, self.resolution[0], self.resolution[1])
        ret_buffer = None
        if self.test_size >= self.kframes:
            ret_buffer = self.test_buffer[self.test_size-self.kframes:self.test_size,:,:]
        else:#only fill what we have, and have preceeding zero frames (which was already done)
            ret_buffer = np.zeros(return_state_shape, dtype=np.float32)
            ret_buffer[self.kframes-self.test_size:,:,:] = self.test_buffer[:self.test_size, :, :]
            # num_repeats = kframes-self.test_size
            # tmp_reshaped = self.test_buffer[0].reshape([1]+list(self.test_buffer[0].shape))#add a leading dummy dimension
            # repeat_frames = np.tile(tmp_reshaped,[num_repeats]+[1]*len(resolution) )
            # ret_buffer[:kframes-self.test_size,:,:] = repeat_frames
        return ret_buffer


class DQN:
    def __init__(self, memory, model, game, actions, resolution, frame_repeat, batch_size, kframes, epochs, discount_factor):
        self.memory = memory
        self.model = model
        self.game = game
        self.actions = actions
        self.resolution = resolution
        self.frame_repeat = frame_repeat
        self.batch_size = batch_size
        self.kframes = kframes
        self.epochs = epochs
        self.discount_factor = discount_factor

    def preprocess(self, img):
        img = skimage.transform.resize(img, [self.resolution[0], self.resolution[1]])
        img = img.astype(np.float32)
        return img

    def learn_from_memory(self, model):
        """ Use replay memory to learn. Ignore s2 if s1 is terminal """

        if self.memory.size > self.batch_size:
            s1, a, s2, isterminal, r = self.memory.get_sample(self.batch_size)

            q = model.predict(s2, batch_size=self.batch_size)
            q2 = np.max(q, axis=1)
            target_q = model.predict(s1, batch_size=self.batch_size)
            target_q[np.arange(target_q.shape[0]), a] = r + self.discount_factor * (1 - isterminal) * q2
            model.fit(s1, target_q, verbose=0)

    def get_best_action(self, state):
        q = self.model.predict(state, batch_size=1)
        m = np.argmax(q, axis=1)[0]
        action = m  # wrong
        return action

    def perform_learning_step(self, epoch):
        """ Makes an action according to eps-greedy policy, observes the result
        (next state, reward) and learns from the transition"""

        def exploration_rate(epoch, decay = 'log'):
            """# Define exploration rate change over time"""

            # return 0.1
            if(decay == 'linear'): 
                start_eps = 1.0
                end_eps = 0.1
                const_eps_epochs = 0.1 * self.epochs  # 10% of learning time
                eps_decay_epochs = 0.8 * self.epochs  # 60% of learning time

                if epoch < const_eps_epochs:
                    return start_eps
                elif epoch < eps_decay_epochs:
                    # Linear decay
                    return start_eps - (epoch - const_eps_epochs) / \
                           (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
                else:
                    return end_eps
            elif(decay == 'log'):
                steepness=7
                start_eps = 5.0
                end_eps = 0.1
                const_eps_epochs = 0.1 * self.epochs  # 10% of learning time
                eps_decay_epochs = 0.8 * self.epochs  # 60% of learning time

                if epoch < const_eps_epochs:
                    return start_eps
                elif epoch < eps_decay_epochs:
                    # Logistic decay
                    return start_eps - np.arctan(steepness*(epoch - const_eps_epochs) / \
                           (eps_decay_epochs - const_eps_epochs)) / np.arctan(steepness) * (start_eps - end_eps)
                else:
                    return end_eps
        s1 = self.preprocess(self.game.get_state().screen_buffer)

        # With probability eps make a random action.
        eps = exploration_rate(epoch)
        if random.random() <= eps:
            a = randint(0, len(self.actions) - 1)
        else:
            # Choose the best action according to the network.
            self.memory.add_to_test_buffer(s1)
            state_kframes = self.memory.get_test_sample()
            state_kframes = state_kframes.reshape([1, self.kframes, self.resolution[0], self.resolution[1]])
            a = self.get_best_action(state_kframes)
        reward = self.game.make_action(self.actions[a], self.frame_repeat)

        isterminal = self.game.is_episode_finished()
        #todo RESET the test buffer ELSEWHERE. not here

        s2 = self.preprocess(self.game.get_state().screen_buffer) if not isterminal else None

        # Remember the transition that was just experienced.
        self.memory.add_transition(s1, a, s2, isterminal, reward)

        self.learn_from_memory(self.model)




