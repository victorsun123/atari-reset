import cv2
import argparse
from atari_reset.ppo import Model
from atari_reset.policies import GRUPolicy
import gym
from skimage.segmentation import mark_boundaries
from lime import lime_image
import horovod.tensorflow as hvd
import tensorflow as tf
from atari_reset.wrappers import VecFrameStack,  SubprocVecEnv,my_wrapper
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image
import pathlib
import pickle

# TO put together video run  `ffmpeg -f image2 -pattern_type glob -framerate 10 -i 'overlay_frame*.jpg' -s 160x224 results.mp4`


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type = str)
    parser.add_argument('--game', type=str, default='MontezumaRevenge')
    parser.add_argument('--num_timesteps', type=int, default=1e8)
    parser.add_argument('--policy', default='gru')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='results', help='Where to save results to')
    parser.add_argument("--noops", help="Use 0 to 30 random noops at the start of each episode", action="store_true")
    parser.add_argument("--sticky", help="Use sticky actions", action="store_true")
    parser.add_argument("--epsgreedy", help="Take random action with probability 0.01", action="store_true")
    parser.add_argument("--state", help="Whther to use state from episode or 0 initial state ", action="store_false")
    parser.add_argument("--grid", help="Whther to use grid segmentation or not ", action="store_true")
    parser.add_argument("--read_path", type=str, default = None)
    args = parser.parse_args()
    #test(args.game, args.num_timesteps, args.policy, args.load_path, args.save_path, args.noops, args.sticky, args.epsgreedy)


    # def make_env(rank):
    #     def env_fn():
    #         env = gym.make("MontezumaRevengeNoFrameskip-v4")
    #         env = bench.Monitor(env, "{}.monitor.json".format(rank))
    #         env = my_wrapper(env, clip_rewards=True)
    #         return env

    #     return env_fn

    #import ipdb; ipdb.set_trace()
    nenvs = 8
    env = gym.make("MontezumaRevengeNoFrameskip-v4")
    # env = my_wrapper(env, clip_rewards=True)
    # env = VecFrameStack(env, 4)

    class Box(gym.Space):
        """
        A box in R^n.
        I.e., each coordinate is bounded.
        Example usage:
        self.action_space = spaces.Box(low=-10, high=10, shape=(1,))
        """
        def __init__(self, low, high, shape=None, dtype=np.uint8):
            """
            Two kinds of valid input:
                Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided
                Box(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape
            """
            if shape is None:
                assert low.shape == high.shape
                self.low = low
                self.high = high
            else:
                assert np.isscalar(low) and np.isscalar(high)
                self.low = low + np.zeros(shape)
                self.high = high + np.zeros(shape)
            self.dtype = dtype
        def contains(self, x):
            return x.shape == self.shape and (x >= self.low).all() and (x <= self.high).all()
        def to_jsonable(self, sample_n):
            return np.array(sample_n).tolist()
        def from_jsonable(self, sample_n):
            return [np.asarray(sample) for sample in sample_n]
        @property
        def shape(self):
            return self.low.shape
        @property
        def size(self):
            return self.low.shape
        def __repr__(self):
            return "Box" + str(self.shape)
        def __eq__(self, other):
            return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)


    ob_space = Box(low=0, high=255, shape= (105, 80, 12), dtype = np.uint8)
    ac_space = env.action_space
    nsteps = 256
    nbatch = nenvs * nsteps
    nsteps_train = nsteps + nsteps // 2
    ent_coef=1e-4
    vf_coef=0.5
    l2_coef=1e-5
    cliprange=0.2

    print("building model")
    hvd.init()
    sess = tf.Session()

    model = Model(policy=GRUPolicy, ob_space=ob_space, ac_space=ac_space, nenv=nenvs, nsteps=nsteps_train, ent_coef=ent_coef,
                      vf_coef=vf_coef, l2_coef=l2_coef, cliprange=cliprange, load_path=args.load_path, test_mode=True, sess = sess)

    # mb_obs, mb_increase_ent, mb_rewards, mb_reward_avg, mb_actions, mb_values, mb_valids, mb_random_resets, mb_dones, mb_neglogpacs, mb_states
    #self.mb_stuff = [obs, [np.zeros(obs[0].shape[0], dtype=np.uint8)], [], [], [], [], [], [random_res], dones, [], states]

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


    def make_pred_fn(state):
        def predict_fn(images):

            batch_size = images.shape[0]
            obs = images[:,7:-7,:]
            obs_new = []
            for i in range(batch_size):
                pixelate = np.array(Image.fromarray(obs[i]).resize((80,105),resample=Image.BILINEAR), dtype=np.uint8)
                stack = np.repeat(pixelate,4, axis=-1)
                obs_new.append(stack)
            obs_new = np.array(obs_new)
            #print("SHAPE", obs.shape)
            done = np.array([False for _ in range(nenvs)])
            #print(done.shape)
            entropy = np.array([0 for _ in range(nenvs)])
            #print(entropy.shape)
            actions, values, states, neglogpacs, logits = model.step(obs_new, state, done, entropy)
            return [softmax(logits[i]) for i in range(nenvs)]
        return predict_fn

    if args.read_path:
        path = args.read_path
    else:
        path = "test/video0/"
    states = 0
    with (open("test/hidden_states/episode_1.pkl", "rb")) as openfile:
        states = pickle.load(openfile)

    states =np.array(states)

    def grid_segmentation(image):
        length = image.shape[0]
        width = image.shape[1]
        grid_length = length//3+1
        grid_width = width//3+1
        segmentation = np.zeros((length, width), dtype=int)
        for i in range(length):
            for j in range(width):
                segmentation[i][j] = int((i//grid_length)*3 + j//grid_width)
        return segmentation

    # def sorted_dir(folder):
    #     def getmtime(name):
    #         path = os.path.join(folder, name)
    #         return os.path.getmtime(path)

    #     return sorted(os.listdir(folder), key=getmtime)


    for i,file in enumerate(sorted(os.listdir(path))):
        f = os.path.join(path, file)
        image = cv2.imread(f)

        if args.state:
            predict_fn = make_pred_fn(states[i,:,:])
        else:
            predict_fn = make_pred_fn(model.initial_state)

        print("running lime on image")
        explainer = lime_image.LimeImageExplainer()
        if args.grid:
            explanation = explainer.explain_instance(image, predict_fn, top_labels=18,batch_size=8, hide_color=0, num_samples=1000, segmentation_fn = grid_segmentation)
        else:
            explanation = explainer.explain_instance(image, predict_fn, top_labels=18,batch_size=8, hide_color=0, num_samples=1000)
        img = [image] * 8
        best = np.argmax(predict_fn(np.array(img))[0])
        temp, mask = explanation.get_image_and_mask(best, positive_only=True, num_features=1, hide_rest=False)
        
        pathlib.Path(args.output_dir).mkdir(parents =True, exist_ok=True)
        plt.imsave(args.output_dir + "/mask_%s" % file, mask)
        plt.imsave(args.output_dir + "/grayscale_%s" % file, temp)
        plt.imsave(args.output_dir + "/overlay_%s" % file, mark_boundaries(temp, mask))








   
