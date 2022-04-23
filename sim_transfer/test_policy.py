import os
import time
from multiprocessing import Process, Queue

import imageio
from spinup.utils.logx import EpochLogger
from spinup.utils.test_policy import load_policy_and_env

from envs.walker2d_custom import Walker2dEnv


# model = torch.load('models/walker2d(2-0)_base/model.pt')
# def get_action(x):
#     with torch.no_grad():
#         x = torch.as_tensor(x, dtype=torch.float32)
#         action = model.act(x)
#     return action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True, **output_args):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    video_process = None
    video_queue = None
    if 'video_path' in output_args:
        video_path = output_args.pop('video_path')
        fps = round(1.0 / env.dt)

        from mujoco_py.mjviewer import save_video
        video_queue = Queue()
        video_process = Process(target=save_video, args=(video_queue, video_path, fps))
        video_process.start()

    logger = EpochLogger(**output_args)
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)
        if video_process is not None:
            frame = env.render('rgb_array', camera_name='track')
            video_queue.put(frame)

        a = get_action(o)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    if video_process is not None:
        video_queue.put(None)
        video_process.join()

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()



def run_exp(train_env, test_env, transfer, render=False, max_ep_len=None, num_episodes=100, **output_args):
    assert not (transfer and train_env == 'walker2d')

    ## Load the trained model
    exp_name = train_env
    if transfer:
        exp_name = exp_name.replace('walker2d', 'transfer')
    if exp_name == 'walker2d':
        exp_name = exp_name + '_base'

    _, get_action = load_policy_and_env(f'/home/iyevenko/Documents/spinningup/data/{exp_name}/{exp_name}_s0')

    ## Load the test env
    xml_name = test_env + '.xml'
    try:
        env = Walker2dEnv(xml_name)
    except:
        xml_name = 'low_friction/'+xml_name
        env = Walker2dEnv(xml_name)


    run_policy(env, get_action, render=render, max_ep_len=max_ep_len, num_episodes=num_episodes, **output_args)
    print('Finished running experiment:', exp_name)
    print('Test environment XML:', xml_name)


def transfer_test():
    output_dir = 'results/transfer_test'
    for test_env in envs[1:4]:
        # Zero Shot Test
        train_env = envs[0]
        transfer = False
        output_fname = test_env.replace('walker2d', 'zeroshot')
        run_exp(train_env, test_env, transfer, output_dir=output_dir, output_fname=output_fname)

        # Transfer Test
        train_env = test_env
        transfer = True
        output_fname = test_env.replace('walker2d', 'transfer')
        run_exp(train_env, test_env, transfer, output_dir=output_dir, output_fname=output_fname)

        # Scratch Test
        train_env = test_env
        transfer = False
        output_fname = test_env.replace('walker2d', 'scratch')
        run_exp(train_env, test_env, transfer, output_dir=output_dir, output_fname=output_fname)

def friction_test():
    output_dir = 'results/friction_test'
    for test_env in envs[4:]:
        # Zero Shot Test
        train_env = envs[0]
        transfer = False
        output_fname = test_env.replace('walker2d', 'zeroshot')
        run_exp(train_env, test_env, transfer, max_ep_len=10000, output_dir=output_dir, output_fname=output_fname)

        # Transfer Test
        train_env = test_env
        transfer = True
        output_fname = test_env.replace('walker2d', 'transfer')
        run_exp(train_env, test_env, transfer, max_ep_len=10000, output_dir=output_dir, output_fname=output_fname)


def save_videos(video_dir, num_episodes=1):
    def make_dirs(dir):
        os.makedirs(dir, exist_ok=True)
        return dir

    # Base Environment
    test_env = envs[0]
    env_dir = make_dirs(os.path.join(video_dir, test_env+'_base'))

    # Scratch
    train_env = test_env
    transfer = False
    video_path = os.path.join(env_dir, 'scratch.mp4')
    run_exp(train_env, test_env, transfer=transfer, video_path=video_path, num_episodes=num_episodes)

    # Transfer Environments
    for test_env in envs[1:4]:
        env_dir = make_dirs(os.path.join(video_dir, test_env))

        # Zero Shot
        train_env = envs[0]
        transfer = False
        video_path = os.path.join(env_dir, 'zeroshot.mp4')
        run_exp(train_env, test_env, transfer=transfer, video_path=video_path, num_episodes=num_episodes)

        # Transfer
        train_env = test_env
        transfer = True
        video_path = os.path.join(env_dir, 'transfer.mp4')
        run_exp(train_env, test_env, transfer=transfer, video_path=video_path, num_episodes=num_episodes)

        # Scratch
        train_env = test_env
        transfer = False
        video_path = os.path.join(env_dir, 'scratch.mp4')
        run_exp(train_env, test_env, transfer=transfer, video_path=video_path, num_episodes=num_episodes)

    # Low Friction Environments
    for test_env in envs[4:]:
        env_dir = make_dirs(os.path.join(video_dir, test_env))

        # Zero Shot
        train_env = envs[0]
        transfer = False
        video_path = os.path.join(env_dir, 'zeroshot.mp4')
        run_exp(train_env, test_env, transfer=transfer, video_path=video_path, num_episodes=num_episodes)

        # Transfer
        train_env = test_env
        transfer = True
        video_path = os.path.join(env_dir, 'transfer.mp4')
        run_exp(train_env, test_env, transfer=transfer, video_path=video_path, num_episodes=num_episodes)


envs = [
    "walker2d",
    "walker2d_low_friction",
    "walker2d_short_joints",
    "walker2d_long_joints",
    'walker2d_friction_80',
    'walker2d_friction_60',
    'walker2d_friction_40',
    'walker2d_friction_20'
]

if __name__ == '__main__':
    # train_env = envs[0]
    # test_env = envs[0]
    # transfer = False
    # run_exp(train_env, test_env, transfer, render=False, video_path='video.mp4')

    # model = torch.load('models/walker2d_base/model.pt')
    # model = add_layers(model, 2, 2, hidden_size=32)
    # for p in model.parameters():
    #     print(p.shape, p.requires_grad)
    #
    # def get_action(x):
    #     with torch.no_grad():
    #         x = torch.as_tensor(x, dtype=torch.float32)
    #         action = model.act(x)
    #     return action
    #
    # xml_name = 'low_friction/walker2d_friction_40.xml'
    # env = Walker2dEnv(xml_name)
    # run_policy(env, get_action, render=True)

    # transfer_test()
    # friction_test()
    save_videos('videos', num_episodes=3)