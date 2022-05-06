import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import yaml
import os
from trainers.global_trainer import GlobalTrainer
from env.dynamic_envinronment import Environment
import gym

from tf2rl.algos.dqn import DQN
from tf2rl.experiments.trainer import Trainer

from global_config import ROOT_DIR


from trainers.conditional_trainer import ConditionalTrainer
from tf2rl.envs.utils import is_atari_env
from tf2rl.envs.atari_wrapper import wrap_dqn
from tf2rl.networks.atari_model import AtariQFunc, AtariCategoricalActor
from env.dynamic_envinronment import Environment

def train_agent_with_env(config):
    env = Environment(config=config, TRAIN_MODE=False)
    trainer = GlobalTrainer(config=config, env=env, policy=None)
    trainer()


if __name__ == '__main__':
    # get configuration
    with open(os.path.join(ROOT_DIR,"config.yaml"), "r") as stream:
        try:
            config = yaml.safe_load(stream)
            # print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

    # Generate / Load Dataset
    if config["generate_dataset"] is True:
        from env.data_generators.scenario_02 import Scenario
        my_scene = Scenario(config)
        my_scene.generate_data()


    env = Environment(config=config, TRAIN_MODE=True)
    test_env = Environment(config=config, TRAIN_MODE=False)



    for num_trial in range(config["algorithm_config"]["trial"]):
        print("==================================================")
        print("==================================================")
        print("==================================================")
        print("=============trial "+str(num_trial) +"=================")
        print("==================================================")
        print("==================================================")



        # Agent Setting
        # Codes are copied from Kei-Ota's project
        parser = ConditionalTrainer.get_argument()
        parser = DQN.get_argument(parser)
        parser.set_defaults(test_interval=100)
        parser.set_defaults(max_steps=20000)
        parser.set_defaults(gpu=1)
        parser.set_defaults(n_warmup=64)
        parser.set_defaults(batch_size=32)
        parser.set_defaults(memory_capacity=int(1e4))
        parser.add_argument('--env-name', type=str,
                            default="SpaceInvadersNoFrameskip-v4")
        args = parser.parse_args()

        # env = gym.make(args.env_name)
        # test_env = gym.make(args.env_name)

        policy = DQN(
            enable_double_dqn=args.enable_double_dqn,
            enable_dueling_dqn=args.enable_dueling_dqn,
            enable_noisy_dqn=args.enable_noisy_dqn,
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.n,
            target_replace_interval=10,
            discount=0.995,
                lr=0.0001,
            gpu=args.gpu,
            memory_capacity=args.memory_capacity,
            batch_size=args.batch_size,
            n_warmup=args.n_warmup)
        trainer = ConditionalTrainer(policy, env, args, test_env=test_env)
        if args.evaluate:
            trainer.evaluate_policy_continuously()
        else:
            trainer()

