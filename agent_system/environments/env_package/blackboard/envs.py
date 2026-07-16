import ray
import gym
import numpy as np
from agent_system.environments.env_package.blackboard.blackboard import BlackboardEnv

@ray.remote(num_cpus=0.2)
class BlackboardWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds its own independent instance of BlackboardEnv.
    """
    def __init__(self, config):
        self.env = BlackboardEnv(config)

    def reset(self, seed_for_reset, current_question):
        """Reset the environment with optional seed"""
        if seed_for_reset is not None:
            obs, info = self.env.reset(seed=seed_for_reset, question=current_question)
        else:
            obs, info = self.env.reset(question=current_question)
        return obs, info

    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info


class BlackboardMultiProcessEnv(gym.Env):
    """
    Ray-based wrapper for the Blackboard environment.
    Each Ray actor creates an independent BlackboardEnv instance.
    The main process communicates with Ray actors to collect step/reset results.
    """

    def __init__(self,
                 seed: int,
                 env_num: int,
                 group_n: int,
                 #resources_per_worker: dict,
                 config: dict,
                 is_train: bool = True,
                 env_kwargs: dict = {}
                 ):
        """
        :param seed: Random seed for reproducibility.
        :param env_num: Number of parallel environments to create.
        :param group_n: Number of agents in each environment.
        :param is_train: Whether the environment is in training mode.
        :param config: Configuration dictionary for the environment.
        """
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.is_train = is_train
        self.config = config
        np.random.seed(seed)

        train_tasks = config.get("train_tasks", [])
        val_tasks = config.get("val_tasks", [])
        if is_train:
            #in training mode, use train tasks
            tasks = train_tasks
        else:
            #in evaluation mode, use val tasks
            tasks = val_tasks
        self.config["tasks"] = tasks
        #环境初始化需要的参数先写在这里，包括最大论述，可行动作等
        self.config["max_steps"] = 8
        self.config["available_agents"] = ["task_decompose", "summarize", "critique", "terminate", "arbitrate", "reason", "direct_answer", "expert", "clean", "question", "debate", "expert2", "verify", "explore", "counterfactual_think", "web_search", 'knowledge_retrieve', 'python_calculate']
        self.config['seed'] = seed

        # -------------------------- Ray actors setup --------------------------
        #print('prosecc_num', self.num_processes)
        # Initialize Ray actors
        self._workers = []
        for i in range(self.num_processes):
            worker = BlackboardWorker.remote(self.config)
            self._workers.append(worker)
        
    def step(self, actions):
        """
        Perform step in parallel.
        :param actions: list, length must equal self.num_processes.
        :return: obs_list, reward_list, done_list, info_list
        """

        assert len(actions) == self.num_processes

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Wait for all workers to finish and gather results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list
    
    def reset(self, questions: list = None):
        """
        Perform reset in parallel.
        :return: obs_list and info_list, the initial observations for each environment
        """
        # randomly generate self.env_num seeds
        if self.is_train:
            seeds = np.random.randint(0, 2**16 - 1, size=self.env_num)
        else:
            seeds = np.random.randint(2**16, 2**32 - 1, size=self.env_num)

        # repeat the seeds for each group
        seeds = np.repeat(seeds, self.group_n)
        seeds = seeds.tolist()

        # Send reset commands to all workers
        futures = []
        #print('workers数量:',len(self._workers))
        for i, worker in enumerate(self._workers):
            #print('let me seesee',type(questions[i]))
            future = worker.reset.remote(seeds[i], questions[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list
    
    def close(self):
        """
        Close all Ray actors
        """
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)

    def __del__(self):
        self.close()


def build_blackboard_envs(seed: int,
                env_num: int,
                group_n: int,
                #resources_per_worker: dict,
                config: dict,
                is_train: bool = True,
                env_kwargs: dict = {}
                ):
    return BlackboardMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        #resources_per_worker=resources_per_worker,
        config=config,
        is_train=is_train,
        env_kwargs=env_kwargs
    )