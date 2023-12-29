# -*- coding: utf-8 -*-

!python --version

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# 
# !pip install setuptools==65.5.0
# 
# !pip install CityLearn==1.8.0
# 
# !pip install ipywidgets==7.7.2
# 
# !pip install matplotlib==3.5.3
# !pip install seaborn==0.12.2
# 
# !pip install stable-baselines3

import inspect
import os
import uuid

from datetime import datetime

from typing import List, Mapping, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from IPython.display import clear_output
from ipywidgets import Button, FloatSlider, HBox, HTML
from ipywidgets import IntProgress, Text, VBox

import math
import numpy as np
import pandas as pd
import random
import re
import requests
import simplejson as json

from citylearn.agents.rbc import HourRBC
from citylearn.agents.q_learning import TabularQLearning
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.reward_function import RewardFunction
from citylearn.wrappers import NormalizedObservationWrapper
from citylearn.wrappers import StableBaselines3Wrapper
from citylearn.wrappers import TabularQLearningWrapper

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

"""# Data"""

DATASET_NAME = 'citylearn_challenge_2022_phase_all'
schema = DataSet.get_schema(DATASET_NAME)

schema

def set_schema_buildings(
schema: dict, count: int, seed: int
) -> Tuple[dict, List[str]]:
    """Randomly select number of buildings to set as active in the schema.
    """

    assert 1 <= count <= 15, 'count must be between 1 and 15.'

    # set random seed
    np.random.seed(seed)

    buildings = list(schema['buildings'].keys())
    # remove buildins 12 and 15 as they have pecularities in their data
    buildings_to_exclude = ['Building_12', 'Building_15']
    for b in buildings_to_exclude:
        buildings.remove(b)

    # randomly select specified number of buildings
    buildings = np.random.choice(buildings, size=count, replace=False).tolist()

    # reorder buildings
    building_ids = [int(b.split('_')[-1]) for b in buildings]
    building_ids = sorted(building_ids)
    buildings = [f'Building_{i}' for i in building_ids]

    # update schema
    for b in schema['buildings']:
        if b in buildings:
            schema['buildings'][b]['include'] = True
        else:
            schema['buildings'][b]['include'] = False

    return schema, buildings

root_directory = schema['root_directory']
building_name = 'Building_1'

def set_schema_simulation_period(
    schema: dict, count: int, seed: int
) -> Tuple[dict, int, int]:
    """Randomly select environment simulation start and end time steps
    that cover a specified number of days.
    """

    assert 1 <= count <= 365, 'count must be between 1 and 365.'

    # set random seed
    np.random.seed(seed)

    # use random dataset to get number of timesteps
    filename = schema['buildings'][building_name]['carbon_intensity']
    filepath = os.path.join(root_directory, filename)
    time_steps = pd.read_csv(filepath).shape[0]

    # set candidate simulation start time steps
    # spaced by the number of specified days
    simulation_start_time_step_list = np.arange(0, time_steps, 24*count)

    # randomly select a simulation start time step
    simulation_start_time_step = np.random.choice(
        simulation_start_time_step_list, size=1
    )[0]
    simulation_end_time_step = simulation_start_time_step + 24*count - 1

    # update schema simulation time steps
    schema['simulation_start_time_step'] = simulation_start_time_step
    schema['simulation_end_time_step'] = simulation_end_time_step

    return schema, simulation_start_time_step, simulation_end_time_step

def set_active_observations(
    schema: dict, active_observations: List[str]
) -> dict:
    """Set the observations that will be part of the environment's
    observation space that is provided to the control agent.
    """

    active_count = 0

    for o in schema['observations']:
        if o in active_observations:
            schema['observations'][o]['active'] = True
            active_count += 1
        else:
            schema['observations'][o]['active'] = False

    valid_observations = list(schema['observations'].keys())
    assert active_count == len(active_observations),\
        'the provided observations are not all valid observations.'\
          f' Valid observations in CityLearn are: {valid_observations}'

    return schema

RANDOM_SEED = 238
BUILDING_COUNT = 2
DAY_COUNT = 7
ACTIVE_OBSERVATIONS = ['hour']

schema, buildings = set_schema_buildings(schema, BUILDING_COUNT, RANDOM_SEED)
schema, simulation_start_time_step, simulation_end_time_step =\
    set_schema_simulation_period(schema, DAY_COUNT, RANDOM_SEED)
schema = set_active_observations(schema, ACTIVE_OBSERVATIONS)
schema['central_agent'] = True

print('Selected buildings:', buildings)
print(
    f'Selected {DAY_COUNT}-day period time steps:',
    (simulation_start_time_step, simulation_end_time_step)
)
print(f'Active observations:', ACTIVE_OBSERVATIONS)

"""# Initializing the env"""

env = CityLearnEnv(schema)

print('Current time step:', env.time_step)
print('environment number of time steps:', env.time_steps)
print('environment uses central agent:', env.central_agent)
print('Common (shared) observations amongst buildings:', env.shared_observations)
print('Number of buildings:', len(env.buildings))

"""# Performance Indicators"""

def get_kpis(env: CityLearnEnv) -> pd.DataFrame:
    """Returns evaluation KPIs.

    Electricity consumption, cost and carbon emissions KPIs are provided
    at the building-level and average district-level. Average daily peak,
    ramping and (1 - load factor) KPIs are provided at the district level.
    """

    kpis = env.evaluate()
    # names of KPIs to retrieve from evaluate function
    kpi_names = [
        'electricity_consumption', 'cost', 'carbon_emissions'
    ]
    kpis = kpis[
        (kpis['cost_function'].isin(kpi_names))
    ].dropna()

    # round up the values to 3 decimal places for readability
    kpis['value'] = kpis['value'].round(3)

    # rename the column that defines the KPIs
    kpis = kpis.rename(columns={'cost_function': 'kpi'})

    return kpis

def plot_building_kpis(envs: Mapping[str, CityLearnEnv]) -> plt.Figure:
    """Plots electricity consumption, cost and carbon emissions
    at the building-level for different control agents in bar charts.
    """

    kpis_list = []

    for k, v in envs.items():
        kpis = get_kpis(v)
        kpis = kpis[kpis['level']=='building'].copy()
        kpis['building_id'] = kpis['name'].str.split('_', expand=True)[1]
        kpis['building_id'] = kpis['building_id'].astype(int).astype(str)
        kpis['env_id'] = k
        kpis_list.append(kpis)

    kpis = pd.concat(kpis_list, ignore_index=True, sort=False)
    kpi_names= kpis['kpi'].unique()
    column_count_limit = 3
    row_count = math.ceil(len(kpi_names)/column_count_limit)
    column_count = min(column_count_limit, len(kpi_names))
    building_count = len(kpis['name'].unique())
    env_count = len(envs)
    figsize = (3.0*column_count, 0.3*env_count*building_count*row_count)
    fig, _ = plt.subplots(
        row_count, column_count, figsize=figsize, sharey=True
    )

    for i, (ax, (k, k_data)) in enumerate(zip(fig.axes, kpis.groupby('kpi'))):
        sns.barplot(x='value', y='name', data=k_data, hue='env_id', ax=ax)
        ax.axvline(1.0, color='black', linestyle='--', label='Baseline')
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_title(k)

        if i == len(kpi_names) - 1:
            ax.legend(
                loc='upper left', bbox_to_anchor=(1.3, 1.0), framealpha=0.0
            )
        else:
            ax.legend().set_visible(False)

        for s in ['right','top']:
            ax.spines[s].set_visible(False)

        for p in ax.patches:
            ax.text(
                p.get_x() + p.get_width(),
                p.get_y() + p.get_height()/2.0,
                p.get_width(), ha='left', va='center'
            )

    plt.tight_layout()
    return fig

def plot_district_kpis(envs: Mapping[str, CityLearnEnv]) -> plt.Figure:
    """Plots electricity consumption, cost, carbon emissions,
    average daily peak, ramping and (1 - load factor) at the
    district-level for different control agents in a bar chart.
    """

    kpis_list = []

    for k, v in envs.items():
        kpis = get_kpis(v)
        kpis = kpis[kpis['level']=='district'].copy()
        kpis['env_id'] = k
        kpis_list.append(kpis)

    kpis = pd.concat(kpis_list, ignore_index=True, sort=False)
    row_count = 1
    column_count = 1
    env_count = len(envs)
    kpi_count = len(kpis['kpi'].unique())
    figsize = (3.0*column_count, 0.225*env_count*kpi_count*row_count)
    fig, ax = plt.subplots(row_count, column_count, figsize=figsize)
    sns.barplot(x='value', y='kpi', data=kpis, hue='env_id', ax=ax)
    ax.axvline(1.0, color='black', linestyle='--', label='Baseline')
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    for s in ['right','top']:
        ax.spines[s].set_visible(False)

    for p in ax.patches:
        ax.text(
            p.get_x() + p.get_width(),
            p.get_y() + p.get_height()/2.0,
            p.get_width(), ha='left', va='center'
        )

    ax.legend(loc='upper left', bbox_to_anchor=(1.3, 1.0), framealpha=0.0)
    plt.tight_layout()

    return fig

def plot_building_load_profiles(envs: Mapping[str, CityLearnEnv]) -> plt.Figure:
    """Plots building-level net electricity consumption profile
    for different control agents.
    """

    building_count = len(list(envs.values())[0].buildings)
    column_count_limit = 4
    row_count = math.ceil(building_count/column_count_limit)
    column_count = min(column_count_limit, building_count)
    figsize = (4.0*column_count, 1.75*row_count)
    fig, _ = plt.subplots(row_count, column_count, figsize=figsize)

    for i, ax in enumerate(fig.axes):
        for k, v in envs.items():
            y = v.buildings[i].net_electricity_consumption
            x = range(len(y))
            ax.plot(x, y, label=k)

        y = v.buildings[i].net_electricity_consumption_without_storage
        ax.plot(x, y, label='Baseline')
        ax.set_title(v.buildings[i].name)
        ax.set_xlabel('Time step')
        ax.set_ylabel('kWh')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(24))

        if i == building_count - 1:
            ax.legend(
                loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0
            )
        else:
            ax.legend().set_visible(False)


    plt.tight_layout()

    return fig

def plot_district_load_profiles(envs: Mapping[str, CityLearnEnv]) -> plt.Figure:
    """Plots district-level net electricty consumption profile
    for different control agents.
    """

    figsize = (5.0, 1.5)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for k, v in envs.items():
        y = v.net_electricity_consumption
        x = range(len(y))
        ax.plot(x, y, label=k)

    y = v.net_electricity_consumption_without_storage
    ax.plot(x, y, label='Baseline')
    ax.set_xlabel('Time step')
    ax.set_ylabel('kWh')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0)

    plt.tight_layout()
    return fig

def plot_battery_soc_profiles(envs: Mapping[str, CityLearnEnv]) -> plt.Figure:
    """Plots building-level battery SoC profiles for different control agents.
    """

    building_count = len(list(envs.values())[0].buildings)
    column_count_limit = 4
    row_count = math.ceil(building_count/column_count_limit)
    column_count = min(column_count_limit, building_count)
    figsize = (4.0*column_count, 1.75*row_count)
    fig, _ = plt.subplots(row_count, column_count, figsize=figsize)

    for i, ax in enumerate(fig.axes):
        for k, v in envs.items():
            soc = np.array(v.buildings[i].electrical_storage.soc)
            capacity = v.buildings[i].electrical_storage.capacity_history[0]
            y = soc/capacity
            x = range(len(y))
            ax.plot(x, y, label=k)

        ax.set_title(v.buildings[i].name)
        ax.set_xlabel('Time step')
        ax.set_ylabel('SoC')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(24))

        if i == building_count - 1:
            ax.legend(
                loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0
            )
        else:
            ax.legend().set_visible(False)


    plt.tight_layout()

    return fig

def plot_simulation_summary(envs: Mapping[str, CityLearnEnv]):
    """Plots KPIs, load and battery SoC profiles for different control agents.
    """

    _ = plot_building_kpis(envs)
    print('Building-level KPIs:')
    plt.show()
    _ = plot_building_load_profiles(envs)
    print('Building-level load profiles:')
    plt.show()
    _ = plot_battery_soc_profiles(envs)
    print('Battery SoC profiles:')
    plt.show()
    _ = plot_district_kpis(envs)
    print('District-level KPIs:')
    plt.show()
    print('District-level load profiles:')
    _ = plot_district_load_profiles(envs)
    plt.show()

"""# Tabular Q-learning Agent - default reward

### The default reward is the electricity consumption from the grid at the current time step returned as a negative value.
"""

help(env.reward_function)

# define active observations and actions and their bin sizes
observation_bins = {'hour': 24}
action_bins = {'electrical_storage': 12}

# initialize list of bin sizes where each building has a dictionary in the list definining its bin sizes
observation_bin_sizes = []
action_bin_sizes = []

for b in env.buildings:
    # add a bin size definition for the buildings
    observation_bin_sizes.append(observation_bins)
    action_bin_sizes.append(action_bins)

# wrapper to discretize the action and observation space
env = TabularQLearningWrapper(
    env.unwrapped,
    observation_bin_sizes=observation_bin_sizes,
    action_bin_sizes=action_bin_sizes
)

class CustomTabularQLearning(TabularQLearning):
    def __init__(
        self, env: CityLearnEnv, loader: IntProgress,
        random_seed: int = None, **kwargs
    ):
        r"""Initialize CustomTQL.
        """

        super().__init__(env=env, random_seed=random_seed, **kwargs)
        self.loader = loader
        self.reward_history = []

    def next_time_step(self):
        if self.env.time_step == 0:
            self.reward_history.append(0)

        else:
            self.reward_history[-1] += sum(self.env.rewards[-1])

        self.loader.value += 1
        super().next_time_step()

def get_loader(**kwargs):
    """Returns a progress bar"""

    kwargs = {
        'value': 0,
        'min': 0,
        'max': 10,
        'description': 'Simulating:',
        'bar_style': '',
        'style': {'bar_color': 'maroon'},
        'orientation': 'horizontal',
        **kwargs
    }
    return IntProgress(**kwargs)

# ----------------- CALCULATE NUMBER OF TRAINING EPISODES -----------------
# factor used to scale the # of training episodes for TQL agent
# -> higher i = more exploration, increased training duration
i = 10
m = env.observation_space[0].n
n = env.action_space[0].n
t = env.time_steps - 1
tql_episodes = m*n*i/t
tql_episodes = int(tql_episodes) # 206 episodes
print('Q-Table dimension:', (m, n))
print('Number of episodes to train:', tql_episodes)

# ------------------------------- SET LOADER ------------------------------
loader = get_loader(max=tql_episodes*t)
display(loader)

# ----------------------- SET MODEL HYPERPARAMETERS -----------------------
tql_kwargs = {
    'epsilon': 1.0, # encourage exploration in the beginning
    'minimum_epsilon': 0.01,
    'epsilon_decay': 0.0096, # ε reaches ε_min after half the total number of training episodes -> (1.0 - 0.01) / (tql_episodes / 2)
    'learning_rate': 0.0001, # smaller = more stable but slower learning
    'discount_factor': 0.98, # γ = 0.9 gives reasonable importance to rewards within the next ten steps, γ = 0.99 gives importance to rewards within the next hundred steps.
}

# ----------------------- INITIALIZE AND TRAIN MODEL ----------------------
print('Selected buildings:', buildings)

# select days
schema, simulation_start_time_step, simulation_end_time_step =\
    set_schema_simulation_period(schema, DAY_COUNT, RANDOM_SEED)
print(
    f'Selected {DAY_COUNT}-day period time steps:',
    (simulation_start_time_step, simulation_end_time_step)
    )

baseline_model = CustomTabularQLearning(
    env=env,
    loader=loader,
    random_seed=RANDOM_SEED,
    **tql_kwargs
)
_ = baseline_model.learn(episodes=tql_episodes)

# ----------------------- OUPUT RESULTS/STATISTICS ------------------------
observations = env.reset()

while not env.done:
    actions = baseline_model.predict(observations, deterministic=True)
    observations, _, _, _ = env.step(actions)

# plot summary and compare with other control results
plot_simulation_summary({'TQL': env})

def plot_table(
    ax: plt.Axes, table: np.ndarray, title: str,
    colorbar_label: str, xlabel: str, ylabel: str
) -> plt.Axes:
    """Plot 2-dimensional table on a heat map.
    """

    x = list(range(table.shape[0]))
    y = list(range(table.shape[1]))
    z = table.T
    pcm = ax.pcolormesh(
        x, y, z, shading='nearest', cmap=cmap,
        edgecolors='black', linewidth=0.0
    )
    _ = fig.colorbar(
        pcm, ax=ax, orientation='horizontal',
        label=colorbar_label, fraction=0.025, pad=0.08
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return ax

cmap = 'coolwarm'
figsize = (12, 8)
fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)
axs[0] = plot_table(
    axs[0], baseline_model.q[0], 'Q-Table',
    'Q-Value', 'State (Hour)', 'Action Index'
)
axs[1] = plot_table(
    axs[1], baseline_model.q_exploration[0], 'Q-Table Exploration',
    'Count','State (Hour)', None
)
axs[2] = plot_table(
    axs[2], baseline_model.q_exploitation[0], 'Q-Table Exploitation',
    'Count', 'State (Hour)', None
)

plt.tight_layout()
plt.show()

print(
    f'Current Tabular Q-Learning epsilon after {tql_episodes}'\
        f' episodes and {baseline_model.time_step} time steps:', baseline_model.epsilon
)

"""# Tuning the model - minimizing Cost (C)

### p_i = -(1 + sign(C_i) * SOC_i)
### Reward_i = sum(p_i * |C_i|)
"""

for b in env.buildings:
  cost = np.array(b.net_electricity_consumption_cost)
  print(cost)

# -------------------- CUSTOMIZE ENVIRONMENT -----------------------------
active_observations = [
    'hour'
]

# --------------- DEFINE CUSTOM REWARD FUNCTION -----------------
class CostMinimizing(RewardFunction):
    def __init__(self, env: CityLearnEnv):
        r"""Initialize CostMinimizing.

        Parameters
        ----------
        env: Mapping[str, CityLearnEnv]
            CityLearn environment instance.
        """

        super().__init__(env)

    def calculate(self) -> List[float]:
        r"""Returns reward for most recent action.

        The reward is designed to minimize electricity cost.
        It is calculated for each building, i and summed to provide the agent
        with a reward that is representative of all n buildings.
        It encourages net-zero energy use by penalizing grid load satisfaction
        when there is energy in the battery as well as penalizing
        net export when the battery is not fully charged through the penalty
        term. There is neither penalty nor reward when the battery
        is fully charged during net export to the grid. Whereas, when the
        battery is charged to capacity and there is net import from the
        grid the penalty is maximized.
        """

        reward_list = []

        for b in self.env.buildings:
            cost = b.net_electricity_consumption_cost[-1]
            battery_capacity = b.electrical_storage.capacity_history[0]
            battery_soc = b.electrical_storage.soc[-1]/battery_capacity
            penalty = -(1.0 + np.sign(cost)*battery_soc)
            reward = penalty*abs(cost)
            reward_list.append(reward)

        reward = [sum(reward_list)]

        return reward

def train_your_custom_sac(
    agent_kwargs: dict, episodes: int, reward_function: RewardFunction,
    building_count: int, day_count: int, active_observations: List[str],
    random_seed: int, reference_envs: Mapping[str, CityLearnEnv] = None
) -> dict:
    """Trains a custom TQL agent on a custom environment.

    Trains an TQL agent using a custom environment and agent hyperparamter
    setup and plots the key performance indicators (KPIs), actions and
    rewards from training and evaluating the agent.
    """

    # get schema
    schema = DataSet.get_schema('citylearn_challenge_2022_phase_all')

    # select buildings
    schema, buildings = set_schema_buildings(
        schema, building_count, random_seed
    )
    print('Selected buildings:', buildings)

    # select days
    schema, simulation_start_time_step, simulation_end_time_step =\
        set_schema_simulation_period(schema, day_count, random_seed)
    print(
        f'Selected {day_count}-day period time steps:',
        (simulation_start_time_step, simulation_end_time_step)
    )

    # set active observations
    schema = set_active_observations(schema, active_observations)
    print(f'Active observations:', active_observations)

    # initialize environment
    env = CityLearnEnv(schema, central_agent=True)

    # set reward function
    env.reward_function = reward_function(env=env)

    # define active observations and actions and their bin sizes
    observation_bins = {'hour': 24}
    action_bins = {'electrical_storage': 12}

    # initialize list of bin sizes where each building
    # has a dictionary in the list definining its bin sizes
    observation_bin_sizes = []
    action_bin_sizes = []

    for b in env.buildings:
        # add a bin size definition for the buildings
        observation_bin_sizes.append(observation_bins)
        action_bin_sizes.append(action_bins)

    print(observation_bin_sizes)
    print(action_bin_sizes)

    # wrap environment
    env = TabularQLearningWrapper(
    env.unwrapped,
    observation_bin_sizes=observation_bin_sizes,
    action_bin_sizes=action_bin_sizes
    )

    print('Q-Table dimension:', (m, n))
    print('Number of episodes to train:', tql_episodes)

    loader = get_loader(max=tql_episodes*t)
    display(loader)

    # initialize agent
    model = CustomTabularQLearning(
    env=env,
    loader=loader,
    random_seed=RANDOM_SEED,
    **agent_kwargs
    )
    _ = model.learn(episodes=tql_episodes)

    observations = env.reset()
    while not env.done:
        actions = model.predict(observations, deterministic=True)
        observations, _, _, _ = env.step(actions)

    # plot summary and compare with other control results
    plot_simulation_summary({'TQL': env})
    print(
    f'Current Tabular Q-Learning epsilon after {tql_episodes}'\
        f' episodes and {model.time_step} time steps:', model.epsilon
    )

results = train_your_custom_sac(
    agent_kwargs=tql_kwargs,
    episodes=tql_episodes,
    reward_function=CostMinimizing,
    building_count=BUILDING_COUNT,
    day_count=DAY_COUNT,
    active_observations=active_observations,
    random_seed=RANDOM_SEED,
    reference_envs={
        'TQL': env
    }
)

"""# Tuning the model - minimizing Carbon Intensity (CI)

### p_i = -(1 + sign(C_i) * SOC_i)
### Reward_i = sum(p_i * (1 - CI_i))
"""

for b in env.buildings:
  ci = b.carbon_intensity.carbon_intensity
  print(ci)

# -------------------- CUSTOMIZE ENVIRONMENT -----------------------------
active_observations = [
    'hour'
]

# --------------- DEFINE CUSTOM REWARD FUNCTION --------------------------
class CIMinimizing(RewardFunction):
    def __init__(self, env: CityLearnEnv):
        r"""Initialize CIMinimizing.
        """
        super().__init__(env)

    def calculate(self) -> List[float]:
        r"""Returns reward for most recent action.

        The reward is designed to minimize carbon intensity.
        """

        reward_list = []

        for b in self.env.buildings:
            cost = b.net_electricity_consumption_cost[-1]

            battery_capacity = b.electrical_storage.capacity_history[0]
            battery_soc = b.electrical_storage.soc[-1]/battery_capacity
            penalty = -(1.0 + np.sign(cost)*battery_soc)
            carbon_intensity_values = np.array(b.carbon_intensity.carbon_intensity)
            normalized_carbon_intensity = (carbon_intensity_values - np.min(carbon_intensity_values)) / (np.max(carbon_intensity_values) - np.min(carbon_intensity_values))
            reward = penalty * (1 - normalized_carbon_intensity[-1])
            reward_list.append(reward)

        reward = [sum(reward_list)]

        return reward

def train_your_custom_sac(
    agent_kwargs: dict, episodes: int, reward_function: RewardFunction,
    building_count: int, day_count: int, active_observations: List[str],
    random_seed: int, reference_envs: Mapping[str, CityLearnEnv] = None,
    show_figures: bool = None
) -> dict:
    """Trains a custom TQL agent on a custom environment.
    """

    # get schema
    schema = DataSet.get_schema('citylearn_challenge_2022_phase_all')

    # select buildings
    schema, buildings = set_schema_buildings(
        schema, building_count, random_seed
    )
    print('Selected buildings:', buildings)

    # select days
    schema, simulation_start_time_step, simulation_end_time_step =\
        set_schema_simulation_period(schema, day_count, random_seed)
    print(
        f'Selected {day_count}-day period time steps:',
        (simulation_start_time_step, simulation_end_time_step)
    )

    # set active observations
    schema = set_active_observations(schema, active_observations)
    print(f'Active observations:', active_observations)

    # initialize environment
    env = CityLearnEnv(schema, central_agent=True)

    # set reward function
    env.reward_function = reward_function(env=env)

    # define active observations and actions and their bin sizes
    observation_bins = {'hour': 24}
    action_bins = {'electrical_storage': 12}

    # initialize list of bin sizes where each building
    # has a dictionary in the list definining its bin sizes
    observation_bin_sizes = []
    action_bin_sizes = []

    for b in env.buildings:
        # add a bin size definition for the buildings
        observation_bin_sizes.append(observation_bins)
        action_bin_sizes.append(action_bins)

    print(observation_bin_sizes)
    print(action_bin_sizes)

    # wrap environment
    env = TabularQLearningWrapper(
    env.unwrapped,
    observation_bin_sizes=observation_bin_sizes,
    action_bin_sizes=action_bin_sizes
    )

    print('Q-Table dimension:', (m, n))
    print('Number of episodes to train:', tql_episodes)

    loader = get_loader(max=tql_episodes*t)
    display(loader)

    # initialize agent
    model = CustomTabularQLearning(
    env=env,
    loader=loader,
    random_seed=RANDOM_SEED,
    **agent_kwargs
    )
    _ = model.learn(episodes=tql_episodes)

    observations = env.reset()
    while not env.done:
        actions = model.predict(observations, deterministic=True)
        observations, _, _, _ = env.step(actions)

    # plot summary and compare with other control results
    plot_simulation_summary({'TQL': env})
    print(
    f'Current Tabular Q-Learning epsilon after {tql_episodes}'\
        f' episodes and {model.time_step} time steps:', model.epsilon
    )

results = train_your_custom_sac(
    agent_kwargs=tql_kwargs,
    episodes=tql_episodes,
    reward_function=CIMinimizing,
    building_count=BUILDING_COUNT,
    day_count=DAY_COUNT,
    active_observations=active_observations,
    random_seed=RANDOM_SEED,
    reference_envs={
        'TQL': env
    },
    show_figures=True,
)
