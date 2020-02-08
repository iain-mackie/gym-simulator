import gym
import gym_simulator.envs.simulator_env
from gym.envs.registration import register
from gym import envs
import numpy as np
import time

import sys, os

class CPM_fixed():
    fixed_cpm = 1.2
    def get_cpm(self):
        return self.fixed_cpm


class CPM_norm():
    def get_cpm(self, history):
        mu = np.mean(history)
        sigma = np.std(history)
        return max(0.0, np.random.normal(loc=mu, scale=sigma))


class CTR_fixed():
    fixed_ctr = 0.1
    def get_ctr(self):
        return self.fixed_ctr


class CTR_norm():
    def get_ctr(self, history):
        mu = np.mean(history)
        sigma = np.std(history)
        return max(0.0, np.random.normal(loc=mu, scale=sigma))


class CVR_fixed():
    fixed_cvr = 0.15
    def get_cvr(self):
        return self.fixed_cvr


class CVR_norm():
    def get_cvr(self, history):
        mu = np.mean(history)
        sigma = np.std(history)
        return max(0.0, np.random.normal(loc=mu, scale=sigma))


class AOV_fixed():
    fixed_cpm = 22.3
    def get_aov(self):
        return self.fixed_cpm


class AOV_norm():
    def get_aov(self, history):
        mu = np.mean(history)
        sigma = np.std(history)
        return max(0.0, np.random.normal(loc=mu, scale=sigma))


class OrganicSessions_fixed():
    fixed_sessions= 1000
    def get_sessions(self):
        return self.fixed_sessions


class OrganicSessions_norm():
    def get_sessions(self, history):
        mu = np.mean(history)
        sigma = np.std(history)
        return max(0.0, np.random.normal(loc=mu, scale=sigma))


if __name__ == '__main__':


    ### fake data  ###

    facebook_ads_cpm = [1.2, 2.5, 3.3, 1.6, 2.5]
    google_ads_cpm = [1.4, 1.5, 2.5, 1.7, 1.3]
    cpm_data = [facebook_ads_cpm, google_ads_cpm]

    facebook_ads_ctr = [0.02, 0.01, 0.015, 0.02, 0.02]
    google_ads_ctr = [0.005, 0.01, 0.007, 0.008, 0.005]
    ctr_data = [facebook_ads_ctr, google_ads_ctr]

    organic_sessions_data = [1000, 1200, 1300, 1100, 900]

    cvr_data = [0.12, 0.1, 0.22, 0.13, 0.14]

    aov_data = [12.2, 10.3, 11.0, 13.8, 14.3]

    env = gym.make('simulator-v0')
    env.configure_env(ad_channels=['facebook_ads', 'google_ads'],
                      cpm_data=cpm_data,
                      ctr_data=ctr_data,
                      organic_sessions_data=organic_sessions_data,
                      cvr_data=cvr_data,
                      aov_data=aov_data,
                      CPM_model=CPM_norm(),
                      CTR_model=CTR_norm(),
                      OrganicSessions_model=OrganicSessions_norm(),
                      CVR_model=CVR_norm(),
                      AOV_model=AOV_norm()
                      )

    action = [3000, 1000]
    episodes = 1000000
    time_horizon = 30

    for e in range(episodes):
        start_time = time.time()

        print('------------------------------')
        print('EPISODE: {}'.format(e))
        print('------------------------------')

        env.reset()
        for t in range(time_horizon):
            print('------------------------------')
            print('TIMESTAMP: {}'.format(t))
            print('------------------------------')
            observation, reward, done, info = env.step(action=action)
            #print(observation, reward, done, info)

        end_time = time.time()
        delta = end_time - start_time

        print(e, delta)

    env.close()
