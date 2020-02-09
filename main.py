import gym
import gym_simulator.envs.simulator_env
import numpy as np
import time

import sys, os

class CPV_fixed():
    fixed_cpv = 0.1
    def run(self):
        return self.fixed_cpv


class CPV_norm():
    def run(self, history):
        mu = np.mean(history)
        sigma = np.std(history)
        return max(0.0, np.random.normal(loc=mu, scale=sigma))


class CVR_fixed():
    fixed_cvr = 0.15
    def run(self):
        return self.fixed_cvr


class CVR_norm():
    def run(self, history):
        mu = np.mean(history)
        sigma = np.std(history)
        return max(0.0, np.random.normal(loc=mu, scale=sigma))


class AOV_fixed():
    fixed_cpm = 22.3
    def run(self):
        return self.fixed_cpm


class AOV_norm():
    def run(self, history):
        mu = np.mean(history)
        sigma = np.std(history)
        return max(0.0, np.random.normal(loc=mu, scale=sigma))


class OtherSessions_fixed():
    fixed_sessions= 1000
    def run(self):
        return self.fixed_sessions


class OtherSessions_norm():
    def run(self, history):
        mu = np.mean(history)
        sigma = np.std(history)
        return max(0.0, np.random.normal(loc=mu, scale=sigma))


if __name__ == '__main__':


    ### fake data  ###

    facebook_ads_cpv = [1.2, 2.5, 3.3, 1.6, 2.5]
    google_ads_cpv = [1.4, 1.5, 2.5, 1.7, 1.3]
    cpv_data = [facebook_ads_cpv, google_ads_cpv]

    other_sessions_data = [1000, 1200, 1300, 1100, 900]

    facebook_ads_cvr = [0.2, 0.5, 0.3, 0.6, 0.5]
    google_ads_cvr = [0.4, 0.5, 0.5, 0.7, 0.3]
    other_sessions_cvr = [0.4, 0.5, 0.5, 0.7, 0.3]
    cvr_data = [facebook_ads_cvr, google_ads_cvr, other_sessions_cvr]

    facebook_ads_aov = [20.2, 21.5, 23.3, 21.6, 21.5]
    google_ads_aov = [25.4, 20.5, 24.5, 20.7, 22.3]
    other_sessions_aov = [23.4, 20.5, 20.5, 22.7, 21.3]
    aov_data = [facebook_ads_aov, google_ads_aov, other_sessions_aov]

    env = gym.make('simulator-v0')
    env.configure_env(ad_channels=['facebook_ads', 'google_ads'],
                      other_channels=['other'],
                      cpv_data=cpv_data,
                      other_sessions_data=other_sessions_data,
                      cvr_data=cvr_data,
                      aov_data=aov_data,
                      CPV_model=CPV_norm(),
                      OtherSessions_model=OtherSessions_norm(),
                      CVR_model=CVR_norm(),
                      AOV_model=AOV_norm()
                      )

    action = [3000, 1000]
    episodes = 2
    time_horizon = 5

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
