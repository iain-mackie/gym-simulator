import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import random
import math

import logging.config


class SimulatorEnv(gym.Env):

    def __init__(self):

        print('Environment initialized')

        # ad channels metadata
        self.ad_channels = []
        self.num_ad_channels = 1

        # data history
        self.cpm_data = []
        self.ctr_data = []
        self.organic_sessions_data = []
        self.cvr_data = []
        self.aov_data = []

        # predictions
        self.cpm_pred = self.init_pred()
        self.ctr_pred = self.init_pred()
        self.organic_sessions_pred = []
        self.cvr_pred = []
        self.aov_pred = []

        # models
        self.CPM_model = []
        self.CTR_model = []
        self.OrganicSessions_model = []
        self.CVR_model = []
        self.AOV_model = []

        # simulator info
        self.done = False
        self.observation_space = []
        self.action_episode_memory = []

    def init_pred(self):
        pred = []
        for i in range(self.num_ad_channels):
            pred.append([])
        return pred

    def configure_env(self, ad_channels, cpm_data, ctr_data, organic_sessions_data, cvr_data, aov_data, CPM_model,
                      CTR_model, OrganicSessions_model, CVR_model, AOV_model):
        print('Configure env')

        self.ad_channels = ad_channels
        self.num_ad_channels = len(self.ad_channels)

        self.cpm_data = cpm_data
        self.ctr_data = ctr_data
        self.organic_sessions_data = organic_sessions_data
        self.cvr_data = cvr_data
        self.aov_data = aov_data

        self.CPM_model = CPM_model
        self.CTR_model = CTR_model
        self.OrganicSessions_model = OrganicSessions_model
        self.CVR_model = CVR_model
        self.AOV_model = AOV_model


    def step(self, action):
        print('Step successful!')

        if self.done:
            raise RuntimeError("Episode is done")

        self._take_action(action=action)
        reward = self._get_reward()
        observation = self._get_state()

        return observation, reward, self.done, {}


    def _take_action(self, action):

        if len(action) != self.num_ad_channels:
            raise RuntimeError("Not every ad channel has an action")

        total_impressions = 0
        total_sessions = 0

        for i in range(self.num_ad_channels):

            ad_spend = action[i]
            print("*** {} spending = £{:,.0f} ***".format(self.ad_channels[i], ad_spend))

            # Impressions
            cpm = self.CPM_model.get_cpm(history=self.cpm_data[i]+self.cpm_pred[i])
            self.cpm_pred[i].append(cpm)
            print('CPM = {}'.format(cpm))
            impressions = (ad_spend / cpm) * 1000
            total_impressions += impressions
            print('Impressions = {:,.0f}'.format(impressions))

            # Sessions
            ctr = self.CTR_model.get_ctr(history=self.ctr_data[i]+self.ctr_pred[i])
            self.ctr_pred[i].append(ctr)
            print('CTR = {}'.format(ctr))
            sessions = impressions * ctr
            total_sessions += sessions
            print('Sessions = {:,.0f}'.format(sessions))

        # Organic Sessions
        organic_sessions = self.OrganicSessions_model.get_sessions(history=self.organic_sessions_data+self.organic_sessions_pred)
        self.organic_sessions_pred.append(organic_sessions)
        print('Organic Sessions: {:,.0f}'.format(organic_sessions))

        # Organic Sessions
        print('*** Total Impressions: {:,.0f} ***'.format(total_impressions))
        print('*** Total Sessions: {:,.0f} ***'.format(total_sessions))

        # Orders
        cvr = self.CVR_model.get_cvr(history=self.cvr_data+self.cvr_pred)
        self.cvr_pred.append(cvr)
        print('CVR = {}'.format(cvr))
        orders = total_sessions * cvr
        print('***Total Orders = {} ***'.format(orders))

        # Sales
        aov = self.AOV_model.get_aov(history=self.aov_data+self.aov_pred)
        self.aov_pred.append(aov)
        print('AOV = {}'.format(aov))
        sales = orders * aov
        print('*** Total Sales = {} ***'.format(sales))
        print(len(self.cpm_data[0]), len(self.cpm_data[1]))
        print(len(self.ctr_data[0]), len(self.ctr_data[1]))
        print(len(self.organic_sessions_data))
        print(len(self.cvr_data))
        print(len(self.aov_data))

        print()


    def _get_reward(self):

        return 0.0


    def _get_state(self):
        """Get the observation."""

        return []


    def reset(self):

        print('Environment reset')
        self.cpm_pred = self.init_pred()
        self.ctr_pred = self.init_pred()
        self.organic_sessions_pred = []
        self.cvr_pred = []
        self.aov_pred = []