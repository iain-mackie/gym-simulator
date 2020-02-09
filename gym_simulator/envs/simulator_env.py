import gym

class SimulatorEnv(gym.Env):

    def __init__(self):

        print('Environment initialized')

        # ad channels metadata
        self.ad_channels = None
        self.num_ad_channels = None
        self.other_channels = None
        self.num_other_channels = None
        self.channels = None
        self.num_channels = None

        # data history
        self.cpv_data = None
        self.other_sessions_data = None
        self.cvr_data = None
        self.aov_data = None

        # predictions
        self.cpv_pred = None
        self.other_sessions_pred = None
        self.cvr_pred = None
        self.aov_pred = None

        # models
        self.CPV_model = None
        self.OtherSessions_model = None
        self.CVR_model = None
        self.AOV_model = None

        # simulator info
        self.done = False
        self.observation_space = []
        self.action_episode_memory = []

    def init_pred(self, num):
        pred = []
        for i in range(num):
            pred.append([])
        return pred

    def configure_env(self, ad_channels, other_channels, cpv_data, other_sessions_data, cvr_data, aov_data, CPV_model,
                      OtherSessions_model, CVR_model, AOV_model):
        print('Configure env')
        self.ad_channels = ad_channels
        self.num_ad_channels = len(ad_channels)
        self.other_channels = other_channels
        self.num_other_channels = len(other_channels)
        self.channels = ad_channels + other_channels
        self.num_channels = self.num_ad_channels + self.num_other_channels

        self.cpv_data = cpv_data
        self.other_sessions_data = other_sessions_data
        self.cvr_data = cvr_data
        self.aov_data = aov_data

        self.CPV_model = CPV_model
        self.OtherSessions_model = OtherSessions_model
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

        total_sessions = 0
        total_orders = 0
        total_sales = 0

        for i in range(self.num_channels):

            if self.channels[i] != 'other':
                ad_spend = action[i]
                print("*** {} spending = £{:,.0f} ***".format(self.channels[i], ad_spend))

                # Sessions
                cpv = self.CPV_model.run(history=self.cpv_data[i]+self.cpv_pred[i])
                self.cpv_pred[i].append(cpv)
                print('CPV = {}'.format(cpv))
                sessions = cpv * ad_spend
                print('Sessions = {:,.0f}'.format(sessions))

            else:
                # Other Sessions
                sessions = self.OtherSessions_model.run(
                    history=self.other_sessions_data + self.other_sessions_pred)
                self.other_sessions_pred.append(sessions)
                print('Other Sessions = {:,.0f}'.format(sessions))

            total_sessions += sessions

            # Orders
            cvr = self.CVR_model.run(history=self.cvr_data[i]+self.cvr_pred[i])
            self.cvr_pred.append(cvr)
            print('CVR = {}'.format(cvr))
            orders = sessions * cvr
            total_orders += orders
            print(' Orders = {}'.format(orders))

            # Sales
            aov = self.AOV_model.run(history=self.aov_data[i]+self.aov_pred[i])
            self.aov_pred.append(aov)
            print('AOV = {}'.format(aov))
            sales = orders * aov
            total_sales += sales

        print('*** Total Sessions: {:,.0f} ***'.format(total_sessions))
        print('*** Total Orders: {:,.0f} ***'.format(total_orders))
        print('*** Total Sales: {:,.0f} ***'.format(total_sales))

        print()


    def _get_reward(self):

        return 0.0


    def _get_state(self):
        """Get the observation."""

        return []


    def reset(self):

        print('Environment reset')
        self.cpv_pred = self.init_pred(num=self.num_ad_channels)
        self.other_sessions_pred = []
        self.cvr_pred = self.init_pred(num=self.num_channels)
        self.aov_pred = self.init_pred(num=self.num_channels)