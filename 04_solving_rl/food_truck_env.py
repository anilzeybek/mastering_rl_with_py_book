import numpy as np
import gym


class FoodTruck(gym.Env):
    def __init__(self):
        self.v_demand = [100, 200, 300, 400]
        self.p_demand = [0.3, 0.4, 0.2, 0.1]

        self.capacity = self.v_demand[-1]
        self.days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Weekend']

        self.unit_cost = 4
        self.net_revenue = 7

        self.action_space = [0, 100, 200, 300, 400]
        # The reason we don't include 400 on state is because at least 100 burger sold everyday,
        # And mondays start with 0 burgers always
        self.state_space = [("Mon", 0)] + [(d, i) for d in self.days[1:] for i in [0, 100, 200, 300]]

    def get_next_state_reward(self, state, action, demand):
        """
        Returns the scenario after choosing the given action in given given state with given demand
        """
        day, inventory = state
        result = {}

        result['next_day'] = self.days[self.days.index(day) + 1]
        result['starting_inventory'] = min(self.capacity, inventory + action)
        result['cost'] = self.unit_cost * action
        # We can only sell what we have already
        result['sales'] = min(result['starting_inventory'], demand)
        result['revenue'] = self.net_revenue * result['sales']
        result['next_inventory'] = result['starting_inventory'] - result['sales']
        result['reward'] = result['revenue'] - result['cost']

        return result

    def get_transition_prob(self, state, action):
        """
        Returns probabilities of next state and reward based on given state and action
        """
        next_s_r_prob = {}
        for i, demand in enumerate(self.v_demand):
            result = self.get_next_state_reward(state, action, demand)

            next_s = (result['next_day'], result['next_inventory'])
            reward = result['reward']
            prob = self.p_demand[i]

            if (next_s, reward) not in next_s_r_prob:
                next_s_r_prob[next_s, reward] = prob
            else:
                next_s_r_prob[next_s, reward] += prob

        return next_s_r_prob

    def reset(self):
        self.day = "Mon"
        self.inventory = 0
        state = (self.day, self.inventory)

        return state

    def is_terminal(self, state):
        day, _ = state
        if day == "Weekend":
            return True
        else:
            return False

    def step(self, action):
        demand = np.random.choice(self.v_demand, p=self.p_demand)
        result = self.get_next_state_reward((self.day, self.inventory), action, demand)

        self.day = result['next_day']
        self.inventory = result['next_inventory']

        state = (self.day, self.inventory)
        reward = result['reward']
        done = self.is_terminal(state)
        info = {'demand': demand, 'sales': result['sales']}

        return state, reward, done, info
