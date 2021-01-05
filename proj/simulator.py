# Author: Quan Wang
# Email: wang4607@purdue.edu
# Last Modified: 2020-12-10 12:34:54

import networkx as nx
import numpy as np
import pandas as pd


class Student:
    STU_COUNT = 0

    def __init__(self, L, friends, probs=(0.40, 0.55, 0.05), cure_dates=(10, 10, 21)):
        """
        Student class
        :param L: After L days, the chances of being fully asymptomatic, mild, severe
                  are 40%, 55%, and 5%, respectively.
        :param friends: The friend list of a student
        :param probs: An interface to change fully asymptomatic, mild, severe probability
        :param cure_dates:  Individuals who are fully asymptomatic or have endured the mild
                            symptoms will be cured and immune against the disease after 10 days since
                            the first day of the infection.
                            Individuals who have severe symptoms will be cured and immune against the
                            disease after 21 days since the first day of the infection.
        """
        assert len(probs) == 3, \
            f"expected length 3 of prob_list but length {len(probs)} received"

        self.stu_id = Student.STU_COUNT
        Student.STU_COUNT += 1

        # status {1: got infected in L days,
        #         2: fully asymptomatic
        #         3: mild symptoms
        #         4: severe symptoms}
        self.status = 0
        self.since_influence = 0
        self.L = L
        self.friends = friends
        self.probs = probs
        self.cure_date = cure_dates
        self.tested_positive = False

    def infect(self):
        self.status = 1

    def test(self):
        if self.status != 0:
            self.tested_positive = True

    def cure(self):
        self.status = 0
        self.since_influence = 0
        self.tested_positive = False

    def spends_one_day(self):
        if self.status > 0:
            self.since_influence += 1

            if self.since_influence == self.L:
                self.status = np.random.choice([2, 3, 4], p=self.probs)

            if self.status > 2:
                self.test()

            if self.since_influence >= self.L and \
                    self.since_influence == self.cure_date[self.status - 2]:
                self.cure()

    def __repr__(self):
        rep = f"{self.stu_id}"
        if self.status > 0:
            rep += "(infected)"
        if self.tested_positive:
            rep += "(tested)"
        return rep


class Covid19Simulator:

    def __init__(self,
                 R=0.5,
                 K=4,
                 P=0.001,
                 I=0.35,
                 L=3,
                 T=0.4,
                 S=0.01,
                 stu_num=1000,
                 std_kwargs={},
                 seed=1):
        """
        Covid19 Simulator
        :param R: rewiring probability of watts strogatz graph
        :param K: Each node is connected to K nearest neighbors in ring topology
        :param P: Probability to be infected due to everyday activities
        :param I: Chance of infection from an infected person to a healthy neighbor
                  during the interaction.
        :param L: After L days, the chances of being fully asymptomatic,
                  mild, severe are 40%, 55%, and 5%, respectively.
        :param T: An individual decides to get tested with probability
                  if at least one of his/her neighbors is notified to be infected.
        :param S: Surveillance testing” for a certain fraction of students every day.
        :param stu_num: total student number
        :param std_kwargs: students' property, see Student
        :param seed: random seed
        """
        self.R = R
        self.K = K
        self.P = P
        self.I = I
        self.L = L
        self.T = T
        self.S = S
        self.stu_num = stu_num
        self.seed = seed

        self.log_dict = {
            "infected number today": [],
            "tested number today": [],
            "tested positive rate today": [],
            "cumulative infected number": [],
            "cumulative tested number": [],
            "seed": [],
            "date": []
        }

        self.graph = None
        self.stu_list = None
        self.cumu_inf = 0
        self.cumu_test = 0
        self.date = 0
        self.reset(seed, std_kwargs)

    def reset(self, seed, std_kwargs={}):
        self.seed = seed
        np.random.seed(self.seed)

        self.cumu_inf = 0
        self.cumu_test = 0
        self.date = 0

        self.graph = self._init_graph()
        Student.STU_COUNT = 0
        self.stu_list = self._init_stu_list(**std_kwargs)

    def _init_graph(self):
        return nx.watts_strogatz_graph(self.stu_num, self.K, self.R, self.seed)

    def _init_stu_list(self, **kwargs):
        friends_its = [self.graph.neighbors(n) for n in self.graph]
        friends_list = []
        for fr_it in friends_its:
            friends_list.append([fr for fr in fr_it])

        return [Student(self.L, friends, **kwargs) for friends in friends_list]

    def _log(self, inf_num, test_num, test_pos_num):
        self.log_dict["infected number today"].append(inf_num)
        self.log_dict["tested number today"].append(test_num)
        self.log_dict["tested positive rate today"].append(test_pos_num / (test_num + 1e-8))

        self.cumu_inf += inf_num
        self.cumu_test += test_num
        self.date += 1
        self.log_dict["cumulative infected number"].append(self.cumu_inf)
        self.log_dict["cumulative tested number"].append(self.cumu_test)
        self.log_dict["date"].append(self.date)
        self.log_dict["seed"].append(self.seed)

    def simulate_one_day(self):
        # An assumption here: the people will interact with each other
        # in the order that how the following loop scans. That is
        # saying, if someone get infected today, he will be contagious
        # in the following interaction. This is true in real world, once
        # someone is infected, he is contagious. The bias comes from the
        # fixed scanning order can be canceled out by running several
        # simulations with different random seeds.

        inf_num = 0
        test_num = 0
        test_pos_num = 0

        for s in self.stu_list:
            should_test = False
            s.spends_one_day()

            if s.tested_positive:
                # this poor guy is isolated
                continue

            if np.random.random_sample() < self.P and s.status == 0:
                # This guy got infected when shopping at Walmart
                s.infect()
                inf_num += 1

            for fr_id in s.friends:
                # interact with friends
                fr = self.stu_list[fr_id]
                if fr.status > 0 and not fr.tested_positive:
                    # one of his friends got infected and did not be isolated
                    if np.random.random_sample() < self.I and s.status == 0:
                        # This guy got infected from his friend.
                        # He did not blame his friend but blame the chocolate shoppe
                        s.infect()
                        inf_num += 1

                if fr.tested_positive:
                    # At least one of his friends tested positive. Dangerous! May take a test.
                    should_test = True

            if should_test:
                if np.random.random_sample() < self.T:
                    test_num += 1
                    s.test()
                    if s.tested_positive:
                        test_pos_num += 1
                continue  # conduct surveillance testing by the way

            if np.random.random_sample() < self.S:
                # Surveillance testing
                test_num += 1
                s.test()
                if s.tested_positive:
                    test_pos_num += 1

        self._log(inf_num, test_num, test_pos_num)

    def get_log(self):
        return pd.DataFrame.from_dict(self.log_dict)

    def draw(self):
        color_map = [s.status + 0.1 for s in self.stu_list]
        pos = nx.circular_layout(self.graph)
        nx.draw(self.graph, pos, node_color=color_map)

    def __repr__(self):
        return f"Covid19Simulator <<@ {id(self)}>>" \
               f"------" \
               f"P={self.P}\n" \
               f"I={self.I}\n" \
               f"L={self.L}\n" \
               f"T={self.T}\n" \
               f"S={self.S}" \
               f"The number of students: {self.stu_num}" \
               f"------"

