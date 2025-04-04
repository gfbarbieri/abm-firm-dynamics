########################################################################
## IMPORTS
########################################################################

import math
import random

from mesa import Agent
from mesa.agent import AgentSet

from typing import Iterable

########################################################################
## WORKER AGENT
########################################################################

class Worker(Agent):
    """
    Represents a worker within an agent-based model. Workers can
    produce output, manage their wealth, and choose to move between
    firms based on various criteria.

    Attributes
    ----------
    mu (float)
        The expected return rate of the worker's investment.
    sigma (float):
        The volatility of the worker's investment return.
    wealth (float):
        The current wealth of the worker.
    employer_id (int):
        The ID of the current employer firm.
    employer_size (int):
        The size of the current employer firm.
    network: AgentSet
        The network of other workers the agent is connected to.
    mutual_acceptance (bool):
        Whether the worker must accept an offer from a firm to switch
        employers.
    global_search_rate (float):
        The rate at which the worker searches for a new employer outside
        their network.
    constant_sigma (float | None):
        The constant sigma value for the worker. If None, the sigma is
        set to a random value that ensures the worker's growth rate is
        positive.
    track_wealth (bool):
        Whether the worker's wealth is updated each step. Alternatively,
        the wealth is fixed at 1, limiting the amount of wealth that can
        be contributed to the firm at each step.
    job_history : list
        A list tracking the worker's employment history.
    """

    def __init__(
            self, model, mutual_acceptance: bool=True,
            global_search_rate: float=0.01, constant_mu: float | None=None,
            constant_sigma: float | None=None, num_neighbors: int | None=None,
            track_wealth: bool=False, positive_g: bool=True
        ) -> None:
        """
        Initializes the worker with random attributes for investment 
        return and volatility, sets initial wealth, and clears
        employment information.

        Attributes
        ----------
        mu : float
            The expected return rate of the worker's investment.
        sigma : float
            The volatility of the worker's investment return.
        g : float
            The growth rate of the worker's investment. If random mu and
            sigma are selected, then the growth rate is designed to be
            positive.
        wealth : float
            The current wealth of the worker.
        employer_id : int | None
            The ID of the current employer firm.
        employer_size : int | None
            The size of the current employer firm.
        network : AgentSet
            The network of other workers the agent is connected to.
        mutual_acceptance : bool
            Whether the worker must accept an offer from a firm to
            switch employers.
        global_search_rate : float
            The rate at which the worker searches for a new employer
            outside their network.
        constant_mu : float | None
            The constant mu vlaue for the worker. IF None, then the mu
            is set to a random value that when combined with a random
            sigma ensures the worker's growth rate is positive.
        constant_sigma : float | None
            The constant sigma value for the worker. If None, the sigma
            is set to a random value that when combined with a random
            sigma ensures the worker's growth rate is positive.
        num_neighbors : int | None
            The number of neighbors the worker has. If None, then the
            number of neighbors is selected at random between 2 and 6.
        track_wealth : bool
            Whether the worker's wealth is updated each step.
            Alternatively, the wealth is fixed at 1, limiting the
            amount of wealth that can be contributed to the firm at
            each step.
        job_history : list
            A list tracking the worker's employment history.
        """

        # Initialize the agent with the model object.
        super().__init__(model)

        # Set the parameters.
        self.mutual_acceptance = mutual_acceptance
        self.global_search_rate = global_search_rate
        self.constant_mu = constant_mu
        self.constant_sigma = constant_sigma
        self.track_wealth = track_wealth

        # Set mu and sigma such that the growth rate is always positive.
        # The only exception is if the user specifically passes a value
        # of mu and sigma that generates a negative growth rate for the
        # agents. In that case, all agents will get the same negative
        # growth rate.
        # 
        # Business logic for assigning positive growth rates:
        # Case 1: If mu and sigma are both constant, then set mu to the
        # constant and set sigma to the constant.
        # 
        # Case 2: If mu is not constant, and sigma is constant, then set
        # mu to a value between no lower than sigma**2/2 and no higher
        # than sigma**2/2 + 0.01 and set sigma to the constant.
        #
        # Case 3: If mu is constant, and sigma is not constant, then set
        # mu to the constant, and set sigma no lower than 0 and no higher
        # than sqrt(2 * mu).
        #
        # Case 4: If mu and sigma are both not constant, then set mu to a
        # random value between 0 and 0.01, and set sigma to a random value
        # no lower than 0 and no higher than sqrt(2 * mu).

        if self.constant_mu and self.constant_sigma:
            self.mu = self.constant_mu
            self.sigma = self.constant_sigma
        elif self.constant_mu is None and self.constant_sigma and positive_g:
            self.mu = self.random.uniform(
                self.constant_sigma**2/2, self.constant_sigma**2/2 + 0.01
            )
            self.sigma = self.constant_sigma
        elif self.constant_mu is None and self.constant_sigma and not positive_g:
            self.mu = self.random.uniform(0, 0.01)
            self.sigma = self.constant_sigma
        elif self.constant_mu and self.constant_sigma is None and positive_g:
            self.mu = self.constant_mu
            self.sigma = self.random.uniform(0, math.sqrt(2 * self.mu))
        elif self.constant_mu and self.constant_sigma is None and not positive_g:
            self.mu = self.constant_mu
            self.sigma = self.random.uniform(0, 0.1)
        elif self.constant_mu is None and self.constant_sigma is None and positive_g:
            self.mu = self.random.uniform(0, 0.01)
            self.sigma = self.random.uniform(0, math.sqrt(2 * self.mu))
        elif self.constant_mu is None and self.constant_sigma is None and not positive_g:
            self.mu = self.random.uniform(0, 0.01)
            self.sigma = self.random.uniform(0, 0.1)

        # Calculate the growth rate. If random selection is used for
        # mu and sigma, then the growth rate is designed to be always
        # positive.
        self.g = self.mu - (self.sigma ** 2) / 2

        # Initialize wealth. For now, this is fixed at 1 and does not
        # change despite the worker receiving a wage. Since the
        # aspects of this model rely entirely on the growth rate of
        # wealth, the actual value of wealth is never used to assess
        # the results of the model. That said, it may affect the
        # dynamics of the model since workers with more wealth will
        # have more resources to invest, making the total output of the
        # firm higher.
        self.wealth = 1

        # Initialize employment information.
        self.employer_id = None
        self.employer_size = None
        self.job_history = []

        # Initialize search, offer, and switch information. Search is a
        # boolean that indicates if the worker was selected to actively
        # search for a new employer in the current step. Offer is a
        # boolean that indicates if the worker has received an offer
        # from a firm during their search. Switch is a boolean that
        # indicates if the worker has accepted an offer and switched
        # employers.
        # 
        # Both the search and offer indicators are diagnostic tools to
        # see how many workers are searching for a new job and how many
        # workers have received an offer. These two indicators can be 
        # removed from the model without affecting any of the results.
        self.search = None
        self.offer = None
        self.switch = None

        # Initialize the number of neighbors.
        if num_neighbors is None:
            self.num_neighbors = self.random.randint(2, 6)
        else:
            self.num_neighbors = num_neighbors

        self.network = None

    def init_network(self) -> None:
        """
        Initializes the worker's network by selecting a random subset of
        other workers from the model's worker list, excluding the
        current worker. The selection is done without replacement.
        """

        # Get the worker agents from the model's agent set, excluding
        # the current worker. This returns an AgentSet and can return an
        # empty set if the current worker is the only worker in the
        # model.
        workers = self.model.workers.select(
            filter_func=lambda x: x.unique_id != self.unique_id
        )

        # Randomly select a subset of agents from the population.
        # The number of agents selected is equal to the number of
        # neighbors. This returns a list and can return an empty list.
        workers = self.random.sample(population=workers, k=self.num_neighbors)

        # Assign to the workers network as a AgentSet object.
        self.network = AgentSet(agents=workers, random=self.random)

    def produce(self) -> float:
        """
        Produces output based on the worker's investment return and
        volatility, updating the worker's output accordingly.

        Output = W * exp((mu - sigma^2/2) + sigma * W)

        Returns
        -------
        float
            The output produced by the worker.
        """

        # Generate a random number from a normal distribution.
        W = random.normalvariate(mu=0, sigma=1)

        # Calculate the output produced by the worker.
        output = self.wealth * math.exp(self.g + self.sigma * W)

        return output

    def select_firm(self) -> None:
        """
        Performs the firm selection process for the worker, comparing
        the current employer's growth rate with potential new employers
        and moving to a new firm if advantageous.
        """

        # Get potential firms.
        potential_firms = self.get_potential_firms()

        # Rank potential firms. By default, the firms are sorted in
        # descending order of growth rate, and the growth rates are
        # returned.
        potential_firms = self.rank_firms(firms=potential_firms)

        # Ask each firm if the worker can join the firm. If True, then
        # as the worker if they want to join the firm. If so, then
        # switch firms. If the worker switches firms, then there is no
        # need to check the remaining firms, so break out of the loop.
        for firm, new_g in potential_firms:

            # If the two-way setting is toggled on, then workers will
            # ask the firm if the worker can join the firm. The firm
            # will only do is if adding the worker will increase the
            # firm's growth rate. Otherwise, the worker can join the
            # firm and the offer is always True.
            if self.mutual_acceptance == True:
                offer = firm.offer(worker=self)
            else:
                offer = True

            # If True, check if the worker wants to join the firm.
            if offer == True:

                # Indicate that the worker has received an offer.
                self.offer = True

                # Get the worker's firm and use the firm agent to
                # calculate the growth rate of the current firm.
                curr_g = (
                    self.get_worker_firms(workers=[self])[0]
                    .calc_time_avg_growth_rate()
                )

                # If the growth rate of the new firm is higher than the
                # growth rate of the current firm, then accept the offer
                # and switch firms.
                if new_g > curr_g:

                    # Change the worker's employer.
                    self.change_employer(firm=firm)

                    # Indicate that the worker has switched firms.
                    self.switch = True

                    # Break out of the loop. No need to check the
                    # remaining firms.
                    break

                else:

                    # Indicate that the worker has not switched firms.
                    self.switch = False

            else:

                # Indicate that the worker has not received an offer.
                self.offer = False

    def rank_firms(
            self, firms: AgentSet, descending: bool=True, return_g: bool=True
        ) -> list[tuple[Agent, float]] | list[Agent]:
        """
        Ranks firms for employment based on their calculated
        time-averaged growth rate, including start-ups and firms from
        the worker's network.

        Parameters
        ----------
        firms
            A list of firms to rank.
        descending
            Whether to sort the firms in descending order. Passed to
            the `sorted` function's `reverse` argument.
        return_g
            Whether to return a the sorted list of firms with or without
            their growth rates.

        Returns
        -------
        list[tuple[Agent, float]] | list[Agent]
            A list of tuples of firms and their growth rates, or a list
            of firms.
        """

        # Calculate the time average growth rate of each firm if the
        # worker was to join the firm. Store the information as a list
        # of tuples.
        g_f = [
            (firm, firm.calc_time_avg_growth_rate(add_workers=[self]))
            for firm in firms
        ]
 
        # Sort the list of tuples descending by growth rate. Lambda
        # focuses the sorting on the second element in the tuple, which
        # is the growth rate.
        g_f = sorted(g_f, key=lambda x: x[1], reverse=descending)

        # Return the sorted list of firms with or without their growth
        # rates.
        if return_g == True:
            return g_f
        elif return_g == False:
            return [g[0] for g in g_f]

    def get_potential_firms(self) -> list[Agent]:
        """
        Generate a list of potential firm agents. Since potential firms
        always includes a start-up firm, the list will always include
        at least one firm.

        Returns
        -------
        list[Agent]
            List of firm agents.
        """

        # Select firms. 1% of the time select firms from the general
        # worker population. Otherwise, select firms from the worker's
        # network. This returns a list and can return an empty list.
        if self.random.random() < self.global_search_rate:
            worker_firms = self.select_firms_with_workers(k=self.num_neighbors)
        else:
            worker_firms = self.select_firms_from_network()

        # Select a "start-up" firm. A start-up firms is a firm with zero
        # employees. This returns a list and can return an empty list.
        startup = self.select_empty_firm()        

        # Combine the two AgentSets to form a set of potential firms.
        # This returns a list of firm agents and can return an empty
        # list.
        potential_firms = worker_firms + startup

        return potential_firms

    def select_firms_with_workers(self, k: int) -> list[Agent]:
        """
        Selects firms with workers at random from the entire population
        of firms.

        Parameters
        ----------
        k : int
            The number of firms to select.

        Returns
        -------
        list
            A list of firms with workers.
        """

        # Get the workers in the model excluding the current worker.
        # If the employer ID of both agents is None, then the agent in
        # the list of workers is not selected. This returns an AgentSet
        # and can return an empty AgentSet.
        workers = self.model.workers.select(
            filter_func=lambda x: x.unique_id != self.unique_id
        )

        # If the number of workers selected is greater than the number
        # of neighbors, then select a random sample of workers equal to
        # the number of neighbors. Otherwise, select all workers.
        if len(workers) > k:
            workers = self.random.sample(population=workers, k=k)

        # Get a list of the firms at which the workers are employed.
        firms = self.get_worker_firms(workers=workers, exclude_current=True)

        return firms

    def select_firms_from_network(self) -> list[Agent]:
        """
        Selects firms from the worker's network, excluding the current
        employer.

        Returns
        -------
        list
            A set of firms from the agent's network.
        """

        # Get the workers in the agent's network that are not employed
        # at the agent's firm. Networks are always made up of workers.
        # If the employer ID of both agents is None, then the agent in
        # the network is not selected. This returns an Agenetset and can
        # return an empty set.
        workers = self.network.select(
            filter_func=lambda x: x.employer_id != self.employer_id
        )

        # Get a list of the firms at which the workers are employed.
        firms = self.get_worker_firms(workers=workers, exclude_current=True)

        return firms

    def select_empty_firm(self) -> list[Agent]:
        """
        Selects a firm with zero workers at random from the entire
        population of firms.

        Returns
        -------
        list
            A list of firms with zero workers.
        """

        # Choose firms with zero workers. This will return an agent
        # set.
        firms = self.model.firms.select(filter_func=lambda x: x.size == 0)

        # Choose an single empty firm at random.
        empty_firm = self.random.sample(firms, k=1)

        return empty_firm

    def get_worker_firms(
            self, workers: Iterable[Agent], exclude_current: bool=False
        ) -> list[Agent]:
        """
        Get a list of the firms at which the workers are employed.

        Parameters
        ----------
        workers : list
            A set of worker agents.
        exclude_current
            Whether to exclude the current employer from the list of
            firms.

        Returns
        -------
        list
            A list of firms with workers.
        """

        # Get a list of the firms at which the workers are employed.
        # Duplicate firms will be ignored. This can also return an empty
        # list.
        firms = []

        for worker in workers:

            # If the current firm is to be excluded, then select the
            # worker's firm, returns an AgentSet. The set should contain
            # only at most one firm as a worker can only be employed by
            # one firm at a time. Otherwise, select the worker's firm
            # without any restrictions.
            if exclude_current == True:
                firm_set = self.model.firms.select(
                    lambda x: (
                        x.unique_id == worker.employer_id
                        and x.unique_id != self.employer_id
                    )
                )
            else:
                firm_set = self.model.firms.select(
                    lambda x: x.unique_id == worker.employer_id
                )

            # Add the firm to the list of firms if it is not already in
            # the list and is not the current employer.
            for firm in firm_set:
                if firm not in firms:
                    firms.append(firm)
    
        return firms

    def change_employer(self, firm: Agent) -> None:
        """
        Changes the worker's employer to the specified firm.

        Parameters
        ----------
        firm : Agent
            The firm agent to set as the employer.
        """

        # If the worker is currently employed, then separate from the
        # current firm. None status indicates that the worker has never
        # been employed, which is true at initialization.
        if self.employer_id != None:

            # Get the current employer firm.
            self.get_worker_firms(workers=[self])[0].separate(self)

        # Have the firm hire the new worker.
        firm.hire(self)

        # Update the worker's employer ID.
        self.employer_id = firm.unique_id

        # Update the worker's work history.
        self.job_history.append([self.model.steps, self.employer_id])

        # Update employer size.
        self.update_employer_size()

    def update_employer_size(self) -> None:
        """
        Updates the worker's employer size to the current size of the
        employer firm.
        """

        # If the employer ID is None, then the worker is not employed.
        # The None status indicates that the worker has never been
        # employed, which is true at initialization.
        if self.employer_id is None:
            self.employer_size = None

        # Otherwise, get the employer firm from the model's firms.
        else:
            # This returns an AgentSet with one firm.
            employer = self.model.firms.select(
                filter_func=lambda x: x.unique_id == self.employer_id
            )

            # Update the employer size.
            self.employer_size = employer[0].size   

    def update_wealth(self, wealth: float) -> None:
        """
        Update the worker's wealth.

        Parameters
        ----------
        wealth: float
            The new wealth for the worker.
        """

        # If the wealth is to be updated, then update the worker's
        # wealth.
        if self.track_wealth == True:
            # Update the worker's wealth.
            self.wealth = wealth
        else:
            # Do not update the worker's wealth.
            pass