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
    """

    def __init__(self, model) -> None:
        """
        Initializes the worker with random attributes for investment 
        return and volatility, sets initial wealth, and clears
        employment information.

        Attributes
        ----------
        mu : float
            The expected return rate of the worker's investment,
            randomly initialized between 0 and 0.01.
        sigma : float
            The volatility of the worker's investment return, randomly
            initialized based on mu.
        g : float
            The growth rate of the worker's investment, calculated as mu
            minus half the variance.
        wealth : float
            The initial wealth of the worker, set to 1.
        employer_id : int or None
            The ID of the current employer firm, initially set to None.
        employer_size : int or None
            The size of the current employer firm, initially set to
            None.
        network : AgentSet or None
            The network of other workers the agent is connected to,
            initially set to None.
        """

        # Initialize the agent with the model object.
        super().__init__(model)

        # Initialize the worker with random investment return and
        # volatility, and calculate the growth rate.
        self.mu = self.random.uniform(0, 0.01)
        self.sigma = self.random.uniform(0, math.sqrt(2 * self.mu))
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

        # Initialize the social network.
        self.num_neighbors = self.random.randint(2, 6)
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
            if self.model.mutual_acceptance == True:
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
            self, firms: AgentSet, ascending: bool=False, return_g: bool=True
        ) -> list[tuple[Agent, float]] | list[Agent]:
        """
        Ranks firms for employment based on their calculated
        time-averaged growth rate, including start-ups and firms from
        the worker's network.

        Parameters
        ----------
        firms
            A list of firms to rank.
        ascending
            Whether to sort the firms in ascending order. Passed to
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
        g_f = sorted(g_f, key=lambda x: x[1], reverse=ascending)

        # Return the sorted list of firms with or without their growth
        # rates.
        if return_g == True:
            return g_f
        elif return_g == False:
            return [g[0] for g in g_f]

    def get_potential_firms(self) -> list[Agent]:
        """
        Generate a list of potential firm agents.

        Returns
        -------
        list[Agent]
            List of firm agents.
        """

        # Select firms. 1% of the time select firms from the general
        # worker population. Otherwise, select firms from the worker's
        # network. This returns a list and can return an empty list.
        if self.random.random() < self.model.global_search_rate:
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
        firms = self.get_worker_firms(workers=workers)

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
        firms = self.get_worker_firms(workers=workers)

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

    def get_worker_firms(self, workers: Iterable[Agent]) -> list[Agent]:
        """
        Get a list of the firms at which the workers are employed.

        Parameters
        ----------
        workers : list
            A set of worker agents.

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

            # Select firms, returns an AgentSet. The set should contain
            # only one firm.
            firm_set = self.model.firms.select(
                lambda x: x.unique_id == worker.employer_id
            )

            # Add the firm to the list of firms if it is not already in
            # the list. This works with empty lists.
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

    def update_wealth(self, wage: float) -> None:
        """
        Update the worker's wealth.

        Parameters
        ----------
        wage: float
            The new wage for the worker.
        """
        
        # Update the worker's wealth.
        self.wealth += wage