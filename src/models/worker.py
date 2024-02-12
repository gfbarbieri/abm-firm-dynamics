###########################################################################
# IMPORTS
###########################################################################

from agentpy import Agent, AgentDList
import math
import random

###############################################################################
# WORKER AGENT
###############################################################################

class Worker(Agent):
    """
    Represents a worker within an agent-based model. Workers can produce output,
    manage their wealth, and choose to move between firms based on various criteria.

    Attributes:
        mu (float): The expected return rate of the worker's investment.
        sigma (float): The volatility of the worker's investment return.
        wealth (float): The current wealth of the worker.
        employer (:obj:`Agent`): The firm the worker is currently employed by.
        employer_id (int): The ID of the current employer firm.
        employer_size (int): The size of the current employer firm.
        wage (float): The wage received from the current employer.
        output (float): The output produced by the worker.
        network (:obj:`AgentDList`): The network of other workers the agent is connected to.
    """

    ###########################################################################
    # INITIALIZATION
    ###########################################################################

    def setup(self):
        """
        Initializes the worker with random attributes for investment return and volatility,
        sets initial wealth, and clears employment information.
        """

        # Initialize attributes:
        self.mu = random.uniform(0, 0.01)
        self.sigma = random.uniform(0, math.sqrt(2 * self.mu))
        self.wealth = 1
        self.employer = None
        self.employer_id = None
        self.employer_size = None
        self.wage = 0
        self.output = 0
        self.network = None

    ###########################################################################
    # SETTERS & GETTERS
    ###########################################################################

    def get_wealth(self):
        """
        Returns the current wealth of the worker.

        Returns:
            float: The worker's wealth.
        """

        return self.wealth

    def get_mu(self):
        """
        Returns the expected return rate of the worker's investment.

        Returns:
            float: The expected return rate (mu).
        """

        return self.mu
    
    def get_sigma(self):
        """
        Returns the volatility of the worker's investment return.

        Returns:
            float: The volatility (sigma).
        """

        return self.sigma
    
    def get_employer(self):
        """
        Returns the firm the worker is currently employed by.

        Returns:
            :obj:`Agent`: The employer firm.
        """

        return self.employer
    
    def get_employer_size(self):
        """
        Returns the size of the current employer firm.

        Returns:
            int: The employer firm's size.
        """

        return self.employer_size
    
    def get_wage(self):
        """
        Returns the wage received from the current employer.

        Returns:
            float: The worker's wage.
        """

        return self.wage

    def set_employer_size(self):
        """
        Updates the employer size attribute based on the current employer's size.
        """

        self.employer_size=self.employer.size
    
    def update_employer(self, firm):
        """
        Updates the worker's employer to the specified firm.

        Parameters:
            firm (:obj:`Agent`): The new employer firm.
        """

        self.employer=firm
        self.employer_id=firm.id

    def update_output(self, output):
        """
        Updates the worker's output.

        Parameters:
            output (float): The new output to set.
        """

        self.output = output

    def update_wealth(self, wealth):
        """
        Updates the worker's wealth and sets the wage to the same value.

        Parameters:
            wealth (float): The new wealth to set.
        """

        self.wealth = wealth
        self.wage = wealth

    def init_network(self):
        """
        Initializes the worker's network by selecting a random subset of other workers.
        """

        self.network = (
            self.model.workers
            .select(self.model.workers.id != self.id)
            .random(n=self.model.p['num_neighbors'], replace=False)
            .to_list()
        )
    
    ###########################################################################
    # WEALTH FUNCTIONS
    ###########################################################################
    
    def produce(self):
        """
        Produces output based on the worker's investment return and volatility, updating
        the worker's output accordingly.

        Returns:
            float: The output produced by the worker.
        """
        
        # Increment wealth following multiplicative wealth dynamic.
        W = random.normalvariate(mu=0, sigma=1)
        
        output = (
            self.get_wealth() *
            math.exp(
                (self.get_mu() - (self.get_sigma() ** 2) / 2) + self.get_sigma() * W
            )
        )

        # Update output.
        self.update_output(output=output)
    
        return output
    
    def calc_time_avg_growth_rate(self, firm, include_self=False):
        """
        Calculates the time-averaged growth rate of wealth for a group of workers,
        optionally including the calling worker.

        Parameters:
            firm (:obj:`Agent`): The firm whose workers to consider.
            include_self (bool): Whether to include the calling worker in the calculation.

        Returns:
            float: The calculated time-averaged growth rate.
        """

        # Get list of workers from firms.
        workers = firm.workers
    
        # If a list of workers are passed, calculate the growth rate of
        # wealth for the group as a cooperating unit.
        if include_self:
            workers = workers + [self]

        # Number of workers in the list.
        N = len(workers)

        # Contribution of each worker in the list.
        g_i = [
            (worker.get_mu() - (worker.get_sigma() ** 2) / (2 * N))
            for worker in workers
        ]

        # Growth rate based on total contributions of workers.
        g = sum(g_i) / N

        return g

    ###########################################################################
    # FIRM SELECTION
    ###########################################################################

    def select_empty_firms(self):
        """
        Selects firms with zero workers at random from the entire population of firms.

        Returns:
            list: A list of firms with zero workers.
        """

        # Choose a firm at random from entire population of firms with zero
        # workers.
        empty_firms = (
            self.model.firms
            .select(self.model.firms.get_size() == 0)
            .random()
            .to_list()
        )

        return empty_firms
    
    def select_neighbor_firms(self):
        """
        Selects firms from the worker's network, excluding the current employer, and
        optionally includes firms from a broader population based on a probability.

        Returns:
            :obj:`AgentDList`: A list of firms selected based on the criteria.
        """

        # Create an agent list of the employers from the agent's network.
        firms = AgentDList(
            self.model,
            self.network.select(self.network.employer.id != self.employer.id).employer
        )

        # 1% of the agents select firms the general population.
        if random.random() < 0.01:

            workers = (
                self.model.workers
                .select(self.model.workers.employer.id != self.employer.id)
            )

            if len(workers) >= self.model.p['num_neighbors']:
                workers = (
                    workers
                    .random(n=self.model.p['num_neighbors'], replace=False)
                    .to_list()
                )

            firms = AgentDList(
                self.model,
                workers.employer
            )
        
        return firms

    def rank_firms(self):
        """
        Ranks potential firms for employment based on their calculated time-averaged growth rate,
        including start-ups and firms from the worker's network.

        Returns:
            tuple: The firm with the highest growth rate and its growth rate value.
        """
    
        # Start-ups are firms with zero employees.
        startup = self.select_empty_firms()

        # Neighbor's firms are from a random sample of an agent's network.
        neighbor_firms = self.select_neighbor_firms()

        # Potential firms: startups plus neighbor's firms.
        potential_firms = neighbor_firms + startup

        # Calculate the time average growth rate of each firm.
        g_f = [
            self.calc_time_avg_growth_rate(firm=firm, include_self=True)
            for firm in potential_firms
        ]
    
        # Get the index value of the firm with the highest growth rate.
        max_idx = g_f.index(max(g_f))
 
        return potential_firms[max_idx], g_f[max_idx]
    
    def firm_selection(self):
        """
        Performs the firm selection process for the worker, comparing the current employer's
        growth rate with potential new employers and moving to a new firm if advantageous.
        """

        curr_g = self.calc_time_avg_growth_rate(firm=self.get_employer())
        new_firm, new_g = self.rank_firms()

        if new_g > curr_g:
            self.employer.separate(self)
            new_firm.hire(self)
            self.update_employer(firm=new_firm)