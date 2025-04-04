########################################################################
## IMPORTS
########################################################################

import math
import numpy as np

from mesa import Model
from mesa.agent import AgentSet
from mesa.datacollection import DataCollector

from .firm import Firm
from .worker import Worker

from ..utilities.data_collection import (
    count_total_firms, count_new_firms, count_continuing_firms,
    count_exiting_firms, firm_size_percentile, avg_growth_rate,
    agg_agent_attrb, largest_firm_growth_rate, job_changes,
    job_creation, job_destruction
)

########################################################################
## MARKET MODEL
########################################################################

class Market(Model):
    """
    Represents the market within an agent-based model. The market
    handles interactions between firms and workers, including hiring,
    firing, and wage setting.
    """

    def __init__(
            self, num_agents: int, activation: float,
            num_steps: int | None=None, mutual_acceptance: bool=True,
            global_search_rate: float=0.01, constant_mu: float | None=None,
            constant_sigma: float | None=None, track_wealth: bool=True,
            correlation_matrix: dict[str, bool] | None=None, seed: int=None,
            positive_g: bool=True, num_neighbors: int | None=None
        ) -> None:
        """
        Initializes the market model with workers and firms. Workers are
        placed in a firm and assigned a social network of workers.

        Parameters
        ----------
        num_agents
            The number of agents in the model.
        activation
            The activation rate of the workers.
        num_steps
            The number of steps to run the model for. If None, the
            model will run until it converges.
        mutual_acceptance
            Whether firms must accept workers before they can join. If
            True, firms will only accept workers that increase their
            growth rate.
        global_search_rate
            The global search rate of the workers.
        constant_mu : float | None
            The constant mu vlaue for the worker. IF None, then the mu
            is set to a random value that when combined with a random
            sigma ensures the worker's growth rate is positive.
        constant_sigma
            A sigma value to pass to all workers to ensure they are
            identical. If None, the sigma is set to a random value.
        track_wealth
            Whether the wealth of the workers is updated. If True, the
            wealth is updated based on the growth rate and the sigma at
            the end of each step.
        correlation_matrix
            A dictionary containing the parameters for the correlation
            matrix. If None, the correlation matrix is not built.
        seed
            The random seed for the model.
        """

        # Initialize the base model object.
        super().__init__(seed=seed)

        # Set the activation rate of the workers to search for jobs.
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.activation = activation

        # Create worker and firm agents. This adds the agents to the
        # model's agent attribute.
        Worker.create_agents(
            model=self, n=num_agents, mutual_acceptance=mutual_acceptance,
            global_search_rate=global_search_rate, track_wealth=track_wealth,
            constant_mu=constant_mu, constant_sigma=constant_sigma,
            positive_g=positive_g, num_neighbors=num_neighbors
        )
        Firm.create_agents(model=self, n=2*num_agents+1)

        # Initialize agent networks.
        self.workers.do('init_network')

        # Initialize the state of the economy as one where each worker
        # is employed by a singleton firm.
        for i in range(num_agents):

            # Get the worker and firm agents. This replaces the firm
            # selection process to ensure that each worker is employed
            # by a singleton firm.
            worker = self.workers[i]
            firm = self.firms[i]
            
            # Change the worker's employer to the firm. This includes
            # separating the worker from any previous employer and
            # hiring the worker by the firm.
            worker.change_employer(firm=firm)

            # Update firm's status to record the firm's activity.
            firm.update_status()

        # After workers have been initialized into firms, we can begin
        # to build the correlation matrix between all workers. The
        # correlation matrix is a square matrix where the element at
        # row i and column j is the correlation between the ith and jth
        # workers. These correlations are used to in the calculation of
        # growth rates.
        if correlation_matrix is None:
            correlation_matrix = {
                'randomize': False,
                'randomize_diagonal': False
            }

        self.build_correlation_matrix(**correlation_matrix)

        # Set up data collection. While a data collection class can
        # accept reporters for different levels, we are instead going to
        # create a different data collector for the model-level data and
        # the agent-level data. This allows us to use the current model
        # step to control data collection at different levels.
        # 
        # For example, the model-level data will be collected for each
        # model step. But agent level data should really only be
        # collected once at at the end as they reflect "stationary"
        # points in the model.
        # 
        # Model Level Reports: Setup after agents are created and fully
        # fully initialized. We are importing functions from a utility
        # module to collect data from the model. In that case, we need
        # to pass the model as an argument to the function. The Mesa
        # documentation says we must pass the function and any
        # parameters as a list to the correct reporter.
        self.model_collector = DataCollector(
            model_reporters={
                "steps": "steps",
                "total_firms": [count_total_firms, [self]],
                "new_firms": [count_new_firms, [self]],
                "continuing_firms": [count_continuing_firms, [self]],
                "exiting_firms": [count_exiting_firms, [self]],
                "median_firm_size": [firm_size_percentile, [self, 0.50]],
                "avg_firm_size": [
                    agg_agent_attrb, [self, 'firms', 'size', 'mean']
                ],
                "max_firm_size": [
                    agg_agent_attrb, [self, 'firms', 'size', 'max']
                ],
                "avg_worker_growth_rate": [avg_growth_rate, [self, 'workers']],
                "avg_firm_growth_rate": [avg_growth_rate, [self, 'firms']],
                "largest_firm_growth_rate": [largest_firm_growth_rate, [self]],
                "job_searches": [
                    agg_agent_attrb, [self, 'workers', 'search', 'sum']
                ],
                "job_offers": [
                    agg_agent_attrb, [self, 'workers', 'offer', 'sum']
                ],
                "job_changes": [job_changes, [self]],
                "job_creation": [job_creation, [self]],
                "job_destruction": [job_destruction, [self]],
            }
        )

        # Agent Level Reporter: Collects data for each firm at the
        # end of the model run.
        self.agent_collector = DataCollector(
            agenttype_reporters={
                Firm: {
                    "size": "size",
                    "output": "output",
                    "output_per_worker": "output_per_worker"
                },
                Worker: {
                    "mu": "mu",
                    "sigma": "sigma",
                    "employer_id": "employer_id",
                    "job_history": "job_history"
                }
            }
        )

    def step(self):
        """
        Executes the main dynamics of the market model during each
        simulation step. This includes workers seeking new employment
        opportunities and firms producing and distributing output.
        """

        # Reset the worker job search attributes; firm net job change
        # counters.
        self.workers.set(attr_name='search', value=False)
        self.workers.set(attr_name='offer', value=False)
        self.workers.set(attr_name='switch', value=False)
        self.firms.set(attr_name='net_change', value=0)

        # Randomly select workers to look for new jobs using the
        # activation rate. Take the ceiling of the product of the
        # number of agents and the activation rate to ensure that
        # at least one worker is selected.
        sel_workers = AgentSet(
            agents=self.random.sample(
                population=self.workers,
                k=math.ceil(self.num_agents*self.activation)
            ),
            random=self.random
        )

        # Set the worker's search attribute to True.
        sel_workers.set(attr_name='search', value=True)

        # Have selected workers search for a new employer. Activate the
        # workers randomly by shuffling the selected workers.
        sel_workers.shuffle_do('select_firm')

        # After labor market churn is complete: the firms produce,
        # distribute their output to their workers, and update their
        # status.
        self.firms.do('produce')
        self.firms.do('distribute')
        self.firms.do('update_status')

    def run_model(self) -> None:
        """
        Runs the model for a given number of steps or until it
        converges.
        """

        # If the number of steps is not None, run the model for the
        # given number of steps.
        if self.num_steps is not None:

            self.step_model()

        # If the number of steps is None, run the model until it
        # converges.
        else:

            self.converge_model()

    def converge_model(self) -> None:
        """
        Runs the model until it converges.
        """
        
        # Run the model until it converges.
        while self.running:

            # Perform a step of the model.
            self.step()

            # Collect the data.
            self.model_collector.collect(self)

            # Check if the model has converged.
            self.running = self.check_mutual_trades()

        # If the model has converged, collect the agent-level data.
        self.agent_collector.collect(self)

    def step_model(self) -> None:
        """
        Runs the model for a given number of steps.
        """

        # Run the model for the given number of steps.
        for i in range(self.num_steps):

            # Perform a step of the model.
            self.step()

            # Collect the data.
            self.model_collector.collect(self)

            # If we are at the last step, collect the agent-level data.
            # This is because the agent-level data is meant to capture
            # the stationary state of the model, which we will treat
            # as the final state of the model.
            if i == self.num_steps - 1:
                self.agent_collector.collect(self)

    def build_correlation_matrix(
            self, randomize: bool=True, randomize_diagonal: bool=False
        ) -> None:
        """
        Builds the correlation matrix between all workers.

        The correlation matrix is a square matrix where the element at
        row i and column j is the correlation between the ith and jth
        workers. The base correlation matrix is the uncorrelated,
        matrix with zeroes everywhere except for the diagonal where
        the correlation is 1 (every agent is fully correlated with
        themselves). Otherwise, the zeroes are replaced with a random
        number between -1 and 1, but the diagonal is always 1. Another
        option is to also replace the diagonal with random values.

        Parameters
        ----------
        randomize : bool
            Whether to randomize the correlation between workers.
        randomize_diagonal : bool
            Whether to randomize the correlation of workers with
            themselves.
        """

        # There are four possible configurations, but only two are
        # relevant to the model.
        # 1. The correlation matrix is not randomized and the diagonal
        #    is not randomized.
        # 2. The correlation matrix is not randomized and the diagonal
        #    is randomized.
        # 3. The correlation matrix is randomized and the diagonal is
        #    not randomized.
        # 4. The correlation matrix is randomized and the diagonal is
        #    randomized.

        # Configuration 1: The correlation matrix is not randomized and
        # the diagonal is not randomized. This returns the identity
        # matrix.
        if not randomize and not randomize_diagonal:

            # Build the base correlation matrix as the identity matrix.
            self.correlation_matrix = np.eye(self.num_agents)

        # Configuration 2: The correlation matrix is randomized and
        # the diagonal is not randomized. This returns a matrix where
        # the diagonal is 1 and the rest are random numbers between -1
        # and 1.
        elif randomize and not randomize_diagonal:

            # Build the correlation matrix where diagnoals are 1 and the
            # rest are random numbers between -1 and 1.
            self.correlation_matrix = np.random.uniform(
                low=-1, high=1, size=(self.num_agents, self.num_agents)
            )

            # Fill the diagonal with 1s.
            np.fill_diagonal(self.correlation_matrix, 1)

        # Configuration 3: The correlation matrix is not randomized and
        # the diagonal is randomized. This returns a matrix where the
        # diagonal is random numbers between -1 and 1 and the rest are
        # 0s.
        elif not randomize and randomize_diagonal:

            # Build the correlation matrix where the diagonal is
            # random numbers between -1 and 1 and the rest are 0s.
            self.correlation_matrix = np.random.uniform(
                low=0, high=0, size=(self.num_agents, self.num_agents)
            )

            # Fill the diagonal with random numbers between -1 and 1.
            np.fill_diagonal(
                self.correlation_matrix,
                np.random.uniform(low=-1, high=1, size=self.num_agents)
            )

        # Configuration 4: The correlation matrix is randomized and the
        # diagonal is randomized. This returns a matrix where the
        # diagonal is random numbers between -1 and 1 and the rest are
        # random numbers between -1 and 1.
        elif randomize and randomize_diagonal:

            # Build the correlation matrix where all elements are a
            # random number between -1 and 1.
            self.correlation_matrix = np.random.uniform(
                low=-1, high=1, size=(self.num_agents, self.num_agents)
            )

    def check_mutual_trades(self) -> bool:
        """
        Checks if any mutually beneficial trades exist between workers
        and firms.

        A trade is mutually beneficial if:
        1. A firm would accept a worker (worker increases firm's growth
        rate)
        2. Worker would accept firm (firm offers better growth rate than
        current employer)
        
        Returns
        -------
        bool
            True if no mutually beneficial trades exist (convergence
            achieved). False if at least one mutually beneficial trade
            exists.
        """

        # For each worker and firm, check if there is a mutually
        # beneficial trade.
        for worker in self.workers:

            # Get the growth rate of the worker's current firm.
            curr_g = worker.get_worker_firms(
                workers=[worker]
            )[0].calc_time_avg_growth_rate()

            # Get all firms except the worker's current employer, and
            # add a singleton firm.
            firms = self.firms.select(
                lambda x: x.unique_id != worker.employer_id and x.size > 0
            )
            firms.add(self.firms.select(lambda x: x.size == 0, at_most=1)[0])

            # For each firm, check if the firm would accept the worker.
            # If so, check if the worker would join the firm. If both
            # are true, then a mutually beneficial trade exists. Else,
            # no trade exists.
            for firm in firms:

                # Check if firm would accept worker.
                offer = firm.offer(worker)

                # If the firm would accept the worker, check if the
                # worker would join the firm.
                if offer == True:

                    # Calculate potential growth rate with worker.
                    new_g = firm.calc_time_avg_growth_rate(
                        add_workers=[worker]
                    )
                    
                    # Check if worker would join firm.
                    if new_g > curr_g:
                        return True

        return False

    @property
    def workers(self) -> AgentSet:
        """
        Get the worker agents from the model.
        """
        return self.agents_by_type[Worker]
    
    @property
    def firms(self) -> AgentSet:
        """
        Get the firm agents from the model.
        """
        return self.agents_by_type[Firm]