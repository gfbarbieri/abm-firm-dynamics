###########################################################################
# IMPORTS
###########################################################################

from agentpy import Model, AgentList
import math
import pandas as pd

from .firm import Firm
from .worker import Worker

###############################################################################
# MARKET MODEL
###############################################################################

class Market(Model):
    """
    Represents a market model in an agent-based simulation. This model
    facilitates interactions between worker and firm agents, simulating
    employment dynamics, production, and distribution processes within a
    market.

    The model initializes with a predefined number of workers and firms,
    assigns workers to firms, and simulates job-seeking, production,
    and wealth distribution through its lifecycle.
    """

    def setup(self):
        """
        Initializes the market model by creating worker and firm agents.
        Each worker is initially assigned to a unique firm. This method also
        initializes the networks between workers.
        """

        # Create worker and firm agents.
        self.workers = AgentList(self, self.p['n_workers'], Worker)
        self.firms = AgentList(self, 2*self.p['n_workers']+1, Firm)

        # Initialize each worker into a singleton firm.
        for i in range(self.p['n_workers']):
            worker = self.workers[i]
            firm = self.firms[i]

            # Assign the worker to the firm, update the firm's worker list.
            worker.update_employer(firm=firm)
            firm.hire(worker)

        # Initialize agent networks.
        self.workers.init_network()

    def step(self):
        """
        Executes the main dynamics of the market model during each simulation
        step. This includes workers seeking new employment opportunities and
        firms producing and distributing output.
        """

        # Randomly select workers to look for new jobs.
        self.workers.random(
            n=math.ceil(self.model.p['n_workers']*self.model.p['active'])
        ).to_list().firm_selection()

        # Produce and distribute output.
        self.firms.produce()
        self.firms.distribute()

    def update(self):
        """
        Updates the model after each simulation step. This involves updating
        each worker's employer size and collecting data on workers and firms
        for analysis.
        """
        
        # Update agent employer size.
        self.workers.set_employer_size()

        # Collect microdata.
        self.workers.record(
            ['wealth','wage','output','employer','employer_id','employer_size']
        )
        self.firms.record(['size','output','output_per_worker'])

    def end(self):
        """
        Finalizes the model simulation by recording initial attributes of
        workers. This method is called after the last simulation step to
        prepare the model's output data for analysis.
        """
        
        # Record initial attributes for workers in separate dataset and add it
        # to the model output when the simulation is complete.
        # self.model.output['initial_attributes'] = ap.DataDict({'Worker':
        # pd.DataFrame(data=data)})
        data = [
            {'id': worker.id, 'mu': worker.mu, 'sigma': worker.sigma, 'wealth': worker.wealth}
            for worker in self.workers
        ]
        self.model.output['initial_attributes'] = pd.DataFrame(data=data)

    ###############################################################################
    # DATA COLLECTION
    ###############################################################################
    # TODO: Build data collection functions, reducing the need to collect
    # microdata at every step. Ultimately, the goal is to reproduce the 
    # data analysis presented by Axtell (2016).
    # Incomplete list of data analysis from Axtell (2016):
    # Dynamics:
    #   1. Number of firms, new firms, exiting firms.
    #   2. Maximum and average firm size.
    #   3. Job-to-job changes, job creation, job destruction
    # Stationary:
    #   1. Firm sizes by employee and output. Graph firm size workers - probability
    #   and firm size output - probability. Potentially at just the end of the
    #   simulation.
    #   2. Labor productivity: firm output / employee. This is the
    #   "returns to sacle" parameter.
    #   3. Gross labor productivity by firm size.
    #   4. Job tenure.