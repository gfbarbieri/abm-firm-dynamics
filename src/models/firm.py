###########################################################################
# IMPORTS
###########################################################################

from agentpy import Agent

###############################################################################
# FIRM AGENT
###############################################################################

class Firm(Agent):
    """
    Represents a firm within an agent-based model. A firm can hire and separate
    workers, produce output, and distribute output among its workers.

    Attributes:
        workers (list): A list of worker agents associated with the firm.
        size (int): The number of workers currently employed by the firm.
        output (float): The total output produced by the firm.
        output_per_worker (float): The output produced per worker.
    """
    
    ###########################################################################
    # INITIALIZATION
    ###########################################################################

    def setup(self):
        """
        Initializes the firm with default values for its attributes.
        """

        # Initialize the following variables: a list of workers, size of 
        # the firm, the growth rate of the firm, the output of the firm, and
        # the output per worker.
        self.workers = []
        self.size = 0
        self.output = 0
        self.output_per_worker = 0

    ###########################################################################
    # SETTERS AND GETTERS
    ###########################################################################

    def set_size(self):
        """
        Updates the size of the firm based on the number of workers.
        """

        self.size = len(self.workers)

    def update_output_per_worker(self, output):
        """
        Updates the output per worker.

        Parameters:
            output (float): The new output per worker to set.
        """

        self.output_per_worker = output

    def update_output(self, output):
        """
        Updates the firm's total output.

        Parameters:
            output (float): The new total output to set.
        """

        self.output = output

    def get_workers(self):
        """
        Returns the list of workers employed by the firm.

        Returns:
            list: The workers of the firm.
        """

        return self.workers

    def get_size(self):
        """
        Returns the size of the firm, i.e., the number of workers.

        Returns:
            int: The size of the firm.
        """

        return self.size
    
    def get_output(self):
        """
        Returns the firm's total output.

        Returns:
            float: The total output of the firm.
        """

        return self.output
    
    def get_output_per_worker(self):
        """
        Returns the output produced per worker.

        Returns:
            float: The output per worker.
        """

        return self.output_per_worker

    ###########################################################################
    # HIRING AND SEPARATIONS
    ###########################################################################

    def hire(self, worker):
        """
        Hires a new worker and adds them to the firm's list of workers.

        Parameters:
            worker: The worker agent to be hired by the firm.
        """

        self.workers.append(worker)
        self.set_size()

    def separate(self, worker):
        """
        Separates a worker from the firm, removing them from the list of workers.

        Parameters:
            worker: The worker agent to be separated from the firm.
        """

        self.workers.remove(worker)
        self.set_size()
    
    ###########################################################################
    # PRODUCTION AND DISTRIBUTION
    ###########################################################################
    
    def produce(self):
        """
        Calculates and updates the firm's total output based on the contributions
        of its workers. If the firm has no workers, the output is set to 0.
        """

        if not self.get_size():
            self.update_output(output=0)
        else:
            # Total output of the firm, which is the sum of all individual
            # contributions
            output = sum(
                [
                    worker.produce()
                    for worker in self.get_workers()
                ]
            )

            # Set output.
            self.update_output(output=output)
    
    def distribute(self):
        """
        Distributes the firm's total output among its workers. If the firm has no
        workers, the output per worker is set to 0. Otherwise, it calculates the
        output per worker and updates each worker's wealth accordingly.
        """

        if not self.get_size():
            self.update_output_per_worker(output=0)
        else:
            self.update_output_per_worker(
                output=(self.get_output() / self.get_size())
            )

            for worker in self.get_workers():
                worker.update_wealth(wealth=self.get_output_per_worker())