########################################################################
## IMPORTS
########################################################################

from mesa import Agent

########################################################################
## FIRM AGENT
########################################################################

class Firm(Agent):
    """
    Represents a firm within an agent-based model. A firm can hire and
    separate workers, produce output, and distribute output among its
    workers.

    Attributes
    ----------
    workers
        A list of worker agents associated with the firm.
    size
        The number of workers currently employed by the firm.
    output
        The total output produced by the firm.
    """

    def __init__(self, model: object) -> None:
        """
        Initializes the firm with default values for its attributes.

        Parameters
        ----------
        model
            The model object to pass to the Agent class.
        """

        # Pass the model object to the Agent class.
        super().__init__(model)

        # Create attributes for the firm.
        self.workers = []
        self.size = 0
        self.output = 0
        self.output_per_worker = 0

        # Record the firm's status. The firm is inactive if it has no
        # workers. All firms are inactive at initialization.
        self.curr_status = False
        self.last_status = False

        # Record net change in the number of workers in the firm.
        self.net_change = 0

    def offer(self, worker: object) -> bool:
        """
        If the growth rate of the firm goes up with the worker, then
        offer the worker the opportunity to join the firm.

        Parameters
        ----------
        worker:
            The worker agent as a Mesa agent object.

        Returns
        -------
        bool
            True if the worker should be hired, False otherwise.
        """

        # Calculate current time average growth rate of the firm, i.e.
        # without the worker. If the firm is empty, then the growth rate
        # is None.
        curr_g = self.calc_time_avg_growth_rate()

        # Calculate the time average growth rate of the firm with the
        # worker.
        new_g = self.calc_time_avg_growth_rate(add_workers=[worker])

        # For the purposes of making an offer, None and 0 will always
        # result in the offer being made. If the growth rate of the firm
        # with the worker is greater than the growth rate of the firm
        # without the worker, then offer the worker the opportunity to
        # join the firm.
        if curr_g is None:
            return True
        elif new_g > curr_g:
            return True
        else:
            return False

    def calc_time_avg_growth_rate(
            self, add_workers: list[Agent]=[]
        ) -> float | None:
        """
        Calculates the time-averaged growth rate of wealth for a group
        of workers.

        Parameters
        ----------
        add_workers
            Additional workers to add to the calculation.

        Returns
        -------
        float
            The calculated time-averaged growth rate.
        """

        # Get list of workers from firms.
        workers = self.workers
    
        # If a list of workers are passed, calculate the growth rate of
        # wealth for the group as a cooperating unit.
        if add_workers:
            workers = workers + add_workers

        # If the number of workers is zero, then the growth rate cannot
        # be calculated and a ZeroDivisionError will be raised in the
        # best case scenario. In this case, the correct answer is to
        # return None and let the user of the function handle the
        # results.
        if len(workers) == 0:
            g = None
        else:
            # Contribution of each worker in the list.
            g_i = [
                (worker.mu - (worker.sigma ** 2) / (2 * len(workers)))
                for worker in workers
            ]

            # Growth rate based on total contributions of workers.
            g = sum(g_i) / len(workers)

        return g

    def hire(self, worker: Agent) -> None:
        """
        Hires a new worker and adds them to the firm's list of workers.

        Parameters
        ----------
        worker
            The worker agent to be hired by the firm as a Mesa agent
            object.

        Raises
        ------
        ValueError
            If the worker is already in the firm's worker list.
        """

        # If the worker is not already in the firm's worker list, add
        # them and update the size of the firm.
        if worker not in self.workers:
            self.workers.append(worker)
            self.size = len(self.workers)
            self.net_change += 1
        else:
            raise ValueError("Worker already in firm's worker list.")

    def separate(self, worker: Agent) -> None:
        """
        Separates a worker from the firm, removing them from the list of
        workers.

        Parameters
        ----------
        worker
            The worker agent to be separated from the firm as a Mesa
            agent object.

        Raises
        ------
        ValueError
            If the worker is not in the firm's worker list.
        """
    
        # If the worker is in the firm's worker list, remove them and
        # update the size of the firm.
        if worker in self.workers:
            self.workers.remove(worker)
            self.size = len(self.workers)
            self.net_change -= 1
        else:
            raise ValueError("Worker not found in firm's worker list.")

    def produce(self) -> None:
        """
        Calculates and updates the firm's total output based on the
        contributions of its workers. If the firm has no workers, the
        output is set to 0.
        """

        # Total output of the firm, which is the sum of all
        # individual contributions. If the firm has no workers, then
        # the output is set to 0.
        self.output = sum([worker.produce() for worker in self.workers])

    def distribute(self) -> None:
        """
        Distributes the firm's total output among its workers. If the
        firm has no workers, the output per worker is set to 0.
        Otherwise, it calculates the output per worker and updates each
        worker's wealth accordingly.
        """

        # Output per worker is the total output divided by the number of
        # workers. If the firm has no workers, the output per worker is
        # set to 0.
        if self.size == 0:
            self.output_per_worker = 0
        else:
            self.output_per_worker = self.output / self.size

        # Update each worker's wealth by the output per worker.
        for worker in self.workers:
            worker.update_wealth(wealth=self.output_per_worker)

    def update_status(self) -> None:
        """
        Updates the status of the firm after the model's step method is
        called. This is used to record the firm's status at the end of
        each model step.
        
        The assumption is that this function is only called once for
        each firm and at the end of each model step. If you call it
        more than once in the same step, then the status will be
        incorrect.

        Notes
        -----
        A firm is either active or inactive. There are four possible
        combinations of activity from t-1 -> t.

        If the firm was active in t-1 and is inactive (active) in t,
        then the firm exited (continued to be active in) the market. If
        the firm was inactive in t-1 and is active (inactive) in t, then
        the firm entered (continued to be inactive in) the market.
        """

        # Set the last status to the current status.
        self.last_status = self.curr_status

        # Set the current status to the new status.
        if self.size > 0:    
            self.curr_status = True
        elif self.size == 0:
            self.curr_status = False
        else:
            raise ValueError(f"Firm size is negative: {self.size}.")