"""
Utility functions for data collection.
"""

########################################################################
## IMPORTS
########################################################################

import numpy as np

from typing import Literal

########################################################################
## DATA COLLECTION FUNCTIONS
########################################################################

def count_total_firms(model: object) -> int:
    """
    Counts the total number of active firms in the model. This captures
    the total number of firms in the model that are currently active.

    Parameters
    ----------
    model : object
        The model object.

    Returns
    -------
    int
        The total number of activefirms in the model.
    """

    # Get the total number of firms in the model.
    total_firms = len(model.firms.select(lambda x: x.size > 0))
    
    return total_firms

def count_new_firms(model: object) -> int:
    """
    Counts the number of new firms in the model.

    Parameters
    ----------
    model : object
        The model object.

    Returns
    -------
    int
        The number of new firms in the model.
    """

    # Get the number of new firms in the model.
    new_firms = len(model.firms.select(
        lambda x: x.curr_status == True and x.last_status == False
    ))

    return new_firms

def count_continuing_firms(model: object) -> int:
    """
    Counts the number of continuing firms in the model.

    Parameters
    ----------
    model : object
        The model object.

    Returns
    -------
    int
        The number of continuing firms in the model.
    """

    # Get the number of continuing firms in the model.
    continuing_firms = len(model.firms.select(
        lambda x: x.curr_status == True and x.last_status == True
    ))

    return continuing_firms

def count_exiting_firms(model: object) -> int:
    """
    Counts the number of exiting firms in the model.

    Parameters
    ----------
    model : object
        The model object.

    Returns
    -------
    int
        The number of exiting firms in the model.
    """

    # Get the number of exiting firms in the model.
    exiting_firms = len(model.firms.select(
        lambda x: x.curr_status == False and x.last_status == True
    ))

    return exiting_firms

def firm_size_percentile(
        model: object, percentile: int, active: bool=True
    ) -> float:
    """
    Returns the firm size distribution in the model. This captures the
    distribution of firm sizes in the model. It is calculated as:

    .. math::

       P(X \leq x)

    where :math:`X` is the firm size and :math:`x` is the percentile.

    Parameters
    ----------
    model : object
        The model object.
    percentile : int
        The percentile used to calculate the firm size distribution.
    active : bool, default=True
        Whether to select only the active firms.

    Returns
    -------
    float
        The firm size distribution in the model.
    """

    # Get the firms from the model with a size greater than 0.
    if active:
        firms = model.firms.select(lambda x: x.size > 0)
    else:
        firms = model.firms

    # Get the percentile of the firm sizes.
    percentile = np.percentile([x.size for x in firms], percentile)

    return percentile

def agg_agent_attrb(
        model: object, agent_type: Literal["firms", "workers"],
        attribute: str, method: str, active: bool=True
    ) -> float:
    """
    Returns the aggregate of the attribute for the agents in the model.
    This function uses the Numpy function to aggregate the attribute.
    The function is applied to the agents in the model. The method
    argument is the Numpy function to apply to the attribute.

    Parameters
    ----------
    model : object
        The model object.
    agent_type : Literal["firms", "workers"]
        The type of agent to aggregate the attribute.
    attribute : str
        The attribute to aggregate.
    method : str
        Valid Numpy function used to aggregate the attribute.
    active : bool, default=True
        Whether to select only the active firms.

    Returns
    -------
    float
        The aggregate of the attribute for the agents in the model.
    """

    # Select the agents to aggregate the attribute.
    if agent_type == "firms":

        # If active is True, select only the active firms.
        if active:
            agents = model.firms.select(lambda x: x.size > 0)
        else:
            agents = model.firms

    elif agent_type == "workers":

        # Select all workers.
        agents = model.workers

    # Get the function to apply to the agents.
    func = getattr(np, method)

    # Apply the function to the agents.
    result = agents.agg(attribute=attribute, func=func)

    return result
    
def avg_growth_rate(
        model: object, agent_type: Literal["firms", "workers"]
    ) -> float:
    """
    Returns the average growth rate of the agents in the model. This
    captures the average growth rate of the agents in the model. It is
    calculated as:

    .. math::

       \frac{1}{n} \sum_{i=1}^{n} g_i

    where :math:`g_i` is the growth rate of agent :math:`i` at time
    :math:`t`.

    Parameters
    ----------
    model : object
        The model object.
    agent_type : Literal["firms", "workers"]
        The type of agent to calculate the average growth rate.

    Returns
    -------
    float
        The average growth rate of the agents in the model.
    """

    # If selecting firms, select only those firms that are active (i.e.
    # have a size greater than 0). If selecting workers, select all
    # workers.
    if agent_type == "firms":

        # Select only the active firms.
        firms = model.firms.select(lambda x: x.size > 0)

        # Calculate the growth rate of the firms at the current step.
        growth_rates = [x.calc_time_avg_growth_rate() for x in firms]

    elif agent_type == "workers":

        # Select all workers.
        agents = model.workers

        # Get the growth rate of the workers.
        growth_rates = [x.g for x in agents]    

    # Calculate the average growth rate.
    avg_growth_rate = np.mean(growth_rates)

    return avg_growth_rate

def largest_firm_growth_rate(model: object) -> float:
    """
    Returns the growth rate of the largest firm in the model.
    
    This captures the growth rate of the most productive firm in the
    economy. It is calculated as:

    .. math::

       \max(g_1, g_2, \ldots, g_n)

    where :math:`g_i` is the growth rate of firm :math:`i` at time
    :math:`t`.

    Parameters
    ----------
    model : object
        The model object.

    Returns
    -------
    float
        The growth rate of the largest firm in the model.
    """

    # Get the firms from the model.
    firms = model.firms.select(lambda x: x.size > 0)

    # Get the maximum growth rate of the firms.
    max_growth_rate = np.max([x.calc_time_avg_growth_rate() for x in firms])

    return max_growth_rate

def job_creation(model: object, proportions: bool=True) -> float:
    """
    Returns the number of job creations in the model.

    Job creation represents the number of new jobs added to the
    economy in a given time period. This captures net hiring by firms
    that are expanding their workforce. It is calculated as:

    .. math::

       JC_t = \sum_{f \in F} \max(0, E_{f,t} - E_{f,t-1})

    where :math:`JC_t` is the number of job creations at time :math:`t`,
    :math:`F` is the set of firms, :math:`E_{f,t}` is the number of
    employees of firm :math:`f` at time :math:`t`, and :math:`t` is the
    current time step.

    Parameters
    ----------
    model : object
        The model object.
    proportions : bool, default=True
        Whether to return the proportions of job creations.

    Returns
    -------
    float
        The number of job creations in the model.
    """

    # Calculate the maximum of either 0 or the net job changes for each
    # firm.
    job_creation = np.sum([
        max(0, x) for x in model.firms.get(attr_names='net_change') if x > 0
    ])

    # If proportions is True, return the proportions of job creations.
    if proportions == True:
        job_creation = job_creation / len(model.workers)

    return job_creation

def job_destruction(model: object, proportions: bool=True) -> float:
    """
    Returns the number of job destructions in the model.

    Job destruction represents the number of jobs lost, meaning firms
    that have reduced employment. This captures net job losses from
    firms that are shrinking. It is calculated as:

    .. math::

       JD_t = \sum_{f \in F} \max(0, E_{f,t-1} - E_{f,t})

    where :math:`JD_t` is the number of job destructions at time
    :math:`t`, :math:`F` is the set of firms, :math:`E_{f,t}` is the
    number of employees of firm :math:`f` at time :math:`t`, and
    :math:`t` is the current time step.

    Parameters
    ----------
    model : object
        The model object.
    proportions : bool, default=True
        Whether to return the proportions of job destructions.

    Returns
    -------
    float
        The number of job destructions in the model.
    """

    # Calculate the maximum of either 0 or the net job changes for each
    # firm.
    job_destruction = np.sum([
        max(0, abs(x))
        for x in model.firms.get(attr_names='net_change') if x < 0
    ])

    # If proportions is True, return the proportions of job destructions.
    if proportions == True:
        job_destruction = job_destruction / len(model.workers)

    return job_destruction

def job_changes(model: object, proportions: bool=True) -> float:
    """
    Returns the number of job changes in the model.

    Job-to-job changes measure how often workers switch employers in a
    given time period. This represents employer-to-employer transitions
    without passing through unemployment. It is calculated as:

    .. math::

       \Delta E_t = JC_t - JD_t
    
    where :math:`\Delta E_t` is the number of job changes at time
    :math:`t`, :math:`JC_t` is the number of job creations at time
    :math:`t`, and :math:`JD_t` is the number of job destructions at time
    :math:`t`.

    Parameters
    ----------
    model : object
        The model object.
    proportions : bool, default=True
        Whether to return the proportions of job changes.

    Returns
    -------
    float
        The number of job changes in the model.
    """

    # Get the number of job changes. The job changes are stored in the 
    # `switch` attribute of the workers.
    job_changes = np.sum([
        x for x in model.workers.get(attr_names='switch')
    ])

    # If proportions is True, return the proportions of job changes.
    if proportions == True:
        job_changes = job_changes / len(model.workers)

    return job_changes