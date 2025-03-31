###########################################################################
# IMPORTS
###########################################################################

import os

from datetime import datetime

from .models.firm import Firm
from .models.market import Market

from .config import base_params

###########################################################################
# EXECUTE MODEL
###########################################################################

def main():
    """
    Main function to run the model.
    """

    # Initialize model with imported parameters.
    print("Initializing market.")
    model = Market(**base_params)

    # Run the model.
    print("Running model.")
    results = model.run_model()

    # Save model results.
    print("Model complete, saving results.")

    # Export the data collector objects.
    data_dir = os.path.join('..', 'data')
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Export the agent data.
    (
        model.agent_collector
        .get_agenttype_vars_dataframe(agent_type=Firm)
        .to_pickle(os.path.join(data_dir, f"get_agenttype_vars_dataframe_{now}.pkl"))
    )

    # Export the model data.
    (
        model.model_collector
        .get_model_vars_dataframe()
        .to_pickle(os.path.join(data_dir, f"get_model_vars_dataframe_{now}.pkl"))
    )

if __name__ == '__main__':
    main()