import unittest
from unittest.mock import MagicMock
from src.models.worker import Worker

class TestWorker(unittest.TestCase):
    def setUp(self):

        # Mock the model to pass to the Worker instance
        self.mock_model = MagicMock()
        self.mock_model.p = {'num_neighbors': 2}
        self.worker = Worker(self.mock_model)

        # Mock a Firm object for employment tests
        self.mock_firm = MagicMock()
        self.mock_firm.id = 1

    def test_initialization(self):
        """
        Test that a Worker initializes with expected default values.
        """

        self.assertAlmostEqual(self.worker.mu, 0, delta=0.01)
        self.assertGreaterEqual(self.worker.sigma, 0)
        self.assertEqual(self.worker.wealth, 1)
        self.assertIsNone(self.worker.employer)
        self.assertIsNone(self.worker.employer_id)

    def test_update_employer(self):
        """
        Test updating the worker's employer updates employer attributes correctly.
        """

        self.worker.update_employer(self.mock_firm)
        self.assertEqual(self.worker.employer, self.mock_firm)
        self.assertEqual(self.worker.employer_id, self.mock_firm.id)

    def test_produce(self):
        """
        Test the produce method calculates output based on the worker's attributes.
        """

        # Setting fixed mu and sigma for predictable output
        self.worker.mu = 0.01
        self.worker.sigma = 0.01

        # Mock random.normalvariate to return a fixed value
        with unittest.mock.patch('random.normalvariate', return_value=1):
            output = self.worker.produce()
            # Check if output is within expected bounds, given the mock value
            self.assertGreater(output, 0)

    def test_firm_selection_advantageous_move(self):
        """
        Test firm selection results in moving to a new firm if advantageous.
        """

        # Setup current and new firm with mock growth rates
        current_firm = MagicMock()
        new_firm = MagicMock()

        self.worker.employer = current_firm
        self.worker.calc_time_avg_growth_rate = MagicMock(side_effect=[0.01, 0.02])
        self.worker.select_empty_firms = MagicMock(return_value=[])
        self.worker.select_neighbor_firms = MagicMock(return_value=[new_firm])
        self.worker.rank_firms = MagicMock(return_value=(new_firm, 0.02))

        self.worker.firm_selection()

        # Verify the worker has moved to the new firm
        self.assertEqual(self.worker.employer, new_firm)

if __name__ == '__main__':
    unittest.main()
