import unittest
from unittest.mock import MagicMock, patch
from src.models.market import Market

class TestMarket(unittest.TestCase):
    def setUp(self):
        """
        Set up a Market instance before each test method.
        """

        # Mock parameters needed for the Market model initialization
        self.model_params = {'n_workers': 10, 'active': 0.5}
        with patch('src.models.market.Model.__init__', return_value=None) as mock_model_init:
            self.market = Market()
            self.market.p = self.model_params
            self.market.setup()

    def test_setup(self):
        """
        Test market setup initializes workers and firms correctly.
        """

        self.assertEqual(len(self.market.workers), self.model_params['n_workers'])
        self.assertEqual(len(self.market.firms), 2*self.model_params['n_workers']+1)
        # Verify each worker is assigned to a firm
        for worker in self.market.workers:
            self.assertIsNotNone(worker.employer)

    def test_step(self):
        """
        Test market step function triggers firm selection and production.
        """

        with patch.object(self.market.workers, 'random') as mock_random:
            with patch.object(self.market.firms, 'produce') as mock_produce, patch.object(self.market.firms, 'distribute') as mock_distribute:
                self.market.step()
                # Verify that workers' firm selection, and firms' production and distribution are called
                mock_random.assert_called()
                mock_produce.assert_called()
                mock_distribute.assert_called()

    def test_update(self):
        """
        Test market update function collects data correctly.
        """

        with patch.object(self.market.workers, 'record') as mock_workers_record, patch.object(self.market.firms, 'record') as mock_firms_record:
            self.market.update()
            # Verify that data recording is attempted for both workers and firms
            mock_workers_record.assert_called()
            mock_firms_record.assert_called()

    def test_end(self):
        """
        Test market end function finalizes the simulation output correctly.
        """
        
        self.market.end()
        # Verify initial attributes data frame is created in the model's output
        self.assertIn('initial_attributes', self.market.model.output)
        self.assertEqual(len(self.market.model.output['initial_attributes']), len(self.market.workers))

if __name__ == '__main__':
    unittest.main()
