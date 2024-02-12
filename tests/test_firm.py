import unittest
from unittest.mock import MagicMock
from src.models.firm import Firm
from src.models.worker import Worker

class TestFirm(unittest.TestCase):

    def setUp(self):
        """
        Set up a Firm instance before each test method.
        """

        self.mock_model = MagicMock()
        self.firm = Firm(self.mock_model)

    def test_initialization(self):
        """
        Test that a Firm is initialized with correct default values.
        """

        self.assertEqual(self.firm.size, 0, "Initial size should be 0")
        self.assertEqual(self.firm.output, 0, "Initial output should be 0")
        self.assertEqual(self.firm.output_per_worker, 0, "Initial output per worker should be 0")
        self.assertEqual(len(self.firm.workers), 0, "Initial workers list should be empty")

    def test_hire(self):
        """
        Test hiring a worker correctly updates the firm's workers list and size.
        """

        worker = Worker(self.mock_model)
        self.firm.hire(worker)
        self.assertIn(worker, self.firm.workers, "Worker should be in the firm's workers list")
        self.assertEqual(self.firm.size, 1, "Firm size should be updated to 1")

    def test_separate(self):
        """
        Test separating a worker correctly updates the firm's workers list and size.
        """
        
        worker = Worker(self.mock_model)
        self.firm.hire(worker)
        self.firm.separate(worker)
        self.assertNotIn(worker, self.firm.workers, "Worker should be removed from the firm's workers list")
        self.assertEqual(self.firm.size, 0, "Firm size should be updated to 0 after separation")

    def test_produce_and_distribute(self):
        """
        Test that produce and distribute correctly updates output and output per worker.
        """
        
        # Assuming the Worker class has a produce method that returns
        # a fixed output. Mocking produce method
        worker1 = Worker(self.mock_model)
        worker1.produce = lambda: 10
        self.firm.hire(worker1)

        # Mocking produce method
        worker2 = Worker(self.mock_model)
        worker2.produce = lambda: 20
        self.firm.hire(worker2)

        self.firm.produce()
        self.firm.distribute()

        # Assuming output is the sum of workers' production
        expected_output = 30
        expected_output_per_worker = expected_output / 2

        self.assertEqual(self.firm.output, expected_output, "Firm's output should equal the sum of workers' production")
        self.assertEqual(self.firm.output_per_worker, expected_output_per_worker, "Output per worker should be correctly calculated")

if __name__ == '__main__':
    unittest.main()