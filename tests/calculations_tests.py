import unittest


class TestDecayingFunctions(unittest.TestCase):

    def test_linear_decay(self):
        from calculations import linear_decay

        test_sigmas = [1,  5,  10,  105]
        test_lambdas = [-0.00005, -0.75, -1, -10]

        for sigma, lambd in zip(test_sigmas, test_lambdas):
            for i in range(100):
                actual = linear_decay(i, sigma, lambd)

            expected_value = linear_decay(100-1, sigma, lambd)

            self.assertEqual(expected_value, actual, "The function returned unexpected value...")
            self.assertTrue( actual < sigma, "the value did not decay. (actual: %f < sigma: %f" %(actual, sigma))

    def test_exponential_decay(self):
        from calculations import exponential_decay

        test_sigmas = [1,  5,  10,  105]
        test_lambdas = [0.00005, 0.75, 1, 10]

        for sigma, lambd in zip(test_sigmas, test_lambdas):
            for i in range(100):
                actual = exponential_decay(i, sigma, lambd)

            expected_value = exponential_decay(100-1, sigma, lambd)

            self.assertEqual(expected_value, actual, "The function returned unexpected value...")
            self.assertTrue( actual < sigma, "the value did not decay. (actual: %f < sigma: %f" %(actual, sigma))

class TestAdjustmentFunctions(unittest.TestCase):
    # TODO: Write the tests.
    pass

class TestNeighbourhoodFunctions(unittest.TestCase):
    # TODO: Write the tests.
    pass
