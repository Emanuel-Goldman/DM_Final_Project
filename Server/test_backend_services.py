import unittest
from backend_services import find_feasible_angle_region

# FILE: test_backend_services.py


class TestFindFeasibleAngleRegion(unittest.TestCase):


    def test_feasible_region_1(self):
        constraints = [
            (1, 2, "<="),  # W1 <= 2W2
            (1, 1, ">="),  # W1 >= W2
        ]
        result = find_feasible_angle_region(constraints)
        self.assertEqual(result, (45, 63.43494882292201))



if __name__ == '__main__':
    unittest.main()