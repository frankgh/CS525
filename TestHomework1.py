import unittest
import numpy as np
import homework1_afguerrerohernan as hw1


class TestHomework1(unittest.TestCase):
    def setUp(self):
        self.A = np.matrix('0 -1;1 0')
        self.B = np.matrix('1 2;3 4')
        self.C = np.matrix('2 -1;-8 2')
        self.D = np.matrix('1 2 8 11 15;3 4 9 12 16;5 6 10 13 17;7 8 11 14 18;8 9 12 15 18')
        self.E = np.matrix('3 1;0 2')
        # self.A = np.array([[0, -1], [1, 0]])
        # self.B = np.array([[1, 2], [3, 4]])
        # self.C = np.array([[2, -1], [-8, 2]])
        self.x = np.array([6, 4, 3, 9])
        self.y = np.array([-1, 2, 1, -4])
        self.z = np.array([6, 2])

        # Results
        self.p1_result = np.matrix('1 1;4 4')
        self.p2_result = np.matrix('-5 -3;9 0')
        self.p3_result = np.matrix('2 -10;2 2')
        self.not_result = np.matrix('1 1;4 3')

    def tearDown(self):
        self.A = None
        self.B = None
        self.C = None
        self.x = None
        self.y = None
        self.p1_result = None
        self.p2_result = None
        self.p1_not_result = None

    def test_problem1(self):
        problem_1_result = hw1.problem1(self.A, self.B)
        self.assertTrue(np.array_equal(self.p1_result, problem_1_result))
        self.assertFalse(np.array_equal(self.not_result, problem_1_result))

    def test_problem2(self):
        problem_2_result = hw1.problem2(self.A, self.B, self.C)
        self.assertTrue(np.array_equal(self.p2_result, problem_2_result))
        self.assertFalse(np.array_equal(self.not_result, problem_2_result))

    def test_problem3(self):
        problem_3_result = hw1.problem3(self.A, self.B, self.C)
        self.assertTrue(np.array_equal(self.p3_result, problem_3_result))
        self.assertFalse(np.array_equal(self.not_result, problem_3_result))

    def test_problem4(self):
        problem_4_result = hw1.problem4(self.x[:, np.newaxis], self.y[:, np.newaxis])
        self.assertEquals(problem_4_result, -31)

    def test_problem5(self):
        self.assertTrue(np.array_equal(np.matrix('0 0;0 0'), hw1.problem5(self.A)))

    def test_problem6(self):
        problem_6_result = hw1.problem6(self.A)
        self.assertTrue(np.array_equal(np.matrix('1;1'), problem_6_result))

    def test_problem7(self):
        problem_7_result = hw1.problem7(self.B)
        self.assertTrue(np.allclose(np.array([[-2, 1], [1.5, -0.5]]), problem_7_result))

    def test_problem8(self):
        problem_8_result = hw1.problem8(self.B, self.z[:, np.newaxis])
        self.assertTrue(np.allclose(np.array([-10., 8.])[:, np.newaxis], problem_8_result))

        a = np.array([[3, 1], [1, 2]])
        b = np.array([9, 8])
        b = b[:, np.newaxis]
        x = hw1.problem8(a, b)
        self.assertTrue(np.allclose(np.array([2., 3])[:, np.newaxis], x))
        self.assertTrue(np.allclose(np.dot(a, x), b))

    def test_problem9(self):
        self.assertTrue(np.allclose(np.array([-9, 5]), hw1.problem9(self.B, self.z)))

    def test_problem10(self):
        problem_10_result = hw1.problem10(self.B, 4)
        self.assertTrue(np.allclose(np.matrix('5 2;3 8'), problem_10_result))

        problem_10_result = hw1.problem10(self.D, 4)
        self.assertTrue(
            np.allclose(np.matrix('5 2 8 11 15;3 8 9 12 16;5 6 14 13 17;7 8 11 18 18;8 9 12 15 22'), problem_10_result))

    def test_problem11(self):
        self.assertEquals(-1, hw1.problem11(self.A, 0, 1))

        self.assertEquals(1, hw1.problem11(self.D, 0, 0))
        self.assertEquals(2, hw1.problem11(self.D, 0, 1))
        self.assertEquals(8, hw1.problem11(self.D, 0, 2))
        self.assertEquals(11, hw1.problem11(self.D, 0, 3))
        self.assertEquals(15, hw1.problem11(self.D, 0, 4))

        self.assertEquals(3, hw1.problem11(self.D, 1, 0))
        self.assertEquals(4, hw1.problem11(self.D, 1, 1))
        self.assertEquals(9, hw1.problem11(self.D, 1, 2))
        self.assertEquals(12, hw1.problem11(self.D, 1, 3))
        self.assertEquals(16, hw1.problem11(self.D, 1, 4))

        self.assertEquals(5, hw1.problem11(self.D, 2, 0))
        self.assertEquals(6, hw1.problem11(self.D, 2, 1))
        self.assertEquals(10, hw1.problem11(self.D, 2, 2))
        self.assertEquals(13, hw1.problem11(self.D, 2, 3))
        self.assertEquals(17, hw1.problem11(self.D, 2, 4))

        self.assertEquals(7, hw1.problem11(self.D, 3, 0))
        self.assertEquals(8, hw1.problem11(self.D, 3, 1))
        self.assertEquals(11, hw1.problem11(self.D, 3, 2))
        self.assertEquals(14, hw1.problem11(self.D, 3, 3))
        self.assertEquals(18, hw1.problem11(self.D, 3, 4))

        self.assertEquals(8, hw1.problem11(self.D, 4, 0))
        self.assertEquals(9, hw1.problem11(self.D, 4, 1))
        self.assertEquals(12, hw1.problem11(self.D, 4, 2))
        self.assertEquals(15, hw1.problem11(self.D, 4, 3))
        self.assertEquals(18, hw1.problem11(self.D, 4, 4))

    def test_problem12(self):
        self.assertEquals(37, hw1.problem12(self.D, 0))
        self.assertEquals(44, hw1.problem12(self.D, 1))
        self.assertEquals(51, hw1.problem12(self.D, 2))
        self.assertEquals(58, hw1.problem12(self.D, 3))
        self.assertEquals(62, hw1.problem12(self.D, 4))

    def test_problem13(self):
        self.assertEquals(4.5, hw1.problem13(self.D, 3, 6))
        self.assertEquals(2.5, hw1.problem13(self.B, 1, 4))
        self.assertEquals(-4.5, hw1.problem13(self.C, -10, 0))

    def test_problem14(self):
        self.assertTrue(np.allclose(np.matrix('0 0;0 1;1 0'), hw1.problem14(np.diag([1, 2, 3]), 2)))
        pass

    def test_problem15(self):
        column_vector = self.x[:, np.newaxis]
        problem15_result = hw1.problem15(column_vector, 2, 4, 5)
        self.assertEquals((4, 2), problem15_result.shape)


if __name__ == '__main__':
    unittest.main()
