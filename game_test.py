"""Tests for game.py."""

import unittest
import numpy as np
import game

class TestGame(unittest.TestCase):

  def test_pack_unpack_array(self):
    start_n = int(np.random.randint(100000))
    start_arr = np.random.randint(7, size=(5, 5))
    n = game.pack_array(start_arr, 3, start_n)
    arr, n = game.unpack_array(start_arr.shape, 3, n)
    assert n == start_n, (n, start_n)
    assert (arr == start_arr).all(), (arr, start_arr)

if __name__ == '__main__':
  unittest.main()
