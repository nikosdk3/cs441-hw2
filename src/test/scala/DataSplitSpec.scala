package nikos.cs441.hw2

import org.scalatest.funsuite.AnyFunSuite

class DataSplitSpec extends AnyFunSuite {
  test("Dataset should be split correctly into training and testing datasets") {
    val data = Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    val (train, test) = Utils.splitDataset(data, 0.6)

    assert(train.length === 6)
    assert(test.length === 4)
    assert(train sameElements Array(1, 2, 3, 4, 5, 6))
    assert(test sameElements Array(7, 8, 9, 10))
  }
}
