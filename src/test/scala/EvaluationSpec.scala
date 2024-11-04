package nikos.cs441.hw2

import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.funsuite.AnyFunSuite

class EvaluationSpec extends AnyFunSuite {
  test("Evaluation accuracy should be calculated correctly") {
    val evaluation = new Evaluation(3) // Assume 3 classes
    val labels = Nd4j.create(Array(0.0, 1.0, 0.0)).reshape(1, 3) // Expected class
    val predictions = Nd4j.create(Array(0.0, 0.9, 0.1)).reshape(1, 3) // Predicted class

    evaluation.eval(labels, predictions)
    assert(evaluation.accuracy() === 1.0) // Should be 100% accurate
  }
}
