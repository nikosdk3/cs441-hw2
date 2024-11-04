package nikos.cs441.hw2

import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.scalatest.funsuite.AnyFunSuite

class ModelCreationSpec extends AnyFunSuite {
  test("Model should be created with correct input, hidden, and output sizes") {
    val numInput = 100
    val numHidden = 50
    val numOutput = 10
    val model = Model.createModel(numInput, numHidden, numOutput)

    val lstmLayer = model.getLayer(0).conf().getLayer.asInstanceOf[LSTM]
    assert(lstmLayer.getNIn === numInput)
    assert(lstmLayer.getNOut === numHidden)

    val rnnOutputLayer = model.getLayer(1).conf().getLayer.asInstanceOf[RnnOutputLayer]
    assert(rnnOutputLayer.getNIn === numHidden)
    assert(rnnOutputLayer.getNOut === numOutput)
  }
}
