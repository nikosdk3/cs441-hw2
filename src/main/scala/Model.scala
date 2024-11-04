package nikos.cs441.hw2

import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam

object Model {
  def createModel(numInput: Int, numHidden: Int, numOutput: Int): MultiLayerNetwork = {
    // Build the configuration
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .updater(new Adam(0.001)) // Learning rate
      .list()
      .layer(0, new LSTM.Builder()
        .nIn(numInput) // Input layer size
        .nOut(numHidden) // Number of units in the hidden layer
        .activation(Activation.RELU) // Activation function
        .build())
      .layer(1, new RnnOutputLayer.Builder()
        .nIn(numHidden) // Input from hidden layer
        .nOut(numOutput) // Output size (vocabulary size)
        .activation(Activation.SOFTMAX) // Softmax for multi-class classification
        .build())
      .build()

    val net = new MultiLayerNetwork(conf)
    net.init()
    net
  }
}
