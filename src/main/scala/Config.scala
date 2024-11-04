package nikos.cs441.hw2

import com.typesafe.config.ConfigFactory

object Config {
  // Load the configuration values from the default location: application.conf
  val conf = ConfigFactory.load()

  def embeddingDim = conf.getInt("config.embeddingDim")
  def batchSize = conf.getInt("config.batchSize")
  def seqLen = conf.getInt("config.seqLen")
  def stride = conf.getInt("config.stride")
  def hiddenLayerDim = conf.getInt("config.hiddenLayerDim")
  def numEpochs = conf.getInt("config.numEpochs")
  def trainTestSplit = conf.getDouble("config.trainTestSplit")
}