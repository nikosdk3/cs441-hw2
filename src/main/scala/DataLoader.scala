package nikos.cs441.hw2


import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.deeplearning4j.models.word2vec.Word2Vec
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory

class DataLoader(val vocabSize: Long, val embeddings: INDArray, val batchSize: Int, val seqLen: Int, val stride: Int, spark: SparkSession) extends Serializable {
  Nd4j.getRandom.setSeed(123)
  private val logger = LoggerFactory.getLogger(getClass)
  private val positionalEmbedding = Nd4j.rand(seqLen, Config.embeddingDim)

  def slidingWindowsWithEmbeddings(data: Array[Int]): RDD[DataSet] = {
    // Create a sequence to hold the windows
    val windows = spark.sparkContext.parallelize(0 until (data.length - seqLen) by stride).map { i =>
      val input = data.slice(i, i + seqLen)
      val target = data(i + seqLen)

      // Embed token and add positional embedding to capture context
      val embeddedInput = Utils.computePositionalEmbedding(Utils.embed(input, embeddings), positionalEmbedding)

      // Reshape to match model input size
      embeddedInput.transpose()
      val inputArray = embeddedInput.reshape(1, Config.embeddingDim, seqLen)

      // Create the target array as one-hot encoded vector across all sequence steps
      val targetArray = Nd4j.zeros(1, vocabSize, seqLen)  // Set sequence length to match input
      for (j <- 0 until seqLen) {
        targetArray.putScalar(Array(0, target, j), 1.0) // Repeat target across sequence length
      }

      new DataSet(inputArray, targetArray) // Add each input and target to a dataset
    }

    logger.info(s"${windows.count()} windows created")

    windows
  }
}
