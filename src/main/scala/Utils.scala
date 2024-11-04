package nikos.cs441.hw2

import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import java.io.{File, PrintWriter}

object Utils {
  def writeToFile(trainingReport: TrainingReport): Unit = {
    val writer = new PrintWriter(new File("Training Report"))
    try {
      // Write epoch statistics
      writer.println("Training Statistics:")
      writer.println(s"${trainingReport.epochStats}")
      writer.println(s"${trainingReport.iterationStats}")
      writer.println(s"${trainingReport.totalExecutors}")
      writer.println()
    } finally {
      writer.close()
    }
  }

  def loadPretrainedModel(modelPath: String): MultiLayerNetwork = {
    // Load the pretrained model from the specified file
    val file = new File(modelPath)
    val model = ModelSerializer.restoreMultiLayerNetwork(file)
    model
  }

  // Process text to obtain token IDs
  def processTokens(lines: Array[String], vec: Word2Vec): Array[Int] = {
    lines.flatMap { line =>
      val tokenizerFactory = new DefaultTokenizerFactory()
      tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor())

      // Tokenize the line
      val tokenizer = tokenizerFactory.create(line)
      Iterator.continually(if (tokenizer.hasMoreTokens) Some(tokenizer.nextToken()) else None)
        .takeWhile(_.isDefined)
        .flatMap { tokenOpt =>
          // Use the Word2Vec model to get the token ID
          val vocabWord = vec.getVocab.tokenFor(tokenOpt.get)
          if (vocabWord != null) Some(vocabWord.getIndex) else None
        }
    }
  }

  def splitDataset(data: Array[Int], trainFraction: Double): (Array[Int], Array[Int]) = {
    val trainSize = (data.length * trainFraction).toInt
    val trainingData = data.take(trainSize)
    val testingData = data.takeRight(data.length - trainSize)
    (trainingData, testingData)
  }

  // Embed a token
  def embed(tokens: Array[Int], embeddings: INDArray): INDArray = {
    // Stack the embeddings for each token in `tokens`
    Nd4j.vstack(
      tokens.map { tokenId =>
        embeddings.getRow(tokenId).dup() // Fetch and duplicate each embedding row for immutability
      }: _*
    )
  }

  // Need only for features
  def computePositionalEmbedding(inputEmbeddings: INDArray, positionalEmbedding: INDArray): INDArray = {
    inputEmbeddings.add(positionalEmbedding)
  }
}