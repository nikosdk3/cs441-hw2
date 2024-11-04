package nikos.cs441.hw2

import org.apache.spark.sql.SparkSession
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory

import java.io.File

object Main {
  private val logger = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    logger.info("Creating Spark Session...")

    // Set up Spark session
    val spark = SparkSession.builder
      .appName("SparkDL4J-LLM")
      .master("local[*]")
      .getOrCreate()

    logger.info("Spark initialized successfully.")

    val filePath = "the-verdict.txt"

    // Strip white space before and after for each line
    val iter = new BasicLineIterator(filePath)
    // Split on white spaces in the line to get words
    val t = new DefaultTokenizerFactory()
    t.setTokenPreProcessor(new CommonPreprocessor())

    // Only used for building a vocabulary
    val vec = new Word2Vec.Builder()
      .iterate(iter)
      .tokenizerFactory(t)
      .build()

    vec.fit()

    // Load lines into an RDD and map each line to token IDs
    val lines = spark.sparkContext.textFile(filePath).collect()
    val data = Utils.processTokens(lines, vec)

    val (trainingData, testingData) = splitDataset(data.take(500), Config.trainTestSplit)
    val embeddings = Nd4j.rand(vec.vocabSize().toInt, Config.embeddingDim)

    val dataLoader = new DataLoader(vec, embeddings, Config.batchSize, Config.seqLen, Config.stride, spark)

    val trainingDataWindows = dataLoader.slidingWindowsWithEmbeddings(trainingData)
    val testingDataWindows = dataLoader.slidingWindowsWithEmbeddings(testingData)

    logger.info(s"Training data count: ${trainingDataWindows.count()}")
    logger.info(s"Testing data count: ${testingDataWindows.count()}")

    val model = Model.createModel(Config.embeddingDim, Config.hiddenLayerDim, vec.vocabSize().toInt)

    // Set up the TrainingMaster configuration
    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(1)
      .batchSizePerWorker(Config.batchSize) // Batch size on each Spark worker
      .averagingFrequency(5) // Frequency of parameter averaging
      .workerPrefetchNumBatches(2)
      .build()

    val sparkModel = new SparkDl4jMultiLayer(spark.sparkContext, model, trainingMaster)

    sparkModel.setListeners(new ScoreIterationListener(10))
    val trainingEval = new Evaluation(vec.vocabSize().toInt)

    val trainingBatches = trainingDataWindows.collect()

    for (epoch <- 1 to Config.numEpochs) { // Training loop
      val startTimeEpoch = System.currentTimeMillis()

      var totalLoss = 0.0
      var count = 0

      val batchProcessingStart = System.currentTimeMillis()
      sparkModel.fit(trainingDataWindows)
      val batchProcessingEnd = System.currentTimeMillis()

      // Iterate over each batch and evaluate
      for ((batch, batchCounter) <- trainingBatches.zipWithIndex) {
        logger.info(s"Stats for iteration $batchCounter")
        val startTimeIter = System.currentTimeMillis()

        val features = batch.getFeatures
        val labels = batch.getLabels
        val output = model.output(features) // Get model output

        trainingEval.eval(labels, output) // Evaluate the predictions
        // Log current learning rate if applicable
        val learningRate = model.conf().getLayer.getUpdaterByParam("Adam").getLearningRate(batchCounter, epoch)

        // Calculate loss for this batch
        val loss = model.score(batch)
        totalLoss += loss
        count += 1

        val endTimeIter = System.currentTimeMillis()
        logger.info(s"Iteration time: ${endTimeIter - startTimeIter} ms")
        logger.info(s"Learning rate: $learningRate")
      }

      // Calculate average loss
      val avgLoss = totalLoss / count

      // L1 and L2 Norms
      val gradientNorm1 = model.getGradient.gradient().norm1()
      val gradientNorm2 = model.getGradient.gradient().norm2()

      val endTimeEpoch = System.currentTimeMillis()
      logger.info(s"Stats for epoch $epoch:")
      logger.info(s"Training Loss: $avgLoss")
      logger.info(s"Training Accuracy: ${trainingEval.accuracy()}")
      logger.info(s"Gradient L1 Norm: $gradientNorm1")
      logger.info(s"Gradient L1 Norm: $gradientNorm2")
      logger.info("Learning rate (Adam): 0.001")
      logger.info(s"Batch Processing Time: ${batchProcessingEnd - batchProcessingStart} ms")
      logger.info(s"Epoch time: ${endTimeEpoch - startTimeEpoch} ms")
    }

    logger.info("Training complete")
    logger.info("Spark Specific Metrics:")
    val executorMemoryStatus = spark.sparkContext.getExecutorMemoryStatus
    logger.info(s"Total executors: ${executorMemoryStatus.size}")
    executorMemoryStatus.foreach { case (executorId, memory) =>
      logger.info(s"Executor ID: $executorId, Memory: $memory")
    }

    // Save the model after training
    ModelSerializer.writeModel(sparkModel.getNetwork, new File("LLM_Spark_Model.zip"), true)

    val modelTest = new ModelTest(embeddings, vec)

    val modelPath = "LLM_Spark_Model.zip" // Path to the pretrained model file
    val modelGen = modelTest.loadPretrainedModel(modelPath)

    // Generate text using the pretrained model
    val query = "I had always thought"
    val generatedSentence = modelTest.generateSentence(query, modelGen, 5) // Generate a sentence with max 5 words

    logger.info("Generated Sentence: " + generatedSentence)

    spark.sparkContext.stop()
  }

  private def splitDataset(data: Array[Int], trainFraction: Double): (Array[Int], Array[Int]) = {
    val trainSize = (data.length * trainFraction).toInt
    val trainingData = data.take(trainSize)
    val testingData = data.takeRight(data.length - trainSize)
    (trainingData, testingData)
  }
}
