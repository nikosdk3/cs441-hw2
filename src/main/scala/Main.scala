package nikos.cs441.hw2

import Stats.{EpochStats, IterationStats}

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
import scala.collection.mutable.ListBuffer

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

    val (trainingData, testingData) = Utils.splitDataset(data, Config.trainTestSplit)
    val embeddings = Nd4j.rand(vec.vocabSize().toInt, Config.embeddingDim)

    val dataLoader = new DataLoader(vec.vocabSize(), embeddings, Config.batchSize, Config.seqLen, Config.stride, spark)

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

    var trainingReport = TrainingReport(Nil, Nil, 0) // Need this as var to accumulate the epoch and iteration data

    for (epoch <- 1 to Config.numEpochs) { // Training loop
      val startTimeEpoch = System.currentTimeMillis()

      var totalLoss = 0.0 // Need var to capture loss at the end of training
      var count = 0 // Need a count var to use for training loss averages

      val batchProcessingStart = System.currentTimeMillis()
      sparkModel.fit(trainingDataWindows)
      val batchProcessingEnd = System.currentTimeMillis()

      // Use a ListBuffer to collect iteration stats, which will be immutable at the end
      val iterationStatsBuilder = ListBuffer[IterationStats]()

      // Iterate over each batch and evaluate
      for ((batch, batchCounter) <- trainingBatches.zipWithIndex) {
        logger.info(s"Stats for iteration $batchCounter")
        val startTimeIter = System.currentTimeMillis()

        val features = batch.getFeatures
        val labels = batch.getLabels
        val output = model.output(features) // Get model output

        trainingEval.eval(labels, output) // Evaluate the predictions
        // Log current learning rate
        val learningRate = model.conf().getLayer.getUpdaterByParam("Adam").getLearningRate(batchCounter, epoch)

        // Calculate loss for this batch
        val loss = sparkModel.getNetwork.score(batch)
        totalLoss += loss
        count += 1

        val endTimeIter = System.currentTimeMillis()
        val iterationTime = endTimeIter - startTimeIter
        logger.info(s"Iteration time: $iterationTime ms")
        logger.info(s"Learning rate: $learningRate")

        val iterationStat = IterationStats(
          learningRate = learningRate,
          iterationTime = endTimeIter - startTimeIter,
          loss = loss
        )

        // Create a new list of iteration stats
        iterationStatsBuilder += iterationStat
      }

      // Calculate average loss
      val avgLoss = totalLoss / count

      val endTimeEpoch = System.currentTimeMillis()

      logger.info(s"Stats for epoch $epoch:")
      logger.info(s"Training Loss: $avgLoss")
      logger.info(s"Training Accuracy: ${trainingEval.accuracy()}")
      logger.info(s"Batch Processing Time: ${batchProcessingEnd - batchProcessingStart} ms")
      logger.info(s"Epoch time: ${endTimeEpoch - startTimeEpoch} ms")

      // Create an instance of EpochStats for this epoch
      val epochStats = EpochStats(
        epoch = epoch,
        trainingLoss = avgLoss,
        trainingAccuracy = trainingEval.accuracy(),
        batchProcessingTime = batchProcessingEnd - batchProcessingStart,
        epochTime = endTimeEpoch - startTimeEpoch
      )

      trainingReport = trainingReport.copy(
        epochStats = trainingReport.epochStats :+ epochStats,
        iterationStats = trainingReport.iterationStats ++ iterationStatsBuilder.toList // Convert ListBuffer to List
      )
    }

    logger.info("Training complete")

    logger.info("Spark Specific Metrics:")
    val executorMemoryStatus = spark.sparkContext.getExecutorMemoryStatus.toList
    logger.info(s"Total executors: ${executorMemoryStatus.size}")
    executorMemoryStatus.foreach { case (executorId, memory) =>
      logger.info(s"Executor ID: $executorId, Memory: $memory")
    }

    trainingReport = trainingReport.copy(
      totalExecutors = executorMemoryStatus.size
    )

    // Write training report to file
    Utils.writeToFile(trainingReport)

    // Save the model after training
    ModelSerializer.writeModel(sparkModel.getNetwork, new File("LLM_Spark_Model.zip"), true)

    val modelTest = new TextGeneration(embeddings, vec)

    val modelPath = "LLM_Spark_Model.zip" // Path to the pretrained model file
    val modelGen = Utils.loadPretrainedModel(modelPath)

    // Generate text using the pretrained model
    val query = "I had always thought"
    val generatedSentence = modelTest.generateSentence(query, modelGen, 5) // Generate a sentence with max 5 words

    logger.info("Generated Sentence: " + generatedSentence)

    spark.sparkContext.stop()
  }
}
