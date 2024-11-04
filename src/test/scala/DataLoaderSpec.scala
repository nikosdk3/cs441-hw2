package nikos.cs441.hw2

import org.apache.spark.sql.SparkSession
import org.deeplearning4j.models.word2vec.Word2Vec
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatestplus.mockito.MockitoSugar.mock

class DataLoaderSpec extends AnyFunSuite with BeforeAndAfterAll {
  private var spark: SparkSession = _
  private var dataLoader: DataLoader = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder
      .appName("DataLoaderTest")
      .master("local[*]")
      .getOrCreate()

    // Mock the Word2Vec model and create a DataLoader instance
    val vec = mock[Word2Vec]
    val embeddings = Nd4j.rand(100, 10) // Example dimensions
    dataLoader = new DataLoader(vec, embeddings, batchSize = 32, seqLen = 5, stride = 1, spark)
  }

  override def afterAll(): Unit = {
    spark.stop()
  }

  test("Sliding windows with embeddings should create the expected number of windows") {
    val inputData = Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    val windowsRDD = dataLoader.slidingWindowsWithEmbeddings(inputData)

    assert(windowsRDD.count() === 6) // Expected count based on seqLen and stride
  }
}