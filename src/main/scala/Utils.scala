package nikos.cs441.hw2

import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object Utils {
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