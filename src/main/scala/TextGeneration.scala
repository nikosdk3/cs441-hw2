package nikos.cs441.hw2

import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.annotation.tailrec

class TextGeneration(embeddings: INDArray, vec: Word2Vec) {
  // Method to generate the next word based on the query using the pretrained model
  private def generateNextWord(context: Array[String], model: MultiLayerNetwork): String = {
    // Tokenize context and convert to embedding (tokenization + embedding is done as part of homework 1)
    val tokens = Utils.processTokens(context, vec)
    val contextEmbedding = Utils.embed(tokens, embeddings).transpose() // Create embeddings for the input

    // Forward pass through the transformer layers (pretrained)
    val reshapedContext = contextEmbedding.reshape(1, Config.embeddingDim, tokens.length)
    val output = model.output(reshapedContext)
    // Find the word with the highest probability (greedy search) or sample
    val predictedWordIndex = Nd4j.argMax(output, 1).getInt(0) // Get the index of the predicted word

    vec.getVocab.wordAtIndex(predictedWordIndex)
  }

  // Method to generate a full sentence based on the seed text
  def generateSentence(seedText: String, model: MultiLayerNetwork, maxWords: Int): String = {
    // Initialize the generated text with the seed text
    val generatedText = new StringBuilder(seedText)
    // Split seed text into words for the context
    val context = seedText.split(" ")

    // Function to generate words based on the current context
    @tailrec
    def generateNextWords(currentContext: Array[String], remainingWords: Int): String = {
      if (remainingWords <= 0) return ""

      // Generate the next word
      val nextWord = generateNextWord(currentContext, model)
      // Append the generated word to the current text
      generatedText.append(" ").append(nextWord)

      // Update the context with the new word
      val updatedContext = (currentContext :+ nextWord).takeRight(Config.seqLen) // Keep the last seqLen words for context

      // Check for stopping conditions
      if (nextWord == "." || nextWord == "END") {
        ""
      } else {
        generateNextWords(updatedContext, remainingWords - 1)
      }
    }

    // Start generating words
    generateNextWords(context, maxWords)

    generatedText.toString()
  }
}
