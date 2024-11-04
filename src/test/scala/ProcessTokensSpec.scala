package nikos.cs441.hw2

import org.deeplearning4j.models.word2vec.wordstore.VocabCache
import org.deeplearning4j.models.word2vec.{VocabWord, Word2Vec}
import org.mockito.Mockito.{mock, when}
import org.scalatest.funsuite.AnyFunSuite

class ProcessTokensSpec extends AnyFunSuite {
  test("processTokens should return correct token indices for known words") {
    // Mock Word2Vec and VocabCache
    val vec = mock(classOf[Word2Vec])

    // Mock VocabCache
    val vocabCache = mock(classOf[VocabCache[VocabWord]])
    when(vec.getVocab).thenReturn(vocabCache)

    // Mock VocabWord for a specific token
    val vocabWordExample = mock(classOf[VocabWord])
    when(vocabWordExample.getIndex).thenReturn(42) // Define an arbitrary index for testing

    // Setup the mock to return `vocabWordExample` when "example" is looked up
    when(vocabCache.tokenFor("example")).thenReturn(vocabWordExample)

    // Sample lines for testing
    val lines = Array("This is an example sentence.")

    // Call processTokens with mock Word2Vec
    val result = Utils.processTokens(lines, vec)

    // Verify that the result contains the expected index
    assert(result.contains(42), "Expected index for 'example' not found in result.")
  }
}
