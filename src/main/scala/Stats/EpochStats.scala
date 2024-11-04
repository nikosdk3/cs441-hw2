package nikos.cs441.hw2
package Stats

case class EpochStats(
                       epoch: Int,
                       trainingLoss: Double,
                       trainingAccuracy: Double,
                       batchProcessingTime: Long,
                       epochTime: Long
                     )
