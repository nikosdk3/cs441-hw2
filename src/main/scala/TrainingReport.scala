package nikos.cs441.hw2

import Stats.{EpochStats, IterationStats}

case class TrainingReport(
                           epochStats: List[EpochStats],
                           iterationStats: List[IterationStats],
                           totalExecutors: Int
                         )
