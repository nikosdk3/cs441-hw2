ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.20"

val logbackVersion = "1.5.11"
val typesafeVersion = "1.4.3"
val dl4jVersion = "1.0.0-M2.1"
val sparkVersion = "3.5.1"
val scalaTestVersion = "3.2.19"
val scalaTestPlusVersion = "3.2.19.0"

libraryDependencies ++= Seq(
  "ch.qos.logback" % "logback-classic" % logbackVersion,
  "com.typesafe" % "config" % typesafeVersion,
  "org.deeplearning4j" % "deeplearning4j-core" % dl4jVersion,
  "org.nd4j" % "nd4j-native-platform" % dl4jVersion,
  "org.deeplearning4j" % "deeplearning4j-nlp" % dl4jVersion,
  "org.deeplearning4j" %% "dl4j-spark" % dl4jVersion,
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.scalatest" %% "scalatest" % scalaTestVersion % "test",
  "org.scalatestplus" %% "mockito-5-12" % scalaTestPlusVersion % "test"
)

lazy val root = (project in file("."))
  .settings(
    name := "hw2",
    idePackagePrefix := Some("nikos.cs441.hw2")
  )