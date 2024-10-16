ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.3.4"

val logbackVersion = "1.5.8"
val typesafeVersion = "1.4.3"
val dl4jVersion = "1.0.0-M2.1"

libraryDependencies ++= Seq (
  "ch.qos.logback" % "logback-classic" % logbackVersion,
  "com.typesafe" % "config" % typesafeVersion,
  "org.deeplearning4j" % "deeplearning4j-core" % dl4jVersion,
  "org.nd4j" % "nd4j-native-platform" % dl4jVersion,
  "org.deeplearning4j" % "deeplearning4j-nlp" % dl4jVersion
)

lazy val root = (project in file("."))
  .settings(
    name := "hw2",
    idePackagePrefix := Some("nikos.cs441.hw2")
  )
