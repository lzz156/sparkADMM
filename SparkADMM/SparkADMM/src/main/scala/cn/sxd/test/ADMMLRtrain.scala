package cn.sxd.test

import java.util.Date

import breeze.numerics.exp
import cn.sxd.linearclassification.{DenseLRADMM, SparseLRADMM}
import cn.sxd.util.LoadClassificationFile.loadLibSVMFile
import cn.sxd.util.MyBLAS.dot
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext, ml}

object  ADMMLRtrain {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setAppName("ADMMLR")
    conf.setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    //println("partion:"+sc.defaultMinPartitions)
    val start_time = new Date().getTime
    val trainingData = loadLibSVMFile(sc,"D://data//train0.1.dat",-1,4)
    val model =SparseLRADMM
      .train(trainingData,100,1.0,2.0,false,false,false)
    val end_time = new Date().getTime
    println("run time: "+(end_time-start_time).toDouble/1000)
    val predictData:RDD[LabeledPoint] = loadLibSVMFile(sc,"D://data//test0.1.dat",-1,4)
    val predictionLabel =  predictData.map { lp =>
      val prediction = model.predict(lp.features)
      (lp.label,prediction)
    }
    val metrics = new BinaryClassificationMetrics(predictionLabel)
  println("precision: "+metrics.areaUnderPR())
  }



  def loadDatasets(
                    spark: SparkSession,
                    input: String,
                    dataFormat: String,
                    testInput: String,
                    algo: String,
                    fracTest: Double,
                    minPartition: Int): (DataFrame, DataFrame, Int) = {


    // Load training data
    val origExamples: DataFrame = loadData(spark, input, dataFormat, minPartition)

    // Load or create test set
    val dataframes: Array[DataFrame] = if (testInput != "") {
      // Load testInput.
      val numFeatures = origExamples.first().getAs[ml.linalg.Vector](1).size
      val origTestExamples: DataFrame =
        loadData(spark, testInput, dataFormat, minPartition, Some(numFeatures))
      Array(origExamples, origTestExamples)
    } else {
      // Split input into training, test.
      origExamples.randomSplit(Array(1.0 - fracTest, fracTest), seed = 12345)
    }

    val training = dataframes(0).cache()
    val test = dataframes(1).cache()

    val numTraining = training.count()
    val numTest = test.count()
    val numFeatures = training.select("features").first().getAs[ml.linalg.Vector](0).size
    println("Loaded data:")
    println(s"  numTraining = $numTraining, numTest = $numTest")
    println(s"  numFeatures = $numFeatures")

    (training, test, numFeatures)
  }

  def loadData(
                spark: SparkSession,
                path: String,
                format: String,
                minPartition: Int,
                expectedNumFeatures: Option[Int] = None): DataFrame = {
    import spark.implicits._

    format match {
      case "dense" => MLUtils.loadLabeledPoints(spark.sparkContext, path).toDF()
      case "libsvm" => expectedNumFeatures match {
        case Some(numFeatures) => loadLibSVMFile(spark.sparkContext,"data/sample_libsvm_data.txt",-1,minPartition).toDF()
        case None => spark.read.format("libsvm").load(path)
      }
      case _ => throw new IllegalArgumentException(s"Bad data format: $format")
    }
  }

}
