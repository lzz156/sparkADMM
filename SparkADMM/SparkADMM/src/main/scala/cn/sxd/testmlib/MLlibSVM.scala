package cn.sxd.testmlib

import java.util.Date

import breeze.numerics.exp
import cn.sxd.util.LoadClassificationFile
import cn.sxd.util.MyBLAS.dot
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object MLlibSVM {
  def main(args: Array[String]): Unit = {


    //分区数
    val minPartition = args(0).toInt
    // 惩罚参数
    val numFeatures = args(1).toInt
    val numIter =  args(2).toInt
    val stepSize: Double = args(3).toDouble
    val regPara: Double = args(4).toDouble.toDouble
    val miniFraction: Double = args(5).toDouble
    //数据集
    val inputFilePath: String = args(6)
    val testinputFilePath: String = args(7)

    val conf = new SparkConf().setAppName("MLlibSVM")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val trainingData:RDD[LabeledPoint] = LoadClassificationFile.loadLibSVMFile(sc,inputFilePath,numFeatures,minPartition)
    println("trainingData的partitions: "+trainingData.getNumPartitions)
    val start_time = new Date().getTime
    val SVM = SVMWithSGD.train(trainingData,numIter,stepSize,regPara,miniFraction)
    val end_time = new Date().getTime
    println("run time: "+(end_time-start_time).toDouble/1000)
    val predictData:RDD[LabeledPoint] = LoadClassificationFile.loadLibSVMFile(sc,testinputFilePath,numFeatures,minPartition)
    val predictionLabel =  predictData.map { lp =>
      val prediction = SVM.predict(lp.features)
      (lp.label,prediction)
    }
    val metrics = new BinaryClassificationMetrics(predictionLabel)
    println("precision: "+metrics.areaUnderPR())

  }
}
