package cn.sxd.testmlib.local

import java.util.Date

import breeze.numerics.exp
import cn.sxd.util.LoadClassificationFile
import cn.sxd.util.MyBLAS.dot
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.optimization.{L1Updater, SquaredL2Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object MLlibSVM {
  def main(args: Array[String]): Unit = {


    //分区数
    val minPartition = 4
    // 惩罚参数
    val RegParam =  1

    //数据集
    val inputFilePath: String = "D://data//train0.1.dat"
    val testinputFilePath: String = "D://data//test0.1.dat"

    val conf = new SparkConf().setAppName("MLlibLR").setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val trainingData:RDD[LabeledPoint] = LoadClassificationFile.loadLibSVMFile(sc,inputFilePath,47236,minPartition)
    println("trainingData的partitions: "+trainingData.getNumPartitions)
    val start_time = new Date().getTime
    val SVM = SVMWithSGD.train(trainingData,100,1.0,0.1,0.8)
    val end_time = new Date().getTime
    println("run time: "+(end_time-start_time).toDouble/1000)
    val predictData:RDD[LabeledPoint] = LoadClassificationFile.loadLibSVMFile(sc,testinputFilePath,47236,minPartition)
    val predictionLabel =  predictData.map { lp =>
      val prediction = SVM.predict(lp.features)
      (lp.label,prediction)
    }
    val metrics = new BinaryClassificationMetrics(predictionLabel)
    println("precision: "+metrics.areaUnderPR())
  }
}
