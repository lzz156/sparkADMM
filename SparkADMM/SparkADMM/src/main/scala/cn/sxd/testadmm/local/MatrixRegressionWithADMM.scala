package cn.sxd.testadmm.local

import java.util.Date

import breeze.numerics.exp
import cn.sxd.regression.RegressionWithADMM
import cn.sxd.util.MyBLAS.dot
import cn.sxd.util.MyLoadLibSVMFile.loadLibSVMFile
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object  MatrixRegressionWithADMM {
  def main(args: Array[String]): Unit = {

    val minPartition = 5
    val numFeatures = 90
    val lambda =  1.0
    val rho =  2.0
    val updateRho: Boolean = false
    val used_primalresid: Boolean = true
    val used_L1orL2: Boolean = false
    val inputFilePath= "D://data//YearPredictionMSD.dat"
    val testinputFilePath = "D://data//YearPredictionMSDt.dat"

    val conf = new SparkConf()
    conf.setAppName("ADMMLR")
    conf.setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val start_time = new Date().getTime
    val trainingData:RDD[LabeledPoint] = loadLibSVMFile(sc,inputFilePath,numFeatures,minPartition)

    val model = RegressionWithADMM.train(trainingData,20,lambda,rho,updateRho,used_primalresid,used_L1orL2)
    val end_time = new Date().getTime
    println("run time: "+(end_time-start_time).toDouble/1000)
    val predictData:RDD[LabeledPoint] = loadLibSVMFile(sc,testinputFilePath,numFeatures,minPartition)
    val total: Int = predictData.collect().size
    val ybar: Double = predictData.map(lp => lp.label).reduce(_ + _) / total
    val SST: Double = predictData.map { lp =>

      val Vay = (lp.label - ybar)* (lp.label - ybar)
      Vay

    }.reduce(_ + _)


    val SSE =  predictData.map { lp =>
      val prediction = model.predict(lp.features)

      val d = (lp.label-prediction)*(lp.label-prediction)
      d


    }reduce(_+_)


    val R: Double = 1-SSE/SST
    println("测试样本数量："+total)
    println("均方误差（Mean Squared Error, MSE）"+SSE/total)
    println("决定系数:"+R)




  }
}
