package cn.sxd.test

import java.util.Date
import cn.sxd.util.LoadClassificationFile.loadLibSVMFile
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object TestSVMWithSGD {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setAppName("logidtic")
    conf.setMaster("local[4]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    val start_time = new Date().getTime
    val trainingData:RDD[LabeledPoint] = loadLibSVMFile(sc,"D://data//train0.1.dat",-1,4)
//    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(trainingData)
//val lsvc = new LinearSVC().setRegParam(0.3).setMaxIter(100)
//    val model = lsvc.fit(data)
val model = SVMWithSGD.train(trainingData,100)
//    val c: LinearSVC = LinearSVC.load("D://data//data_0.dat")
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

}
