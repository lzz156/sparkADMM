package cn.sxd.test

import java.util.Date

import breeze.numerics.exp
import cn.sxd.util.LoadClassificationFile
import cn.sxd.util.MyBLAS.dot
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.optimization.GradientDescent
import org.apache.spark.mllib.regression.{LabeledPoint, LassoWithSGD}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object logistic {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setAppName("logidtic")
    conf.setMaster("local[4]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    val start_time = new Date().getTime
    val trainingData:RDD[LabeledPoint] = LoadClassificationFile.loadLibSVMFile(sc,"D://data//a9a.dat",-1,4)
//    MLUtils.loadLabeledPoints()
    println("trainingData的partitions: "+trainingData.getNumPartitions)
    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(trainingData)
//    val optimizer: GradientDescent = new LassoWithSGD().optimizer
//    LogisticRegressionWithSGD.train()
    val end_time = new Date().getTime
    println("run time: "+(end_time-start_time).toDouble/1000)
    val predictData:RDD[LabeledPoint] = LoadClassificationFile.loadLibSVMFile(sc,"D://data//a9at.dat",123,4)
    val predictionLabel =  predictData.map { lp =>
      val prediction = model.predict(lp.features)
      (lp.label,prediction)
    }
    val metrics = new BinaryClassificationMetrics(predictionLabel)
    println("precision: "+metrics.areaUnderPR())


    val testSize: Long = predictData.count()
    val ytrue: RDD[Double] = predictData.map(x => x.label)
    val ypre = predictData.map { lp => {
      val flag: Double = dot(lp.features.toSparse, model.weights)
      val label: Double = 1.0 / (exp(- flag) + 1.0)
      if ((label < 0.5) && (lp.label<=0)) {
        0.0
      } else if((label >= 0.5) && (lp.label>0))  {
        1.0
      }
    }
    }

    val accuracy = ypre.zip(ytrue).map(x => if (x._1 == x._2) 1.0 else 0.0).reduce(_ + _) / testSize.toDouble
    println("precision: " + accuracy * 100 + "%")
  }

}
