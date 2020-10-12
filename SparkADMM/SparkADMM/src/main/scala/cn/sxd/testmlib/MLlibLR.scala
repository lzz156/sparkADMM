package cn.sxd.testmlib

import java.util.Date

import breeze.numerics.exp
import cn.sxd.util.LoadClassificationFile
import cn.sxd.util.MyBLAS.dot
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.optimization.{LBFGS, SquaredL2Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object MLlibLR {


  def main(args: Array[String]): Unit = {


    val minPartition = args(0).toInt
    //数据集
    val numFeatures = args(1).toInt
    val inputFilePath: String = args(2)
    val testinputFilePath: String = args(3)

    val conf = new SparkConf()
    conf.setAppName("MLlibLR")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val trainingData:RDD[LabeledPoint] = LoadClassificationFile.loadLibSVMFile(sc,inputFilePath,numFeatures,minPartition)
    println("trainingData的partitions: "+trainingData.getNumPartitions)
    val start_time = new Date().getTime
    val LR = new LogisticRegressionWithLBFGS().setNumClasses(2)
//    LR.optimizer.setRegParam(1.0).setUpdater(new SquaredL2Updater)
    val model: LogisticRegressionModel = LR.run(trainingData)
    val end_time = new Date().getTime
    println("run time: "+(end_time-start_time).toDouble/1000)
    val predictData:RDD[LabeledPoint] = LoadClassificationFile.loadLibSVMFile(sc,testinputFilePath,numFeatures,minPartition)
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
