package cn.sxd.test

import java.util.Date

import cn.sxd.util.MyLoadLibSVMFile.loadLibSVMFile
import cn.sxd.linearclassification.SparseSVMWithADMM
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object  SVMADMMtrain {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setAppName("ADMMLR")
    conf.setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    //println("partion:"+sc.defaultMRRORinPartitions)
    val start_time = new Date().getTime
    val trainingData:RDD[LabeledPoint] = loadLibSVMFile(sc,"D://data//train0.1.dat",-1,6)
    // rho 1.0 cee 2.0
    val model = SparseSVMWithADMM.train(trainingData,100,1.0,2.0,1.0,true,true,true)
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
