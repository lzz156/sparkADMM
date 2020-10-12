package cn.sxd.test

import java.util.Date

import breeze.numerics.exp
import cn.sxd.util.MyLoadLibSVMFile.loadLibSVMFile
import cn.sxd.regression.RegressionWithADMM
import cn.sxd.util.MyBLAS.dot
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object  LASSOADMMtrain {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setAppName("ADMMLR")
    conf.setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    //println("partion:"+sc.defaultMRRORinPartitions)
    val start_time = new Date().getTime
    val trainingData:RDD[LabeledPoint] = loadLibSVMFile(sc,"D://data//eunite2001.dat",-1,4)
    // rho 1.0 cee 2.0
    val model = RegressionWithADMM.train(trainingData,100,2.0,2.0,false,true,true)
    val end_time = new Date().getTime
    println("run time: "+(end_time-start_time).toDouble/1000)
    val predictData:RDD[LabeledPoint] = loadLibSVMFile(sc,"D://data//eunite2001t.dat",-1,4)
    val predictionLabel =  predictData.map { lp =>
      val prediction = model.predict(lp.features)
      (lp.label,prediction)
    }
    predictionLabel.collect().foreach(println(_))
/*    val metrics = new BinaryClassificationMetrics(predictionLabel)
    println("precision: "+metrics.areaUnderPR())*/


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
