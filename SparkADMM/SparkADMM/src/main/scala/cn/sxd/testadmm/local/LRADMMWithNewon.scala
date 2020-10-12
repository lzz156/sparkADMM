package cn.sxd.testadmm.local

import java.util.Date

import breeze.numerics.exp
import cn.sxd.linearclassification.DenseLRADMM
import cn.sxd.util.LoadClassificationFile.loadLibSVMFile
import cn.sxd.util.MyBLAS.dot
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object LRADMMWithNewon {


  def main(args: Array[String]): Unit = {
    val minPartition = 4
    val lambda =  1.0
    val rho =  2.0
    val updateRho: Boolean = false
    val used_primalresid: Boolean = true
    val used_L1orL2: Boolean = false
    val inputFilePath= "D://data//a9a.dat"
    val testinputFilePath = "D://data//a9at.dat"


    val conf = new SparkConf().setAppName("LRADMMWithLBFGS").setMaster("local[*]")
      //使用 Kryo 序列化库，如果要使用 Java 序列化库，需要把该行屏蔽掉//使用 Kryo 序列化库，如果要使用 Java 序列化库，需要把该行屏蔽掉
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    //在 Kryo 序列化库中注册自定义的类集合，如果要使用 Java 序列化库，需要把该行屏蔽掉
    conf.set("spark.kryo.registrator", "cn.sxd.admm.ADMMKryoRegistrator")
    // 设置jar包的路径,如果有其他的依赖包,可以在这里添加,逗号隔开
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val start_time = new Date().getTime
    val trainingData = loadLibSVMFile(sc,inputFilePath,123,minPartition)
    val model =DenseLRADMM
      .train(trainingData,100,lambda,rho,updateRho,used_primalresid,used_L1orL2)
    val end_time = new Date().getTime
    println("run time: "+(end_time-start_time).toDouble/1000)
    val predictData:RDD[LabeledPoint] = loadLibSVMFile(sc,testinputFilePath,123,minPartition)
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
