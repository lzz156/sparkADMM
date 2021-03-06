package cn.sxd.testadmm.local

import java.util.Date

import cn.sxd.linearclassification.SparseSVMWithADMM
import cn.sxd.util.LoadClassificationFile.loadLibSVMFile
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object SVMADMMWithLBFGS {

  def main(args: Array[String]): Unit = {



    val minPartition = 4
    val lambda =  1.0
    val rho =  2.0
    val cee =  1.0
    val updateRho: Boolean = false
    val used_primalresid: Boolean = false
    val used_L1orL2: Boolean = false
    val inputFilePath= "D://data//train0.1.dat"
    val testinputFilePath = "D://data//test0.1.dat"



    val conf = new SparkConf().setAppName("LRADMMWithLBFGS").setMaster("local[*]")
      //使用 Kryo 序列化库，如果要使用 Java 序列化库，需要把该行屏蔽掉//使用 Kryo 序列化库，如果要使用 Java 序列化库，需要把该行屏蔽掉
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    //在 Kryo 序列化库中注册自定义的类集合，如果要使用 Java 序列化库，需要把该行屏蔽掉
    conf.set("spark.kryo.registrator", "cn.sxd.admm.ADMMKryoRegistrator")
    // 设置jar包的路径,如果有其他的依赖包,可以在这里添加,逗号隔开
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val start_time = new Date().getTime
    val trainingData = loadLibSVMFile(sc,inputFilePath,-1,minPartition)
    val model =SparseSVMWithADMM
      .train(trainingData,100,lambda,rho,cee,updateRho,used_primalresid,used_L1orL2)
    val end_time = new Date().getTime
    println("run time: "+(end_time-start_time).toDouble/1000)
    val predictData:RDD[LabeledPoint] = loadLibSVMFile(sc,testinputFilePath,-1,minPartition)
    val predictionLabel =  predictData.map { lp =>
      val prediction = model.predict(lp.features)
      (lp.label,prediction)
    }
    val metrics = new BinaryClassificationMetrics(predictionLabel)
    println("precision: "+metrics.areaUnderPR())
  }



}
