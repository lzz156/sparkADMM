package cn.sxd.testmlib

import cn.sxd.util.LoadClassificationFile
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import cn.sxd.util.MyBLAS.{axpy, dot}

object MLRegression {

  def main(args: Array[String]): Unit = {

    val MaxIter = args(0).toInt
    val RegParam =  args(1).toDouble
    val setElasticNetParam =  args(2).toDouble

    //数据集
    val inputFilePath: String = args(3)
    val testinputFilePath: String = args(4)

    val spark = SparkSession.builder().appName("MLRegression").getOrCreate()
    Logger.getRootLogger.setLevel(Level.WARN)


    val training = spark.read.format("libsvm").load(inputFilePath)
    print(training)
    val lr = new LinearRegression()
    //最大迭代次数
    lr.setMaxIter(MaxIter)
    //正则化参数
    lr.setRegParam(RegParam)
    //学习率
    lr.setElasticNetParam(setElasticNetParam)
    val lrModel = lr.fit(training)
    val coefficients: linalg.Vector = lrModel.coefficients
    val rmodel: Vector = Vectors.dense(coefficients.toArray)
    val predictData: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(spark.sparkContext, testinputFilePath, 8)
    val total: Int = predictData.collect().size
    val ybar: Double = predictData.map(lp => lp.label).reduce(_ + _) / total
    val SST: Double = predictData.map { lp =>

      val Vay = (lp.label - ybar)* (lp.label - ybar)
      Vay

    }.reduce(_ + _)

    val SSE = predictData.map { lp =>
      val prediction: Double = dot(rmodel, lp.features)
      val d = (lp.label-prediction)*(lp.label-prediction)
      d
    }reduce(_ +_ )

    val R: Double = 1-SSE/SST
    println("均方误差（Mean Squared Error, MSE）"+SSE/total)
    println("决定系数:"+R)

  }
}
