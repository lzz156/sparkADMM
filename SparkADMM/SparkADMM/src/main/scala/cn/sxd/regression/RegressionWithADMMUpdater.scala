package cn.sxd.regression

import breeze.linalg.inv
import cn.sxd.util.MyBLAS.{axpy, dot}
import cn.sxd.admm.{ADMMState, ADMMUpdater, DenseMatrixUpdater}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

case class RegressionWithADMMUpdater(lambda: Double, myrho: Double,
                                     lbfgsMaxNumIterations: Int = 3,
                                     lbfgsHistory: Int = 5,
                                     lbfgsTolerance: Double = 1E-4)
  extends DenseMatrixUpdater {

  override var rho: Double = myrho

  override def xUpdate(data: (BDM[Double], BDV[Double]), state: ADMMState, z: BV[Double]): ADMMState = {
    val regularizerGradient = (z - state.y / rho)
    val n: Int = data._1.rows
    val p: BDM[Double] = (data._1.t * data._1).map(x => x / n)
    for (i <- 0 until z.size)
      p(i, i) += rho

    lazy val factor = inv(p)
    val q: BDV[Double] = (data._1.t * data._2).map(x => x / n)

    val xNew: BDV[Double] = (factor * (q + regularizerGradient * rho)).toDenseVector

    state.copy(x = xNew)
  }


  def convertDataToMatrix1(data: Array[LabeledPoint]): (BDM[Double], BDV[Double]) = {
    //样本数据矩阵的行数 n
    val n = data.length
    //样本数据矩阵的列数 p
    val p = data(0).features.size
    val x = data.map(x => x.features.toArray).flatMap(x => x)
    val b = data.map(x => x.label)
    val y: BDV[Double] = new BDV[Double](b)
    val mat = new BDM(p, n, x)
    (mat.t.copy, y)
  }


  def convertDataToMatrix2(data: Array[LabeledPoint]): (BDM[Double], BDV[Double]) = {
    //样本数据矩阵的行数 n
    val n = data.length
    //样本数据矩阵的列数 p
    //    val p = data(0).features.size
    val x = data.map(x => x.features.toArray)
    val b = data.map(x => x.label)
    val y: BDV[Double] = new BDV[Double](b)
    val mat = BDM(x: _*)
    (mat, y)
  }

  def prox(x: BDV[Double], rho: Double): BDV[Double] = {
    x.map(x_i => softThreshold(x_i, lambda / rho))
  }

  // 软阈值  进行更新 参数 lambda/rho
  def softThreshold(x: Double, thresh: Double): Double = {
    //
    Math.max(1.0 - thresh / Math.abs(x), 0.0) * x
  }

  override def L1Update(admmStates: RDD[(ADMMState)]) = {
    ADMMUpdater.soft_shresholdZUpdate(lambda, rho)(admmStates)
  }

  override def L2Update(states: RDD[ADMMState]) = {
    ADMMUpdater.linearZUpdate(lambda, rho)(states)
  }

  def fromBreeze(breezeVector: BV[Double]): Vector = {
    breezeVector match {
      case v: BDV[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new DenseVector(v.data)
        } else {
          new DenseVector(v.toArray) // Can't use underlying array directly, so make a new one
        }
      case v: BSV[Double] =>
        if (v.index.length == v.used) {
          new SparseVector(v.length, v.index, v.data)
        } else {
          new SparseVector(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }
    }
  }

  def toBreeze(values: Array[Double]): BV[Double] = new BDV[Double](values)

}
