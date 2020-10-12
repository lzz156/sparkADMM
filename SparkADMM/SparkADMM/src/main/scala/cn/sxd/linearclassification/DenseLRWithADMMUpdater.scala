package cn.sxd.linearclassification

import breeze.linalg.{*, inv, norm, DenseMatrix => BDM, DenseVector => BDV, SparseVector => BSV, Vector => BV}
import breeze.numerics.exp
import cn.sxd.admm.{ADMMState, ADMMUpdater, DenseMatrixUpdater}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.util.Random
import scala.util.control.Breaks

case class DenseLRWithADMMUpdater(lambda: Double, myrho: Double,
                                lbfgsMaxNumIterations: Int = 3,
                                lbfgsHistory: Int = 5,
                                lbfgsTolerance: Double = 1E-4)
  extends DenseMatrixUpdater {



  override var rho: Double = myrho
  override def xUpdate(data: (BDM[Double], BDV[Double]), state: ADMMState,z:BV[Double]): ADMMState = {
    val v = ( z - state.y/rho)
    val m: Int = data._1.rows
    // Parameters related to convergence
    val max_iter: Int = 100
    val eps_abs: Double = 1e-6
    val eps_rel: Double = 1e-6
    val y: BDV[Double] = data._2.map(x => 2*x-1)
//    println(data._1)
//    println(data._2)
    // 待求解的模型参数
    val x_k: BDV[Double] = state.x
    var iter = 0
    val loop = new Breaks
    loop.breakable {
      for(i <- 0 until max_iter) {

        //每个样本对应的假设模型是bhat中sigmod函数的值
        val mu = pi(data._1, x_k)
        //？ 预测值*（1-预测值）
        val w = mu *:* (1.0-mu)
//        val g1: BDV[Double] = data._1.t*(y*mu-y)
        val g1: BDV[Double] = data._1.t*(y*mu-y)
        val grad = g1.map(x =>x/m) + rho * (x_k - v)

        //TODO why? Hessian = X'WX + lambda * I
        val wx = data._1(::, *) *:* w
        val H = (data._1.t*wx).map(x =>x/m)
        for(i <- 0 until z.size)
          H(i, i) += rho


//        val w = mu *:* (1.0 - mu)

        // Gradient = -X'(y-mu) + lambda*(beta-v)
//        val grad = x.t * (mu - y) + lambda * (bhat - v)

     /*   //TODO why? Hessian = X'WX + lambda * I
        val wx = x(::, *) :* w
        val H = x.t * wx
        for(i <- 0 until dim_p)
          H(i, i) += lambda*/


        val d_k: BDV[Double] = -inv(H) * (grad)
        x_k +=d_k
//        iter = i
//        println(iter)
        val r = norm(d_k)
        if(r < eps_abs * math.sqrt(z.size) || r < eps_rel * norm(x_k)) {
          loop.break
        }

        /*if(r < 1e-3) {
          loop.break
        }*/
      }
    }

    state.copy(x=x_k)
  }

  def pi(x: BDM[Double], b: BDV[Double]): BDV[Double] = {
    1.0 / (exp(- x * b) + 1.0)
  }

  def lambdas(n: Int, minr: Double = 0.001, maxr: Double = 1.0, nlambda: Int = 50): Array[Double] = {
    val logmin = math.log(minr * n)
    val logmax = math.log(maxr * n)
    val step = (logmax - logmin) / (nlambda - 1.0)
    val lambda = new Array[Double](nlambda)
    for (i <- 0 until nlambda) {
      lambda(i) = math.exp(logmax - i * step)
    }
    return lambda
  }

  def Singlelambdas(n: Int,i:Int, minr: Double = 0.001, maxr: Double = 1.0, nlambda: Int = 50)= {
    val logmin = math.log(minr * n)
    val logmax = math.log(maxr * n)
    val step = (logmax - logmin) / (i - 1.0)
    val lambda = math.exp(logmax - i * step)
    lambda
  }




  def convertDataToMatrix1(data: Array[LabeledPoint]):(BDM[Double], BDV[Double]) ={
    //样本数据矩阵的行数 n
    val n = data.length
    //样本数据矩阵的列数 p
    val p = data(0).features.size
    val x = data.map(x => x.features.toArray).flatMap(x => x)
    val b = data.map (x => x.label)
    val y: BDV[Double] = new BDV[Double](b)
    val mat = new BDM(p, n, x)
    (mat.t.copy,y)
  }


  def convertDataToMatrix2(data: Array[LabeledPoint]):(BDM[Double], BDV[Double]) ={
    //样本数据矩阵的行数 n
    val n = data.length
    //样本数据矩阵的列数 p
//    val p = data(0).features.size
    val x = data.map(x => x.features.toArray)
    val b = data.map (x => x.label)
    val y: BDV[Double] = new BDV[Double](b)
    val mat = BDM(x:_*)
    (mat,y)
  }

  def prox(x: BDV[Double], rho: Double): BDV[Double] = {
    x.map(x_i => softThreshold(x_i,lambda/rho))
  }

  // 软阈值  进行更新 参数 lambda/rho
  def softThreshold(x: Double, thresh: Double): Double = {
    //
    Math.max(1.0-thresh/Math.abs(x), 0.0)*x
  }

  override def L1Update(admmStates: RDD[( ADMMState)])= {
   ADMMUpdater.soft_shresholdZUpdate(lambda, rho)(admmStates)
  }

  override def L2Update(states: RDD[ADMMState])= {
    ADMMUpdater.linearZUpdate(lambda, rho)(states)
  }

  def fromBreeze(breezeVector: BV[Double]): Vector = {
    breezeVector match {
      case v: BDV[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new DenseVector(v.data)
        } else {
          new DenseVector(v.toArray)  // Can't use underlying array directly, so make a new one
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
