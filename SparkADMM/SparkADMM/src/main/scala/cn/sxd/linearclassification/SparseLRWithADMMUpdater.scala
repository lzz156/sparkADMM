package cn.sxd.linearclassification

import breeze.linalg.norm
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, SparseVector => BSV, Vector => BV}
import cn.sxd.util.MyBLAS.{axpy, dot}
import cn.sxd.admm.{ADMMState, ADMMUpdater, LabeledPointUpdater}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

case class SparseLRWithADMMUpdater(lambda: Double,
                                   myrho: Double,
                                   lbfgsMaxNumIterations: Int = 10,
                                   lbfgsHistory: Int = 5,
                                   lbfgsTolerance: Double = 1E-4)
  extends ADMMUpdater with LabeledPointUpdater {

  override var rho: Double = myrho

  override def xUpdate(data: Array[(Double, linalg.Vector)], state: ADMMState, z: BV[Double]): ADMMState = {
    val f = new DiffFunction[BDV[Double]] {
      override def calculate(x: BDV[Double]): (Double, BDV[Double]) = {
        (objective(data, state, z)(x), gradient(data, state, z)(x))
      }
    }

    val lbfgs = new LBFGS[BDV[Double]](maxIter = lbfgsMaxNumIterations,
      m = lbfgsHistory,
      tolerance = lbfgsTolerance)

    //warm start
    val states = lbfgs.iterations(new CachedDiffFunction(f), state.x)
    var step = states.next()
    while (states.hasNext) {
      step = states.next()
    }
    val xNew: BDV[Double] = step.x


    //   val xNew = lbfgs.minimize(f, state.x)
    //    val x2New=1.7*xNew+(1-1.7)*z

    state.copy(x = xNew)
  }

  //, dim:Int

  override def L1Update(states: RDD[ADMMState]) = {
    ADMMUpdater.soft_shresholdZUpdate(lambda, rho)(states)
  }

  override def L2Update(states: RDD[ADMMState]) = {
    ADMMUpdater.linearZUpdate(lambda, rho)(states)
  }


  //损失函数
  def objective(data: Array[(Double, linalg.Vector)], state: ADMMState, z: BV[Double])(weight: BDV[Double]): Double = {
    val lossObjective = data
      //逻辑回归的损失函数求和公式
      .map(lp => {
        // 转换为 y={-1，1}标签
        val scaledLabel = 2 * lp._1 - 1
        val margin = scaledLabel * dot(lp._2.toSparse, weight.data)
        logPhi(margin)
      }).sum
    val regularizerObjective = norm(weight - z + state.y / rho)
    val totalObjective = rho / 2 * regularizerObjective * regularizerObjective + lossObjective
    //println("object value"+totalObjective)
    totalObjective
  }

  //对于每个RDD中的features求梯度
  def gradient(data: Array[(Double, linalg.Vector)], state: ADMMState, z: BV[Double])(weight: BDV[Double]): BDV[Double] = {
    val lossGradient = BDV.zeros[Double](weight.length)
    data.foreach(lp => {
      //fromBreeze(weights),fromBreeze(lossGradient)
      val scaledLabel = 2 * lp._1 - 1
      val margin = scaledLabel * dot(lp._2.toSparse, weight.data)
      val a = scaledLabel * (phi(margin) - 1)
      axpy(a, lp._2.toSparse, lossGradient.data)
    }

    )
    val regularizerGradient = (weight - z + state.y / rho)
    val totalGradient = lossGradient + rho * regularizerGradient
    totalGradient
  }


  //防止溢出
  private def logPhi(margin: Double): Double = {
    if (margin > 0)
      math.log(1 + math.exp(-margin))

//    margin + math.log1p(math.exp(-margin))
    else
      (-margin + math.log(1 + math.exp(margin)))
//    math.log1p(math.exp(margin))

  }


  //逻辑回归的原始损失函数 log(1+e^-YiWiTXi)
  private def phi(margin: Double): Double = {
    1 / (1 + math.exp(-margin))
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


}
