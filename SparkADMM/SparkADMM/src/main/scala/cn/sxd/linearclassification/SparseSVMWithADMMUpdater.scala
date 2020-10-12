package cn.sxd.linearclassification

import breeze.linalg.{DenseVector, Vector, norm}
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS}
import cn.sxd.admm.{ADMMState, ADMMUpdater, LabeledPointUpdater}
import cn.sxd.util.MyBLAS.{axpy, dot}
import org.apache.spark.mllib.linalg
import org.apache.spark.rdd.RDD

case class SparseSVMWithADMMUpdater( lambda: Double,
                                     myrho: Double,
                                     cee: Double,
                                     lbfgsMaxNumIterations: Int = 5,
                                     lbfgsHistory: Int = 10,
                                     lbfgsTolerance: Double = 1E-4) extends ADMMUpdater with LabeledPointUpdater {
  override var rho: Double = myrho

  override def xUpdate(data: Array[(Double, linalg.Vector)], state: ADMMState, z: Vector[Double]): ADMMState = {
    // Our convex objective function that we seek to optimize
    val f = new DiffFunction[DenseVector[Double]] {
      override def calculate(x: DenseVector[Double]) = {
        (objective(data, state, z)(x), gradient(data, state, z)(x))
      }
    }

    val lbfgs = new LBFGS[DenseVector[Double]](
      maxIter = lbfgsMaxNumIterations,
      m = lbfgsHistory,
      tolerance = lbfgsTolerance)

    val states = lbfgs.iterations(new CachedDiffFunction(f), state.x)
    var step = states.next()
    while (states.hasNext) {
      step = states.next()
    }
    val xNew = step.x

    //        val xNew = lbfgs.minimize(f, state.x) // this is the "warm start" approach

    /*  val sgd: SimpleSGD[DenseVector[Double]] = new SimpleSGD[DenseVector[Double]](4,50)
      val xNew=sgd.minimize(f,state.x)
      state.copy(x = xNew)*/
    //   val x2New=1.5*xNew+(1-1.5)*z

    state.copy(x = xNew)

  }


  override def L1Update(states: RDD[(ADMMState)]) = {
    ADMMUpdater.soft_shresholdZUpdate(lambda, rho)(states)
  }

  override def L2Update(states: RDD[(ADMMState)]) = {
    ADMMUpdater.linearZUpdate(lambda, rho)(states)
  }


  //SVM的目标函数 参考论文有效的线性分类器里的SVM损失函数
  def objective(data: Array[(Double, linalg.Vector)], state: ADMMState, z: Vector[Double])(weight: DenseVector[Double]): Double = {
    val v = z - state.y / rho
    val regularizerObjective = norm(weight - v)
    //特征误差求和
    val lossObjective = data
      .map { lp => {
        val margin = math.max(1.0 - lp._1 * dot(lp._2.toSparse, weight.data), 0)
        val loss = math.pow(margin, 2)
        loss
      }
      }
      .sum
    //总的目标函数
    val totalObjective = cee * lossObjective + rho / 2 * regularizerObjective * regularizerObjective
    totalObjective
  }

  def gradient(data: Array[(Double, linalg.Vector)], state: ADMMState, z: Vector[Double])(weight: DenseVector[Double]): DenseVector[Double] = {
    // Eq (20) in
    // http:web.eecs.umich.edu/~honglak/aistats12-admmDistributedSVM.pdf
    val lossGradient: DenseVector[Double] = DenseVector.zeros[Double](weight.length)
    val v = z - state.y / rho
    val regularizerGradient = weight - v

    data.foreach {  lp => {
      val scaledLabel = 2 * lp._1 - 1
      val margin = math.max(1.0 - scaledLabel * dot(lp._2.toSparse, weight.data), 0)
      if (margin > 0) {


        val d: Double = dot(lp._2.toSparse, weight.data)
        val a: Double = d - scaledLabel
        // y += a * x 求得梯度向量
        axpy(a, lp._2.toSparse, lossGradient.data)
      }
    }

    }
    val totalGradient = rho * regularizerGradient + 2 * cee * lossGradient
    totalGradient
  }


}
