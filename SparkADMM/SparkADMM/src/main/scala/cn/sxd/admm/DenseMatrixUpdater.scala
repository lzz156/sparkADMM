package cn.sxd.admm

import breeze.linalg.{DenseMatrix, DenseVector, Vector}

trait DenseMatrixUpdater extends ADMMUpdater {
  def xUpdate(data:(DenseMatrix[Double], DenseVector[Double]),state: ADMMState,z:Vector[Double]): ADMMState
}
