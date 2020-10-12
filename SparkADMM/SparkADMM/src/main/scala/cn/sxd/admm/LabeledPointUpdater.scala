package cn.sxd.admm

import breeze.linalg.Vector
import org.apache.spark.mllib.linalg

trait LabeledPointUpdater extends ADMMUpdater {
  def xUpdate(data:Array[(Double, linalg.Vector)],state: ADMMState,z:Vector[Double]): ADMMState
}
