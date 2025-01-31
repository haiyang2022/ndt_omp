/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#ifndef PCL_REGISTRATION_NDT_OMP_H_
#define PCL_REGISTRATION_NDT_OMP_H_

#include <pcl/registration/registration.h>
#include <pcl/search/impl/search.hpp>
#include "voxel_grid_covariance_omp.h"

#include <unsupported/Eigen/NonLinearOptimization>

namespace pclomp
{	// 枚举定义邻域搜索方式:
	// KDTREE：使用 KD-Tree 进行邻域搜索
	// DIRECT26 / DIRECT7 / DIRECT1：基于体素直接搜索
	// （一次取 26 / 7 / 1 个周边邻域体素），通常用于加速邻域查找
	enum NeighborSearchMethod {
		KDTREE,
		DIRECT26,
		DIRECT7,
		DIRECT1
	};

	/** \brief A 3D Normal Distribution Transform registration implementation for point cloud data.
	  * \note For more information please see
	  * <b>Magnusson, M. (2009). The Three-Dimensional Normal-Distributions Transform —
	  * an Efficient Representation for Registration, Surface Analysis, and Loop Detection.
	  * PhD thesis, Orebro University. Orebro Studies in Technology 36.</b>,
	  * <b>More, J., and Thuente, D. (1994). Line Search Algorithm with Guaranteed Sufficient Decrease
	  * In ACM Transactions on Mathematical Software.</b> and
	  * Sun, W. and Yuan, Y, (2006) Optimization Theory and Methods: Nonlinear Programming. 89-100
	  * \note Math refactored by Todor Stoyanov.
	  * \author Brian Okorn (Space and Naval Warfare Systems Center Pacific)
	  */
	// 定义一个 NDT 模板类，继承自 PCL 的 pcl::Registration 基类
	template<typename PointSource, typename PointTarget>
	class NormalDistributionsTransform : public pcl::Registration<PointSource, PointTarget>
	{
		// 定义了一系列 智能指针与点云类型 的别名
	protected:

		typedef typename pcl::Registration<PointSource, PointTarget>::PointCloudSource PointCloudSource;
		typedef typename PointCloudSource::Ptr PointCloudSourcePtr;
		typedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;

		typedef typename pcl::Registration<PointSource, PointTarget>::PointCloudTarget PointCloudTarget;
		typedef typename PointCloudTarget::Ptr PointCloudTargetPtr;
		typedef typename PointCloudTarget::ConstPtr PointCloudTargetConstPtr;

		typedef pcl::PointIndices::Ptr PointIndicesPtr;
		typedef pcl::PointIndices::ConstPtr PointIndicesConstPtr;

		/** \brief Typename of searchable voxel grid containing mean and covariance. */
		// VoxelGridCovariance：存储目标点云中每个体素 (voxel) 的均值与协方差等统计数据，用于 NDT 优化
		typedef pclomp::VoxelGridCovariance<PointTarget> TargetGrid;
		/** \brief Typename of pointer to searchable voxel grid. */
		// 通过 TargetGrid 与其指针类型加速对体素数据的访问
		typedef TargetGrid* TargetGridPtr;
		/** \brief Typename of const pointer to searchable voxel grid. */
		typedef const TargetGrid* TargetGridConstPtr;
		/** \brief Typename of const pointer to searchable voxel grid leaf. */
		typedef typename TargetGrid::LeafConstPtr TargetGridLeafConstPtr;


	public:
	// 智能指针：兼容 PCL 1.10 及之前不同版本的 shared_ptr 实现
#if PCL_VERSION >= PCL_VERSION_CALC(1, 10, 0)
		typedef pcl::shared_ptr< NormalDistributionsTransform<PointSource, PointTarget> > Ptr;
		typedef pcl::shared_ptr< const NormalDistributionsTransform<PointSource, PointTarget> > ConstPtr;
#else
		typedef boost::shared_ptr< NormalDistributionsTransform<PointSource, PointTarget> > Ptr;
		typedef boost::shared_ptr< const NormalDistributionsTransform<PointSource, PointTarget> > ConstPtr;
#endif


		/** \brief Constructor.
		  * Sets \ref outlier_ratio_ to 0.35, \ref step_size_ to 0.05 and \ref resolution_ to 1.0
		  */
		NormalDistributionsTransform();

		/** \brief Empty destructor */
		virtual ~NormalDistributionsTransform() {}
	// 设置 NDT 计算中使用的线程数量
    void setNumThreads(int n) {
      num_threads_ = n;
    }

		/** \brief Provide a pointer to the input target (e.g., the point cloud that we want to align the input source to).
		  * \param[in] cloud the input point cloud target
		  */
		// 重写 setInputTarget,覆盖 Registration 基类的 setInputTarget 函数
		// 在设置目标点云后会调用 init() 函数，对目标点云进行体素栅格化并计算均值、协方差等
		inline void
			setInputTarget(const PointCloudTargetConstPtr &cloud)
		{
			pcl::Registration<PointSource, PointTarget>::setInputTarget(cloud);
			init();
		}

		/** \brief Set/change the voxel grid resolution.
		  * \param[in] resolution side length of voxels
		  */
		// 设置参数:
		// resolution_：每个体素的边长
		// step_size_：NDT 中 More-Thuente 线搜索的 最大步长
		// outlier_ratio_：用于处理离群点的比率（NDT 误差模型中用到）
		inline void
			setResolution(float resolution)
		{
			// Prevents unnecessary voxel initiations
			if (resolution_ != resolution)
			{
				resolution_ = resolution;
				if (input_)
					init();
			}
		}

		/** \brief Get voxel grid resolution.
		  * \return side length of voxels
		  */
		inline float
			getResolution() const
		{
			return (resolution_);
		}

		/** \brief Get the newton line search maximum step length.
		  * \return maximum step length
		  */
		inline double
			getStepSize() const
		{
			return (step_size_);
		}

		/** \brief Set/change the newton line search maximum step length.
		  * \param[in] step_size maximum step length
		  */
		inline void
			setStepSize(double step_size)
		{
			step_size_ = step_size;
		}

		/** \brief Get the point cloud outlier ratio.
		  * \return outlier ratio
		  */
		inline double
			getOutlierRatio() const
		{
			return (outlier_ratio_);
		}

		/** \brief Set/change the point cloud outlier ratio.
		  * \param[in] outlier_ratio outlier ratio
		  */
		inline void
			setOutlierRatio(double outlier_ratio)
		{
			outlier_ratio_ = outlier_ratio;
		}
		// 设置邻域搜索方式（前面枚举定义的 KDTREE / DIRECT26 / DIRECT7 / DIRECT1）
		inline void setNeighborhoodSearchMethod(NeighborSearchMethod method) {
			search_method = method;
		}

		/** \brief Get the registration alignment probability.
		  * \return transformation probability
		  */
		// trans_probability_：配准完成后的 变换概率（NDT 对数似然值转化的衡量指标）
		inline double
			getTransformationProbability() const
		{
			return (trans_probability_);
		}

		/** \brief Get the number of iterations required to calculate alignment.
		  * \return final number of iterations
		  */
		// nr_iterations_：算法最终迭代次数
		inline int
			getFinalNumIteration() const
		{
			return (nr_iterations_);
		}

		/** \brief Convert 6 element transformation vector to affine transformation.
		  * \param[in] x transformation vector of the form [x, y, z, roll, pitch, yaw]
		  * \param[out] trans affine transform corresponding to given transformation vector
		  */
		// 将 6 维向量 [x, y, z, roll, pitch, yaw] 转换为 4x4 变换矩阵或 Eigen::Affine3f。
		// 用于表示 位姿（平移 + 旋转）
		static void
			convertTransform(const Eigen::Matrix<double, 6, 1> &x, Eigen::Affine3f &trans)
		{
			trans = Eigen::Translation<float, 3>(float(x(0)), float(x(1)), float(x(2))) *
				Eigen::AngleAxis<float>(float(x(3)), Eigen::Vector3f::UnitX()) *
				Eigen::AngleAxis<float>(float(x(4)), Eigen::Vector3f::UnitY()) *
				Eigen::AngleAxis<float>(float(x(5)), Eigen::Vector3f::UnitZ());
		}

		/** \brief Convert 6 element transformation vector to transformation matrix.
		  * \param[in] x transformation vector of the form [x, y, z, roll, pitch, yaw]
		  * \param[out] trans 4x4 transformation matrix corresponding to given transformation vector
		  */
		static void
			convertTransform(const Eigen::Matrix<double, 6, 1> &x, Eigen::Matrix4f &trans)
		{
			Eigen::Affine3f _affine;
			convertTransform(x, _affine);
			trans = _affine.matrix();
		}

		// negative log likelihood function
		// lower is better
		// 评分函数
		// 负对数似然函数：用于计算 NDT 模型下某一变换对应的匹配得分（score）。分数越低，表示匹配效果越好。
		double calculateScore(const PointCloudSource& cloud) const;

	protected:
		// protected范围，NDT 算法的内部实现，包括对每个迭代步骤的计算和线搜索处理
		using pcl::Registration<PointSource, PointTarget>::reg_name_;
		using pcl::Registration<PointSource, PointTarget>::getClassName;
		using pcl::Registration<PointSource, PointTarget>::input_;
		using pcl::Registration<PointSource, PointTarget>::indices_;
		using pcl::Registration<PointSource, PointTarget>::target_;
		using pcl::Registration<PointSource, PointTarget>::nr_iterations_;
		using pcl::Registration<PointSource, PointTarget>::max_iterations_;
		using pcl::Registration<PointSource, PointTarget>::previous_transformation_;
		using pcl::Registration<PointSource, PointTarget>::final_transformation_;
		using pcl::Registration<PointSource, PointTarget>::transformation_;
		using pcl::Registration<PointSource, PointTarget>::transformation_epsilon_;
		using pcl::Registration<PointSource, PointTarget>::converged_;
		using pcl::Registration<PointSource, PointTarget>::corr_dist_threshold_;
		using pcl::Registration<PointSource, PointTarget>::inlier_threshold_;

		using pcl::Registration<PointSource, PointTarget>::update_visualizer_;

		/** \brief Estimate the transformation and returns the transformed source (input) as output.
		  * \param[out] output the resultant input transformed point cloud dataset
		  */
		// 不带初始位姿的重载，默认使用单位矩阵作为初始变换
		virtual void
			computeTransformation(PointCloudSource &output)
		{
			computeTransformation(output, Eigen::Matrix4f::Identity());
		}

		/** \brief Estimate the transformation and returns the transformed source (input) as output.
		  * \param[out] output the resultant input transformed point cloud dataset
		  * \param[in] guess the initial gross estimation of the transformation
		  */
		// 核心配准函数：给定初始变换 guess，执行 NDT，对源点云进行若干轮迭代，输出最终配准到目标点云的结果。
		virtual void
			computeTransformation(PointCloudSource &output, const Eigen::Matrix4f &guess);

		/** \brief Initiate covariance voxel structure. */
		// 构建体素栅格：调用 VoxelGridCovariance 的 setLeafSize 设置体素尺寸
		// setInputCloud：传入目标点云
		// filter(true)：构建体素网格，并 为每个体素计算均值和协方差
		// 目标点云被分块到若干体素中，每个体素记录了点云分布的统计信息
		void inline
			init()
		{
			target_cells_.setLeafSize(resolution_, resolution_, resolution_);
			target_cells_.setInputCloud(target_);
			// Initiate voxel structure.
			target_cells_.filter(true);
		}

		/** \brief Compute derivatives of probability function w.r.t. the transformation vector.
		  * \note Equation 6.10, 6.12 and 6.13 [Magnusson 2009].
		  * \param[out] score_gradient the gradient vector of the probability function w.r.t. the transformation vector
		  * \param[out] hessian the hessian matrix of the probability function w.r.t. the transformation vector
		  * \param[in] trans_cloud transformed point cloud
		  * \param[in] p the current transform vector
		  * \param[in] compute_hessian flag to calculate hessian, unnecessary for step calculation.
		  */
		// 在 NDT 的迭代中，会基于当前位姿参数 p（6D），将源点云变换后得到 trans_cloud，并计算：
		// score_gradient：目标函数（负对数似然）的梯度。
		// hessian：目标函数的hessian matrix。
		// 这两个量用于基于Gaussian-Newton或类似方法来迭代求解最优位姿
		double
			computeDerivatives(Eigen::Matrix<double, 6, 1> &score_gradient,
				Eigen::Matrix<double, 6, 6> &hessian,
				PointCloudSource &trans_cloud,
				Eigen::Matrix<double, 6, 1> &p,
				bool compute_hessian = true);

		/** \brief Compute individual point contributions to derivatives of probability function w.r.t. the transformation vector.
		  * \note Equation 6.10, 6.12 and 6.13 [Magnusson 2009].
		  * \param[in,out] score_gradient the gradient vector of the probability function w.r.t. the transformation vector
		  * \param[in,out] hessian the hessian matrix of the probability function w.r.t. the transformation vector
		  * \param[in] x_trans transformed point minus mean of occupied covariance voxel
		  * \param[in] c_inv covariance of occupied covariance voxel
		  * \param[in] compute_hessian flag to calculate hessian, unnecessary for step calculation.
		  */
		// 更新单个点对梯度与海森矩阵的贡献。
		// 每个落入同一体素的点，都要计算相应的误差并加到总梯度和海森矩阵上
		double
			updateDerivatives(Eigen::Matrix<double, 6, 1> &score_gradient,
				Eigen::Matrix<double, 6, 6> &hessian,
				const Eigen::Matrix<float, 4, 6> &point_gradient_,
				const Eigen::Matrix<float, 24, 6> &point_hessian_,
				const Eigen::Vector3d &x_trans, const Eigen::Matrix3d &c_inv,
				bool compute_hessian = true) const;

		/** \brief Precompute angular components of derivatives.
		  * \note Equation 6.19 and 6.21 [Magnusson 2009].
		  * \param[in] p the current transform vector
		  * \param[in] compute_hessian flag to calculate hessian, unnecessary for step calculation.
		  */
		// 对 旋转量（roll/pitch/yaw） 预先计算一些中间量，方便后续快速更新梯度和海森矩阵
		void
			computeAngleDerivatives(Eigen::Matrix<double, 6, 1> &p, bool compute_hessian = true);

		/** \brief Compute point derivatives.
		  * \note Equation 6.18-21 [Magnusson 2009].
		  * \param[in] x point from the input cloud
		  * \param[in] compute_hessian flag to calculate hessian, unnecessary for step calculation.
		  */
		// 计算 点在局部坐标变换下 的偏导信息，这些信息会被用来求梯度和海森矩阵
		void
			computePointDerivatives(Eigen::Vector3d &x, Eigen::Matrix<double, 3, 6>& point_gradient_, Eigen::Matrix<double, 18, 6>& point_hessian_, bool compute_hessian = true) const;

		void
			computePointDerivatives(Eigen::Vector3d &x, Eigen::Matrix<float, 4, 6>& point_gradient_, Eigen::Matrix<float, 24, 6>& point_hessian_, bool compute_hessian = true) const;

		/** \brief Compute hessian of probability function w.r.t. the transformation vector.
		  * \note Equation 6.13 [Magnusson 2009].
		  * \param[out] hessian the hessian matrix of the probability function w.r.t. the transformation vector
		  * \param[in] trans_cloud transformed point cloud
		  * \param[in] p the current transform vector
		  */
		// 单独计算 海森矩阵 的函数，有时可以只更新梯度，不更新海森矩阵，以节省计算时间
		void
			computeHessian(Eigen::Matrix<double, 6, 6> &hessian,
				PointCloudSource &trans_cloud,
				Eigen::Matrix<double, 6, 1> &p);

		/** \brief Compute individual point contributions to hessian of probability function w.r.t. the transformation vector.
		  * \note Equation 6.13 [Magnusson 2009].
		  * \param[in,out] hessian the hessian matrix of the probability function w.r.t. the transformation vector
		  * \param[in] x_trans transformed point minus mean of occupied covariance voxel
		  * \param[in] c_inv covariance of occupied covariance voxel
		  */
		void
			updateHessian(Eigen::Matrix<double, 6, 6> &hessian,
				const Eigen::Matrix<double, 3, 6> &point_gradient_,
				const Eigen::Matrix<double, 18, 6> &point_hessian_,
				const Eigen::Vector3d &x_trans, const Eigen::Matrix3d &c_inv) const;

		/** \brief Compute line search step length and update transform and probability derivatives using More-Thuente method.
		  * \note Search Algorithm [More, Thuente 1994]
		  * \param[in] x initial transformation vector, \f$ x \f$ in Equation 1.3 (Moore, Thuente 1994) and \f$ \vec{p} \f$ in Algorithm 2 [Magnusson 2009]
		  * \param[in] step_dir descent direction, \f$ p \f$ in Equation 1.3 (Moore, Thuente 1994) and \f$ \delta \vec{p} \f$ normalized in Algorithm 2 [Magnusson 2009]
		  * \param[in] step_init initial step length estimate, \f$ \alpha_0 \f$ in Moore-Thuente (1994) and the normal of \f$ \delta \vec{p} \f$ in Algorithm 2 [Magnusson 2009]
		  * \param[in] step_max maximum step length, \f$ \alpha_max \f$ in Moore-Thuente (1994)
		  * \param[in] step_min minimum step length, \f$ \alpha_min \f$ in Moore-Thuente (1994)
		  * \param[out] score final score function value, \f$ f(x + \alpha p) \f$ in Equation 1.3 (Moore, Thuente 1994) and \f$ score \f$ in Algorithm 2 [Magnusson 2009]
		  * \param[in,out] score_gradient gradient of score function w.r.t. transformation vector, \f$ f'(x + \alpha p) \f$ in Moore-Thuente (1994) and \f$ \vec{g} \f$ in Algorithm 2 [Magnusson 2009]
		  * \param[out] hessian hessian of score function w.r.t. transformation vector, \f$ f''(x + \alpha p) \f$ in Moore-Thuente (1994) and \f$ H \f$ in Algorithm 2 [Magnusson 2009]
		  * \param[in,out] trans_cloud transformed point cloud, \f$ X \f$ transformed by \f$ T(\vec{p},\vec{x}) \f$ in Algorithm 2 [Magnusson 2009]
		  * \return final step length
		  */
		// More-Thuente 线搜索：用于确定在梯度下降/牛顿迭代时，每次迭代的步长。
		double
			computeStepLengthMT(const Eigen::Matrix<double, 6, 1> &x,
				Eigen::Matrix<double, 6, 1> &step_dir,
				double step_init,
				double step_max, double step_min,
				double &score,
				Eigen::Matrix<double, 6, 1> &score_gradient,
				Eigen::Matrix<double, 6, 6> &hessian,
				PointCloudSource &trans_cloud);

		/** \brief Update interval of possible step lengths for More-Thuente method, \f$ I \f$ in More-Thuente (1994)
		  * \note Updating Algorithm until some value satisfies \f$ \psi(\alpha_k) \leq 0 \f$ and \f$ \phi'(\alpha_k) \geq 0 \f$
		  * and Modified Updating Algorithm from then on [More, Thuente 1994].
		  * \param[in,out] a_l first endpoint of interval \f$ I \f$, \f$ \alpha_l \f$ in Moore-Thuente (1994)
		  * \param[in,out] f_l value at first endpoint, \f$ f_l \f$ in Moore-Thuente (1994), \f$ \psi(\alpha_l) \f$ for Update Algorithm and \f$ \phi(\alpha_l) \f$ for Modified Update Algorithm
		  * \param[in,out] g_l derivative at first endpoint, \f$ g_l \f$ in Moore-Thuente (1994), \f$ \psi'(\alpha_l) \f$ for Update Algorithm and \f$ \phi'(\alpha_l) \f$ for Modified Update Algorithm
		  * \param[in,out] a_u second endpoint of interval \f$ I \f$, \f$ \alpha_u \f$ in Moore-Thuente (1994)
		  * \param[in,out] f_u value at second endpoint, \f$ f_u \f$ in Moore-Thuente (1994), \f$ \psi(\alpha_u) \f$ for Update Algorithm and \f$ \phi(\alpha_u) \f$ for Modified Update Algorithm
		  * \param[in,out] g_u derivative at second endpoint, \f$ g_u \f$ in Moore-Thuente (1994), \f$ \psi'(\alpha_u) \f$ for Update Algorithm and \f$ \phi'(\alpha_u) \f$ for Modified Update Algorithm
		  * \param[in] a_t trial value, \f$ \alpha_t \f$ in Moore-Thuente (1994)
		  * \param[in] f_t value at trial value, \f$ f_t \f$ in Moore-Thuente (1994), \f$ \psi(\alpha_t) \f$ for Update Algorithm and \f$ \phi(\alpha_t) \f$ for Modified Update Algorithm
		  * \param[in] g_t derivative at trial value, \f$ g_t \f$ in Moore-Thuente (1994), \f$ \psi'(\alpha_t) \f$ for Update Algorithm and \f$ \phi'(\alpha_t) \f$ for Modified Update Algorithm
		  * \return if interval converges
		  */
		// More-Thuente 线搜索 的辅助函数，维护可行的步长区间并选择新的试探步长
		bool
			updateIntervalMT(double &a_l, double &f_l, double &g_l,
				double &a_u, double &f_u, double &g_u,
				double a_t, double f_t, double g_t);

		/** \brief Select new trial value for More-Thuente method.
		  * \note Trial Value Selection [More, Thuente 1994], \f$ \psi(\alpha_k) \f$ is used for \f$ f_k \f$ and \f$ g_k \f$
		  * until some value satisfies the test \f$ \psi(\alpha_k) \leq 0 \f$ and \f$ \phi'(\alpha_k) \geq 0 \f$
		  * then \f$ \phi(\alpha_k) \f$ is used from then on.
		  * \note Interpolation Minimizer equations from Optimization Theory and Methods: Nonlinear Programming By Wenyu Sun, Ya-xiang Yuan (89-100).
		  * \param[in] a_l first endpoint of interval \f$ I \f$, \f$ \alpha_l \f$ in Moore-Thuente (1994)
		  * \param[in] f_l value at first endpoint, \f$ f_l \f$ in Moore-Thuente (1994)
		  * \param[in] g_l derivative at first endpoint, \f$ g_l \f$ in Moore-Thuente (1994)
		  * \param[in] a_u second endpoint of interval \f$ I \f$, \f$ \alpha_u \f$ in Moore-Thuente (1994)
		  * \param[in] f_u value at second endpoint, \f$ f_u \f$ in Moore-Thuente (1994)
		  * \param[in] g_u derivative at second endpoint, \f$ g_u \f$ in Moore-Thuente (1994)
		  * \param[in] a_t previous trial value, \f$ \alpha_t \f$ in Moore-Thuente (1994)
		  * \param[in] f_t value at previous trial value, \f$ f_t \f$ in Moore-Thuente (1994)
		  * \param[in] g_t derivative at previous trial value, \f$ g_t \f$ in Moore-Thuente (1994)
		  * \return new trial value
		  */
		double
			trialValueSelectionMT(double a_l, double f_l, double g_l,
				double a_u, double f_u, double g_u,
				double a_t, double f_t, double g_t);

		/** \brief Auxiliary function used to determine endpoints of More-Thuente interval.
		  * \note \f$ \psi(\alpha) \f$ in Equation 1.6 (Moore, Thuente 1994)
		  * \param[in] a the step length, \f$ \alpha \f$ in More-Thuente (1994)
		  * \param[in] f_a function value at step length a, \f$ \phi(\alpha) \f$ in More-Thuente (1994)
		  * \param[in] f_0 initial function value, \f$ \phi(0) \f$ in Moore-Thuente (1994)
		  * \param[in] g_0 initial function gradiant, \f$ \phi'(0) \f$ in More-Thuente (1994)
		  * \param[in] mu the step length, constant \f$ \mu \f$ in Equation 1.1 [More, Thuente 1994]
		  * \return sufficient decrease value
		  */
		inline double
			auxiliaryFunction_PsiMT(double a, double f_a, double f_0, double g_0, double mu = 1.e-4)
		{
			return (f_a - f_0 - mu * g_0 * a);
		}

		/** \brief Auxiliary function derivative used to determine endpoints of More-Thuente interval.
		  * \note \f$ \psi'(\alpha) \f$, derivative of Equation 1.6 (Moore, Thuente 1994)
		  * \param[in] g_a function gradient at step length a, \f$ \phi'(\alpha) \f$ in More-Thuente (1994)
		  * \param[in] g_0 initial function gradiant, \f$ \phi'(0) \f$ in More-Thuente (1994)
		  * \param[in] mu the step length, constant \f$ \mu \f$ in Equation 1.1 [More, Thuente 1994]
		  * \return sufficient decrease derivative
		  */
		inline double
			auxiliaryFunction_dPsiMT(double g_a, double g_0, double mu = 1.e-4)
		{
			return (g_a - mu * g_0);
		}
		// 这些成员变量存储了 NDT 算法的核心超参数 和 中间变量
		/** \brief The voxel grid generated from target cloud containing point means and covariances. */
		TargetGrid target_cells_;	// 目标点云的统计信息（均值+协方差）的体素网格

		//double fitness_epsilon_;

		/** \brief The side length of voxels. */
		// resolution_ / step_size_ / outlier_ratio_：对应论文中提到的 NDT 参数
		float resolution_;

		/** \brief The maximum step length. */
		double step_size_;

		/** \brief The ratio of outliers of points w.r.t. a normal distribution, Equation 6.7 [Magnusson 2009]. */
		double outlier_ratio_;

		// gauss_d1_, gauss_d2_, gauss_d3_ 作为 归一化系数（normalization constants），用于将点与体素高斯分布之间的距离转换为概率密度
		/** \brief The normalization constants used fit the point distribution to a normal distribution, Equation 6.8 [Magnusson 2009]. */
		double gauss_d1_, gauss_d2_, gauss_d3_;

		/** \brief The probability score of the transform applied to the input cloud, Equation 6.9 and 6.10 [Magnusson 2009]. */
		double trans_probability_;	// NDT 最终得到的变换质量评价

		/** \brief Precomputed Angular Gradient
		  *
		  * The precomputed angular derivatives for the jacobian of a transformation vector, Equation 6.19 [Magnusson 2009].
		  */
		Eigen::Vector3d j_ang_a_, j_ang_b_, j_ang_c_, j_ang_d_, j_ang_e_, j_ang_f_, j_ang_g_, j_ang_h_;

		Eigen::Matrix<float, 8, 4> j_ang;

		/** \brief Precomputed Angular Hessian
		  *
		  * The precomputed angular derivatives for the hessian of a transformation vector, Equation 6.19 [Magnusson 2009].
		  */
		Eigen::Vector3d h_ang_a2_, h_ang_a3_,
			h_ang_b2_, h_ang_b3_,
			h_ang_c2_, h_ang_c3_,
			h_ang_d1_, h_ang_d2_, h_ang_d3_,
			h_ang_e1_, h_ang_e2_, h_ang_e3_,
			h_ang_f1_, h_ang_f2_, h_ang_f3_;

		Eigen::Matrix<float, 16, 4> h_ang;

		/** \brief The first order derivative of the transformation of a point w.r.t. the transform vector, \f$ J_E \f$ in Equation 6.18 [Magnusson 2009]. */
  //      Eigen::Matrix<double, 3, 6> point_gradient_;

		/** \brief The second order derivative of the transformation of a point w.r.t. the transform vector, \f$ H_E \f$ in Equation 6.20 [Magnusson 2009]. */
  //      Eigen::Matrix<double, 18, 6> point_hessian_;

    int num_threads_;

	public:
		NeighborSearchMethod search_method;
		// 保证在使用 Eigen 动态分配对象时对齐（内存对齐）以优化矩阵运算
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};

}

#endif // PCL_REGISTRATION_NDT_H_
