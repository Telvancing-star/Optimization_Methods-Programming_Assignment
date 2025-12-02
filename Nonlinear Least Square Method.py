import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据
def load_data(file_path):
    data = pd.read_csv(file_path)
    t = data['t_i'].values
    y = data['y_i'].values
    return t, y

# 定义拟合函数、残差函数和目标函数
def model_function(x, t):
    x1, x2, x3, x4 = x
    numerator = x1 * (t**2 + x2 * t)
    denominator = t**2 + x3 * t + x4
    if isinstance(t, np.ndarray):
        result = np.zeros_like(t)
        valid_indices = denominator != 0
        result[valid_indices] = numerator[valid_indices] / denominator[valid_indices]
        result[~valid_indices] = np.nan
        return result
    else:
        if denominator != 0:
            return numerator / denominator
        else:
            return np.nan
def residual_function(x, t, y):
    return y - model_function(x, t)
def objective_function(x, t, y):
    residuals = residual_function(x, t, y)
    return 0.5 * np.sum(residuals**2)

# 计算雅可比矩阵
def jacobian_matrix(x, t):
    x1, x2, x3, x4 = x
    m = len(t)
    J = np.zeros((m, 4))
    denominator = t**2 + x3 * t + x4
    denominator_squared = denominator**2
    numerator = x1 * (t**2 + x2 * t)
    valid_indices = denominator != 0
    J[valid_indices, 0] = -(t[valid_indices]**2 + x2 * t[valid_indices]) / denominator[valid_indices]
    J[valid_indices, 1] = -(x1 * t[valid_indices]) / denominator[valid_indices]
    J[valid_indices, 2] = numerator[valid_indices] * t[valid_indices] / denominator_squared[valid_indices]
    J[valid_indices, 3] = numerator[valid_indices] / denominator_squared[valid_indices]
    J[~valid_indices, :] = 0
    return J

# 添加收敛判断和结果分析功能
def analyze_results(x_opt, t, y):
    y_fit = model_function(x_opt, t)
    residuals = y - y_fit
    m = len(y)
    n = len(x_opt)
    # 残差平方和
    rss = np.sum(residuals**2)
    # 总平方和
    tss = np.sum((y - np.mean(y))**2)
    # 决定系数 R²
    r_squared = 1 - rss / tss
    adjusted_r_squared = 1 - (rss / (m - n)) / (tss / (m - 1)) if m > n else float('nan')
    # 均方误差
    mse = rss / m
    # 均方根误差
    rmse = np.sqrt(mse)
    # 平均绝对误差
    mae = np.mean(np.abs(residuals))
    # 残差的标准差
    residual_std = np.std(residuals, ddof=1)
    # 参数标准差的估计
    J = jacobian_matrix(x_opt, t)
    JTJ = J.T @ J
    try:
        cov_matrix = residual_std**2 * np.linalg.inv(JTJ)
        param_std = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        cov_matrix = residual_std**2 * np.linalg.pinv(JTJ)
        param_std = np.sqrt(np.diag(cov_matrix))
        print("警告: 计算参数标准差时矩阵接近奇异")
    return {
        '参数估计': x_opt,
        '参数标准差': param_std,
        '残差平方和': rss,
        '总平方和': tss,
        '决定系数(R²)': r_squared,
        '调整后的决定系数': adjusted_r_squared,
        '均方误差(MSE)': mse,
        '均方根误差(RMSE)': rmse,
        '平均绝对误差(MAE)': mae,
        '残差标准差': residual_std,
        '拟合值': y_fit,
        '残差': residuals
    }

# 实现Gauss-Newton迭代算法
def gauss_newton(x0, t, y, tol=1e-6, max_iter=100, verbose=False):
    x = x0.copy()
    obj_values = [objective_function(x, t, y)]
    gradient_norms = []
    iter_count = 0
    converged = False
    # 迭代主循环
    for i in range(max_iter):
        r = residual_function(x, t, y)
        J = jacobian_matrix(x, t)
        JTJ = J.T @ J
        JTr = J.T @ r
        gradient_norm = np.linalg.norm(JTr)
        gradient_norms.append(gradient_norm)
        # 求解线性方程组 JTJ * delta_x = -JTr
        try:
            delta_x = np.linalg.solve(JTJ, -JTr)
        except np.linalg.LinAlgError:
            delta_x = np.linalg.pinv(JTJ) @ (-JTr)
            print("警告: 矩阵接近奇异，使用伪逆求解")
        # 更新参数
        x_new = x + delta_x
        new_obj_value = objective_function(x_new, t, y)
        obj_values.append(new_obj_value)
        delta_obj = np.abs(obj_values[-2] - obj_values[-1])
        delta_x_norm = np.linalg.norm(delta_x)
        if (delta_obj < tol or delta_x_norm < tol or gradient_norm < tol):
            if i >= 3:
                recent_delta_objs = np.abs(np.diff(obj_values[-4:]))
                recent_delta_x_norms = [np.linalg.norm(delta_x)]
                if (np.all(recent_delta_objs < tol) or 
                    np.all(np.array(recent_delta_x_norms) < tol)):
                    converged = True
                    x = x_new
                    iter_count = i + 1
                    if verbose:
                        print(f"迭代 {iter_count} 收敛！")
                    break
        x = x_new
        iter_count = i + 1
        if verbose:
            print(f"迭代 {iter_count}: 目标函数值 = {new_obj_value:.6e}, 梯度范数 = {gradient_norm:.6e}, Δx范数 = {delta_x_norm:.6e}")
    if not converged and iter_count >= max_iter:
        print(f"警告: 达到最大迭代次数 {max_iter}，可能未收敛")
    return x, obj_values, iter_count, converged, gradient_norms

# 打印结果分析
def print_analysis(results, t, y):
    print("\n=== 拟合结果分析 ===")
    print(f"参数估计: {results['参数估计']}")
    print(f"参数标准差: {results['参数标准差']}")
    print(f"残差平方和 (RSS): {results['残差平方和']:.6f}")
    print(f"总平方和 (TSS): {results['总平方和']:.6f}")
    print(f"决定系数 (R²): {results['决定系数(R²)']:.6f}")
    print(f"调整后的决定系数: {results['调整后的决定系数']:.6f}")
    print(f"均方误差 (MSE): {results['均方误差(MSE)']:.6f}")
    print(f"均方根误差 (RMSE): {results['均方根误差(RMSE)']:.6f}")
    print(f"平均绝对误差 (MAE): {results['平均绝对误差(MAE)']:.6f}")
    print(f"残差标准差: {results['残差标准差']:.6f}")
    
    # 打印参数估计及其95%置信区间
    print("\n参数估计及其95%置信区间:")
    for i, (param, std) in enumerate(zip(results['参数估计'], results['参数标准差'])):
        # 95%置信区间的临界值（假设t分布，自由度很大时接近1.96）
        critical_value = 1.96
        lower = param - critical_value * std
        upper = param + critical_value * std
        print(f"x{i+1}: {param:.6f} [{lower:.6f}, {upper:.6f}] (标准差: {std:.6f})")
    
    # 打印拟合值和真实值的比较
    print("\n所有数据点的拟合结果:")
    for i in range(len(results['拟合值'])):
        y_true = y[i] if i < len(y) else float('nan')
        y_fit = results['拟合值'][i]
        residual = results['残差'][i]
        print(f"t={t[i]:.4f}: y_true={y_true:.6f}, y_fit={y_fit:.6f}, 残差={residual:.6f}")

# 可视化拟合结果
def visualize_results(t, y, optimal_params, obj_values, gradient_norms, method_name='', show_plot=False):
    # 计算拟合值
    y_fit = model_function(optimal_params, t)
    residuals = y - y_fit
    
    # 创建一个2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 数据点和拟合曲线对比图
    axes[0, 0].scatter(t, y, color='blue', marker='o', label='实际数据点')
    
    # 生成更多的点来绘制平滑的拟合曲线
    t_smooth = np.linspace(min(t), max(t), 100)
    y_smooth = model_function(optimal_params, t_smooth)
    axes[0, 0].plot(t_smooth, y_smooth, color='red', linewidth=2, label='拟合曲线')
    
    axes[0, 0].set_xlabel('t')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title('数据点与拟合曲线对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. 残差图
    axes[0, 1].scatter(t, residuals, color='green', marker='o', label='残差')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0, 1].set_xlabel('t')
    axes[0, 1].set_ylabel('残差')
    axes[0, 1].set_title('残差图')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. 目标函数值收敛过程
    axes[1, 0].semilogy(range(len(obj_values)), obj_values, color='purple', linewidth=2, label='目标函数值')
    axes[1, 0].set_xlabel('迭代次数')
    axes[1, 0].set_ylabel('目标函数值 (对数尺度)')
    axes[1, 0].set_title('目标函数值收敛过程')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. 梯度范数收敛过程
    axes[1, 1].semilogy(range(len(gradient_norms)), gradient_norms, color='orange', linewidth=2, label='梯度范数')
    axes[1, 1].set_xlabel('迭代次数')
    axes[1, 1].set_ylabel('梯度范数 (对数尺度)')
    axes[1, 1].set_title('梯度范数收敛过程')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 调整子图布局
    plt.tight_layout()
    
    # 添加全局标题
    if method_name:
        fig.suptitle(f'非线性最小二乘拟合结果 - {method_name} (最优参数: {optimal_params.round(6)})', fontsize=16, y=1.02)
    else:
        fig.suptitle(f'非线性最小二乘拟合结果 (最优参数: {optimal_params.round(6)})', fontsize=16, y=1.02)
    
    # 保存图片
    if method_name:
        filename = f'Nonlinear Least Square Method/fitting_results_{method_name}.png'
    else:
        filename = 'Nonlinear Least Square Method/fitting_results.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n拟合结果可视化已保存为 '{filename.split('/')[-1]}'")
    
    # 显示图形（可选）
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

# 探索不同初始点对收敛性的影响
def explore_initial_points(t, y, initial_points, tol=1e-6, max_iter=100, show_plot=False):
    print("\n=== 探索不同初始点对收敛性的影响 ===")
    results = {}
    for i, x0 in enumerate(initial_points):
        print(f"\n测试初始点 {i+1}: {x0}")
        optimal_params, obj_values, iter_count, converged, gradient_norms = gauss_newton(
            x0, t, y, tol=tol, max_iter=max_iter, verbose=False
        )
        result = analyze_results(optimal_params, t, y)
        results[i] = {
            'initial_point': x0,
            'optimal_params': optimal_params,
            'obj_values': obj_values,
            'iter_count': iter_count,
            'converged': converged,
            'gradient_norms': gradient_norms,
            'analysis': result
        }
        print(f"  迭代次数: {iter_count}")
        print(f"  是否收敛: {converged}")
        print(f"  最优参数: {optimal_params.round(6)}")
        print(f"  最终目标函数值: {obj_values[-1]:.6e}")
        print(f"  决定系数 R²: {result['决定系数(R²)']:.6f}")
    visualize_initial_points_comparison(t, y, results, show_plot=show_plot)
    return results

# 可视化不同初始点的结果比较
def visualize_initial_points_comparison(t, y, results, show_plot=False):
    """
    可视化不同初始点的收敛结果比较
    """
    num_initial_points = len(results)
    
    # 创建图形
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 不同初始点的拟合曲线对比
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(t, y, color='blue', marker='o', label='实际数据点')
    
    # 生成平滑的t值用于绘制曲线
    t_smooth = np.linspace(min(t), max(t), 100)
    colors = plt.cm.viridis(np.linspace(0, 1, num_initial_points))
    
    for i, (key, result) in enumerate(results.items()):
        x_opt = result['optimal_params']
        y_smooth = model_function(x_opt, t_smooth)
        ax1.plot(t_smooth, y_smooth, color=colors[i], linewidth=2, 
                 label=f'初始点 {i+1}: {result["initial_point"].round(3)}')
    
    ax1.set_xlabel('t')
    ax1.set_ylabel('y')
    ax1.set_title('不同初始点的拟合曲线对比')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    
    # 2. 不同初始点的收敛路径（目标函数值）
    ax2 = fig.add_subplot(2, 2, 2)
    for i, (key, result) in enumerate(results.items()):
        obj_values = result['obj_values']
        ax2.semilogy(range(len(obj_values)), obj_values, color=colors[i], 
                     linewidth=2, label=f'初始点 {i+1}')
    
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('目标函数值 (对数尺度)')
    ax2.set_title('不同初始点的目标函数值收敛过程')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 不同初始点的迭代次数比较
    ax3 = fig.add_subplot(2, 2, 3)
    iter_counts = [results[i]['iter_count'] for i in results]
    initial_point_labels = [f'初始点 {i+1}' for i in results]
    ax3.bar(initial_point_labels, iter_counts, color=colors)
    ax3.set_xlabel('初始点')
    ax3.set_ylabel('迭代次数')
    ax3.set_title('不同初始点的迭代次数比较')
    ax3.grid(True, axis='y')
    
    # 4. 不同初始点的最终目标函数值比较
    ax4 = fig.add_subplot(2, 2, 4)
    final_obj_values = [results[i]['obj_values'][-1] for i in results]
    ax4.bar(initial_point_labels, final_obj_values, color=colors)
    ax4.set_xlabel('初始点')
    ax4.set_ylabel('最终目标函数值')
    ax4.set_title('不同初始点的最终目标函数值比较')
    ax4.grid(True, axis='y')
    
    # 调整布局
    plt.tight_layout()
    
    # 添加全局标题
    fig.suptitle('不同初始点对Gauss-Newton算法收敛性的影响', fontsize=16, y=1.02)
    
    # 保存图片
    plt.savefig('Nonlinear Least Square Method/initial_points_comparison.png', dpi=300, bbox_inches='tight')
    print("\n初始点比较结果可视化已保存为 'initial_points_comparison.png'")
    
    # 显示图形（可选）
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

# 线性搜索（Armijo准则）
def exact_line_search(x, delta_x, t, y):
    """
    使用Armijo准则进行线性搜索，找到最优步长alpha
    
    参数:
    x: 当前参数
    delta_x: 搜索方向
    t: 自变量数据
    y: 因变量数据
    
    返回:
    alpha: 最优步长
    """
    # Armijo准则参数
    alpha0 = 1.0  # 初始步长
    rho = 0.5     # 步长收缩因子
    c = 1e-4      # Armijo常数
    
    # 当前目标函数值
    f0 = objective_function(x, t, y)
    
    # 计算方向导数
    r = residual_function(x, t, y)
    J = jacobian_matrix(x, t)
    g = -J.T @ r
    gTd = np.dot(g, delta_x)
    
    # Armijo准则
    alpha = alpha0
    while True:
        x_new = x + alpha * delta_x
        f_new = objective_function(x_new, t, y)
        if f_new <= f0 + c * alpha * gTd:
            break
        alpha *= rho
        if alpha < 1e-10:
            break
    
    return alpha

# 带Armijo准则线性搜索的阻尼Gauss-Newton方法
def damped_gauss_newton_with_exact_line_search(x0, t, y, tol=1e-6, max_iter=100, verbose=False):
    print("\n=== 带Armijo准则线性搜索的阻尼Gauss-Newton方法 ===")
    x = x0.copy()
    obj_values = [objective_function(x, t, y)]
    gradient_norms = []
    iter_count = 0
    converged = False
    # 迭代主循环
    for i in range(max_iter):
        r = residual_function(x, t, y)
        J = jacobian_matrix(x, t)
        JTJ = J.T @ J
        JTr = J.T @ r
        gradient_norm = np.linalg.norm(JTr)
        gradient_norms.append(gradient_norm)
        try:
            delta_x = np.linalg.solve(JTJ, -JTr)
        except np.linalg.LinAlgError:
            delta_x = np.linalg.pinv(JTJ) @ (-JTr)
            print("警告: 矩阵接近奇异，使用伪逆求解")
        alpha = exact_line_search(x, delta_x, t, y)
        x_new = x + alpha * delta_x
        new_obj_value = objective_function(x_new, t, y)
        obj_values.append(new_obj_value)
        delta_obj = np.abs(obj_values[-2] - obj_values[-1])
        delta_x_norm = np.linalg.norm(alpha * delta_x)
        if (delta_obj < tol or delta_x_norm < tol or gradient_norm < tol):
            if i >= 3:
                recent_delta_objs = np.abs(np.diff(obj_values[-4:]))
                if np.all(recent_delta_objs < tol):
                    converged = True
                    x = x_new
                    iter_count = i + 1
                    if verbose:
                        print(f"迭代 {iter_count} 收敛！")
                    break
        x = x_new
        iter_count = i + 1
        if verbose:
            print(f"迭代 {iter_count}: 目标函数值 = {new_obj_value:.6e}, 梯度范数 = {gradient_norm:.6e}, ")
            print(f"       Δx范数 = {delta_x_norm:.6e}, 步长 alpha = {alpha:.6e}")
    if not converged and iter_count >= max_iter:
        print(f"警告: 达到最大迭代次数 {max_iter}，可能未收敛")
    return x, obj_values, iter_count, converged, gradient_norms

# 测试数据加载和函数定义
if __name__ == "__main__":
    # 加载数据
    file_path = 'data.csv'
    t, y = load_data(file_path)
    print("=== 非线性最小二乘问题求解 ===")
    print("使用Gauss-Newton方法求解教材149页习题")
    print("模型函数: f(x, t) = x1(t² + x2*t) / (t² + x3*t + x4)")
    print(f"\n数据点数量: {len(t)}")
    
    # 1. 原始Gauss-Newton方法
    print("\n=== 1. 原始Gauss-Newton方法 ===")
    x0 = np.array([1.0, 1.0, 1.0, 1.0])  # 初始猜测
    print(f"\n初始参数: {x0}")
    optimal_params_gauss, obj_values_gauss, iter_count_gauss, converged_gauss, gradient_norms_gauss = gauss_newton(
        x0, t, y, tol=1e-6, max_iter=100, verbose=True
    )
    results_gauss = analyze_results(optimal_params_gauss, t, y)
    print(f"\n原始Gauss-Newton方法结果:")
    print(f"  迭代次数: {iter_count_gauss}")
    print(f"  是否收敛: {converged_gauss}")
    print(f"  最优参数: {optimal_params_gauss.round(6)}")
    print(f"  最终目标函数值: {obj_values_gauss[-1]:.6e}")
    print(f"  决定系数 R²: {results_gauss['决定系数(R²)']:.6f}")
    # 可视化原始Gauss-Newton方法的结果
    print("\n可视化原始Gauss-Newton方法的拟合结果...")
    visualize_results(t, y, optimal_params_gauss, obj_values_gauss, gradient_norms_gauss, method_name='original_gauss_newton', show_plot=False)
    
    # 2. 探索不同初始点的影响（使用5个初始点）
    print("\n=== 2. 探索不同初始点对收敛性的影响 ===")
    initial_points = [
        np.array([1.0, 1.0, 1.0, 1.0]),      # 初始猜测1
        np.array([2.0, 0.5, 1.5, 0.8]),      # 初始猜测2
        np.array([0.5, 2.0, 0.5, 2.0]),      # 初始猜测3
        np.array([1.5, 1.5, 1.5, 1.5]),      # 初始猜测4
        np.array([1.0, 0.0, 1.0, 0.0])       # 初始猜测5
    ]
    initial_points_results = explore_initial_points(t, y, initial_points, show_plot=False)
    
    # 3. 带Armijo准则线性搜索的阻尼Gauss-Newton方法
    print("\n=== 3. 带Armijo准则线性搜索的阻尼Gauss-Newton方法 ===")
    optimal_params_damped, obj_values_damped, iter_count_damped, converged_damped, gradient_norms_damped = damped_gauss_newton_with_exact_line_search(
        x0, t, y, tol=1e-6, max_iter=100, verbose=True
    )
    results_damped = analyze_results(optimal_params_damped, t, y)
    print(f"\n带Armijo准则线性搜索的阻尼Gauss-Newton方法结果:")
    print(f"  迭代次数: {iter_count_damped}")
    print(f"  是否收敛: {converged_damped}")
    print(f"  最优参数: {optimal_params_damped.round(6)}")
    print(f"  最终目标函数值: {obj_values_damped[-1]:.6e}")
    print(f"  决定系数 R²: {results_damped['决定系数(R²)']:.6f}")
    # 可视化阻尼Gauss-Newton方法的结果
    print("\n可视化阻尼Gauss-Newton方法的拟合结果...")
    visualize_results(t, y, optimal_params_damped, obj_values_damped, gradient_norms_damped, method_name='damped_gauss_newton', show_plot=False)
    
    # 4. 两种方法结果简要比较
    print("\n=== 4. 两种方法结果比较 ===")
    print(f"\n迭代次数比较: 原始Gauss-Newton ({iter_count_gauss}) vs 阻尼Gauss-Newton ({iter_count_damped})")
    print(f"目标函数值比较: 原始Gauss-Newton ({obj_values_gauss[-1]:.6e}) vs 阻尼Gauss-Newton ({obj_values_damped[-1]:.6e})")
    print(f"收敛情况: 原始Gauss-Newton ({converged_gauss}) vs 阻尼Gauss-Newton ({converged_damped})")
    
    # 5. 比较收敛过程（生成对比图）
    print("\n=== 5. 收敛过程比较 ===")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 目标函数值收敛过程比较
    axes[0].semilogy(range(len(obj_values_gauss)), obj_values_gauss, color='blue', linewidth=2, label='原始Gauss-Newton')
    axes[0].semilogy(range(len(obj_values_damped)), obj_values_damped, color='red', linewidth=2, label='阻尼Gauss-Newton（Armijo准则线性搜索）')
    axes[0].set_xlabel('迭代次数')
    axes[0].set_ylabel('目标函数值 (对数尺度)')
    axes[0].set_title('目标函数值收敛过程比较')
    axes[0].legend()
    axes[0].grid(True)
    
    # 梯度范数收敛过程比较
    axes[1].semilogy(range(len(gradient_norms_gauss)), gradient_norms_gauss, color='blue', linewidth=2, label='原始Gauss-Newton')
    axes[1].semilogy(range(len(gradient_norms_damped)), gradient_norms_damped, color='red', linewidth=2, label='阻尼Gauss-Newton（Armijo准则线性搜索）')
    axes[1].set_xlabel('迭代次数')
    axes[1].set_ylabel('梯度范数 (对数尺度)')
    axes[1].set_title('梯度范数收敛过程比较')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('Nonlinear Least Square Method/comparison_results.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("\n方法比较结果已保存为 'comparison_results.png'")
    
    print("\n=== 程序执行完成 ===")
