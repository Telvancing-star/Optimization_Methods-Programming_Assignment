import math
import matplotlib.pyplot as plt

# 目标函数 φ(α) = 1 − αe^{-α^2}
def objective_function(alpha):
    # 定义目标函数φ(α) = 1 − αe^{-α^2}
    return 1 - alpha * math.exp(-alpha**2)

# 目标函数的导数，用于验证极小值点
def objective_function_derivative(alpha):
    # 根据微积分，φ'(α) = -e^{-α^2} + 2α^2e^{-α^2} = e^{-α^2}(2α^2 - 1)
    return math.exp(-alpha**2) * (2 * alpha**2 - 1)

# 0.618法（黄金分割法）实现
def golden_section_search(f, a, b, tol=0.01, max_iter=100):
    """使用黄金分割法（0.618法）求解单峰函数的极小值点
    Args:
        f: 目标函数
        a: 初始区间左端点
        b: 初始区间右端点
        tol: 精度要求
        max_iter: 最大迭代次数
    Returns:
        tuple: (极小值点近似值, 迭代次数, 迭代历史)
    """
    # 黄金分割比例
    gr = (math.sqrt(5) - 1) / 2  # 约0.618
    iter_count = 0
    # 记录迭代过程
    iterations = []
    while (b - a) > tol and iter_count < max_iter:
        # 计算两个内分点
        p = a + (1 - gr) * (b - a)
        q = a + gr * (b - a)
        # 计算函数值
        f_p = f(p)
        f_q = f(q)
        # 记录当前迭代信息
        iterations.append({
            'iter': iter_count + 1,
            'a': a,
            'b': b,
            'p': p,
            'q': q,
            'f_p': f_p,
            'f_q': f_q,
            'interval_length': b - a
        })
        # 比较函数值，缩小区间
        if f_p < f_q:
            b = q
        else:
            a = p
        iter_count += 1
    # 返回区间中点作为极小值点的近似
    min_point = (a + b) / 2
    return min_point, iter_count, iterations

# 改进的黄金分割法，结合二次插值法提高精度
def improved_golden_section_search(f, a, b, tol=0.01, max_iter=100):
    """使用改进的黄金分割法求解单峰函数的极小值点
    在黄金分割法收敛后，对最终区间进行二次插值优化以提高精度
    Args:
        f: 目标函数
        a: 初始区间左端点
        b: 初始区间右端点
        tol: 精度要求
        max_iter: 最大迭代次数
    Returns:
        tuple: (极小值点近似值, 迭代次数, 迭代历史)
    """
    # 首先使用标准黄金分割法找到一个较小的区间
    golden_min, golden_iter, golden_history = golden_section_search(f, a, b, tol, max_iter)
    
    # 获取黄金分割法收敛后的最终区间
    final_a = golden_history[-1]['a']
    final_b = golden_history[-1]['b']
    
    # 在最终区间内使用三点二次插值法进一步精确化
    # 选择三个点：区间端点和中点
    x1 = final_a
    x3 = final_b
    x2 = (x1 + x3) / 2
    f1 = f(x1)
    f2 = f(x2)
    f3 = f(x3)
    
    # 计算二次插值多项式的极小点
    numerator = f1 * (x2**2 - x3**2) + f2 * (x3**2 - x1**2) + f3 * (x1**2 - x2**2)
    denominator = 2 * (f1 * (x2 - x3) + f2 * (x3 - x1) + f3 * (x1 - x2))
    
    improved_min = 0.0
    if abs(denominator) < 1e-10:
        # 如果分母太小，使用黄金分割法的结果
        improved_min = golden_min
    else:
        # 计算二次插值的极小点
        improved_min = numerator / denominator
        # 确保在区间内
        improved_min = max(x1, min(x3, improved_min))
    
    # 记录改进后的信息
    improved_history = golden_history.copy()
    improved_history.append({
        'iter': golden_iter + 1,
        'a': x1,
        'b': x3,
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'improved_min': improved_min,
        'interval_length': x3 - x1,
        'method': 'quadratic_refinement'
    })
    
    return improved_min, golden_iter + 1, improved_history

# 三点二次插值法实现
def quadratic_interpolation_search(f, a, b, tol=0.01, max_iter=100):
    """使用三点二次插值法求解函数的极小值点
    Args:
        f: 目标函数
        a: 初始区间左端点
        b: 初始区间右端点
        tol: 精度要求
        max_iter: 最大迭代次数
    Returns:
        tuple: (极小值点近似值, 迭代次数, 迭代历史)
    """
    # 初始化三个点，确保满足高-低-高特征
    x1 = a
    x3 = b
    f1 = f(x1)
    f3 = f(x3)
    
    # 寻找满足高-低-高特征的中间点
    # 首先尝试中点
    x2 = (a + b) / 2
    f2 = f(x2)
    
    # 如果中点不满足高-低-高特征，在区间内搜索
    if not (f2 < f1 and f2 < f3):
        # 在区间内均匀采样多个点，寻找最小值点作为中间点
        sample_points = 10  # 采样点数
        best_x = x2
        best_f = f2
        
        for i in range(1, sample_points):
            x = a + i * (b - a) / sample_points
            fx = f(x)
            if fx < best_f:
                best_x = x
                best_f = fx
        
        # 确保中间点在区间内，且不等于端点
        x2 = best_x
        f2 = best_f
        
        # 如果仍然找不到满足条件的中间点，使用黄金分割比例选择点
        if not (f2 < f1 and f2 < f3):
            gr = (math.sqrt(5) - 1) / 2  # 黄金分割比例
            x2 = a + gr * (b - a)
            f2 = f(x2)
    
    # 确保三个点按顺序排列
    if x1 > x3:
        x1, x3 = x3, x1
        f1, f3 = f3, f1
    if x2 < x1:
        x1, x2 = x2, x1
        f1, f2 = f2, f1
    if x2 > x3:
        x2, x3 = x3, x2
        f2, f3 = f3, f2
    
    iter_count = 0
    iterations = []
    while abs(x3 - x1) > tol and iter_count < max_iter:
        # 记录当前迭代信息
        iterations.append({
            'iter': iter_count + 1,
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'f1': f1,
            'f2': f2,
            'f3': f3,
            'interval_length': x3 - x1
        })
        # 计算二次插值多项式的极小点
        # 使用公式：x_min = [f1(x2²-x3²) + f2(x3²-x1²) + f3(x1²-x2²)] / [2(f1(x2-x3) + f2(x3-x1) + f3(x1-x2))]
        numerator = f1 * (x2**2 - x3**2) + f2 * (x3**2 - x1**2) + f3 * (x1**2 - x2**2)
        denominator = 2 * (f1 * (x2 - x3) + f2 * (x3 - x1) + f3 * (x1 - x2))
        # 避免除零错误
        if abs(denominator) < 1e-10:
            break
        x_min = numerator / denominator
        # 确保x_min在区间[x1, x3]内
        x_min = max(x1, min(x3, x_min))
        # 计算x_min处的函数值
        f_min = f(x_min)
        # 更新三点
        # 根据x_min的位置和f_min的值来决定如何更新三点
        if x_min < x2:
            if f_min < f2:
                # 新的三点：x1, x_min, x2
                x3 = x2
                f3 = f2
                x2 = x_min
                f2 = f_min
            else:
                # 新的三点：x_min, x2, x3
                x1 = x_min
                f1 = f_min
        else:
            if f_min < f2:
                # 新的三点：x2, x_min, x3
                x1 = x2
                f1 = f2
                x2 = x_min
                f2 = f_min
            else:
                # 新的三点：x1, x2, x_min
                x3 = x_min
                f3 = f_min
        
        iter_count += 1
    # 返回中点作为极小值点的近似
    min_point = (x1 + x3) / 2
    return min_point, iter_count, iterations

# 绘制函数图像和迭代过程
def plot_results(f, golden_result, quad_result, a, b):
    """绘制函数图像和两种算法的迭代过程
    Args:
        f: 目标函数
        golden_result: 黄金分割法结果
        quad_result: 二次插值法结果
        a: 初始区间左端点
        b: 初始区间右端点
    """
    # 设置字体大小
    plt.rcParams.update({
        'font.size': 16,  # 全局字体大小
        'axes.titlesize': 20,  # 标题字体大小
        'axes.labelsize': 18,  # 坐标轴标签字体大小
        'xtick.labelsize': 14,  # x轴刻度字体大小
        'ytick.labelsize': 14,  # y轴刻度字体大小
        'legend.fontsize': 14,  # 图例字体大小
    })
    
    # 生成函数图像的x值
    x_values = [a + i * (b - a) / 1000 for i in range(1001)]
    y_values = [f(x) for x in x_values]
    plt.figure(figsize=(10, 6))  # 调整图片大小，减少空白
    # 绘制函数图像
    plt.plot(x_values, y_values, 'b-', label='Objective Function', linewidth=2)
    # 标记黄金分割法的极小值点
    golden_min, golden_iter, _ = golden_result
    plt.plot(golden_min, f(golden_min), 'ro', markersize=10, label=f'Golden Section Minimum: {golden_min:.6f}')
    # 标记二次插值法的极小值点
    quad_min, quad_iter, _ = quad_result
    plt.plot(quad_min, f(quad_min), 'go', markersize=10, label=f'Quadratic Interpolation Minimum: {quad_min:.6f}')
    plt.title('Objective Function and Optimization Results')
    plt.xlabel('α')
    plt.ylabel('φ(α)')
    plt.grid(True, linewidth=1.5)
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, frameon=True)
    plt.tight_layout()  # 减少空白
    # 保存图像到指定文件夹
    plt.savefig('Golden Section Search & 3-Point Quadratic Interpolation Results/optimization_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 重置字体大小设置，避免影响其他绘图
    plt.rcParams.update(plt.rcParamsDefault)

# 探索不同初始区间对算法的影响
def explore_initial_intervals(f, intervals, tol=0.01):
    """探索不同初始区间对两种算法的影响
    Args:
        f: 目标函数
        intervals: 初始区间列表，每个元素为(a, b)
        tol: 精度要求
    Returns:
        dict: 不同初始区间下的算法结果
    """
    results = {}
    for a, b in intervals:
        print(f"\n正在测试初始区间 [{a}, {b}]...")
        # 黄金分割法
        golden_min, golden_iter, golden_history = golden_section_search(f, a, b, tol)
        # 三点二次插值法
        quad_min, quad_iter, quad_history = quadratic_interpolation_search(f, a, b, tol)
        
        results[(a, b)] = {
            'golden': {
                'min': golden_min,
                'iter': golden_iter,
                'history': golden_history,
                'value': f(golden_min)
            },
            'quadratic': {
                'min': quad_min,
                'iter': quad_iter,
                'history': quad_history,
                'value': f(quad_min)
            }
        }
        
        print(f"黄金分割法: 极小值点 = {golden_min:.6f}, 迭代次数 = {golden_iter}")
        print(f"三点二次插值法: 极小值点 = {quad_min:.6f}, 迭代次数 = {quad_iter}")
    
    # 绘制不同初始区间的结果对比图
    plot_initial_interval_comparison(f, results, intervals)
    return results

# 绘制不同初始区间的结果对比图
def plot_initial_interval_comparison(f, results, intervals):
    """绘制不同初始区间下两种算法的结果对比图
    Args:
        f: 目标函数
        results: 不同初始区间下的算法结果
        intervals: 初始区间列表
    """
    # 设置字体大小
    plt.rcParams.update({
        'font.size': 16,  # 全局字体大小
        'axes.titlesize': 20,  # 标题字体大小
        'axes.labelsize': 18,  # 坐标轴标签字体大小
        'xtick.labelsize': 14,  # x轴刻度字体大小
        'ytick.labelsize': 14,  # y轴刻度字体大小
        'legend.fontsize': 14,  # 图例字体大小
    })
    
    plt.figure(figsize=(12, 6))  # 调整图片大小，减少空白
    
    # 生成函数图像的x值
    all_a = [a for a, b in intervals]
    all_b = [b for a, b in intervals]
    x_min = min(all_a)
    x_max = max(all_b)
    x_values = [x_min + i * (x_max - x_min) / 1000 for i in range(1001)]
    y_values = [f(x) for x in x_values]
    plt.plot(x_values, y_values, 'b-', alpha=0.5, label='Objective Function', linewidth=2)
    
    # 标记不同初始区间的结果
    colors = ['r', 'g', 'b', 'm', 'c', 'y']
    # 为不同区间设置不同的点大小，从大到小排列
    sizes = [15, 12, 10, 8, 6, 4]  # 不同区间的点大小
    for i, (a, b) in enumerate(intervals):
        golden_min = results[(a, b)]['golden']['min']
        quad_min = results[(a, b)]['quadratic']['min']
        
        # 绘制初始区间
        plt.axvline(x=a, color=colors[i], linestyle='--', alpha=0.3, linewidth=2)
        plt.axvline(x=b, color=colors[i], linestyle='--', alpha=0.3, linewidth=2)
        
        # 获取当前区间对应的点大小
        size = sizes[i % len(sizes)]
        
        # 标记结果点，使用不同大小区分不同区间
        plt.plot(golden_min, f(golden_min), f'{colors[i]}o', markersize=size + 4, 
                label=f'Interval [{a},{b}] - Golden: {golden_min:.6f}')
        plt.plot(quad_min, f(quad_min), f'{colors[i]}s', markersize=size, 
                label=f'Interval [{a},{b}] - Quadratic: {quad_min:.6f}')
    
    plt.title('Effect of Different Initial Intervals on Optimization Results')
    plt.xlabel('α')
    plt.ylabel('φ(α)')
    plt.grid(True, linewidth=1.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig('Golden Section Search & 3-Point Quadratic Interpolation Results/initial_interval_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 重置字体大小设置
    plt.rcParams.update(plt.rcParamsDefault)
    print("\n已生成初始区间对比图像: Golden Section Search & 3-Point Quadratic Interpolation Results/initial_interval_comparison.png")

# 探索不同精度要求对算法的影响
def explore_precision(f, a, b, precisions):
    """探索不同精度要求对两种算法的影响
    Args:
        f: 目标函数
        a: 初始区间左端点
        b: 初始区间右端点
        precisions: 精度要求列表
    Returns:
        dict: 不同精度要求下的算法结果
    """
    results = {}
    for tol in precisions:
        print(f"\n正在测试精度要求 {tol}...")
        # 黄金分割法
        golden_min, golden_iter, golden_history = golden_section_search(f, a, b, tol)
        # 三点二次插值法
        quad_min, quad_iter, quad_history = quadratic_interpolation_search(f, a, b, tol)
        
        results[tol] = {
            'golden': {
                'min': golden_min,
                'iter': golden_iter,
                'history': golden_history,
                'value': f(golden_min)
            },
            'quadratic': {
                'min': quad_min,
                'iter': quad_iter,
                'history': quad_history,
                'value': f(quad_min)
            }
        }
        
        print(f"黄金分割法: 极小值点 = {golden_min:.6f}, 迭代次数 = {golden_iter}")
        print(f"三点二次插值法: 极小值点 = {quad_min:.6f}, 迭代次数 = {quad_iter}")
    
    # 绘制不同精度要求的结果对比图
    plot_precision_comparison(results, precisions)
    return results

# 绘制不同精度要求的结果对比图
def plot_precision_comparison(results, precisions):
    """绘制不同精度要求下两种算法的结果对比图
    Args:
        results: 不同精度要求下的算法结果
        precisions: 精度要求列表
    """
    # 设置字体大小
    plt.rcParams.update({
        'font.size': 16,  # 全局字体大小
        'axes.titlesize': 20,  # 标题字体大小
        'axes.labelsize': 18,  # 坐标轴标签字体大小
        'xtick.labelsize': 14,  # x轴刻度字体大小
        'ytick.labelsize': 14,  # y轴刻度字体大小
        'legend.fontsize': 14,  # 图例字体大小
    })
    
    plt.figure(figsize=(10, 6))  # 调整图片大小，减少空白
    
    # 绘制迭代次数与精度的关系
    golden_iters = [results[tol]['golden']['iter'] for tol in precisions]
    quad_iters = [results[tol]['quadratic']['iter'] for tol in precisions]
    
    plt.plot(precisions, golden_iters, 'ro-', label='Golden Section Method', linewidth=3, markersize=10)
    plt.plot(precisions, quad_iters, 'go-', label='Quadratic Interpolation Method', linewidth=3, markersize=10)
    
    plt.xscale('log')
    plt.title('Effect of Precision on Iteration Count')
    plt.xlabel('Precision (log scale)')
    plt.ylabel('Iteration Count')
    plt.grid(True, which='both', ls='--', linewidth=1.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig('Golden Section Search & 3-Point Quadratic Interpolation Results/precision_vs_iterations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n已生成精度与迭代次数关系图像: Golden Section Search & 3-Point Quadratic Interpolation Results/precision_vs_iterations.png")
    
    # 绘制精度与结果误差的关系
    theoretical_min = 1 / math.sqrt(2)
    golden_errors = [abs(results[tol]['golden']['min'] - theoretical_min) for tol in precisions]
    quad_errors = [abs(results[tol]['quadratic']['min'] - theoretical_min) for tol in precisions]
    
    plt.figure(figsize=(10, 6))  # 调整图片大小，减少空白
    plt.plot(precisions, golden_errors, 'ro-', label='Golden Section Method', linewidth=3, markersize=10)
    plt.plot(precisions, quad_errors, 'go-', label='Quadratic Interpolation Method', linewidth=3, markersize=10)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Effect of Precision on Result Error')
    plt.xlabel('Precision (log scale)')
    plt.ylabel('Error (log scale)')
    plt.grid(True, which='both', ls='--', linewidth=1.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig('Golden Section Search & 3-Point Quadratic Interpolation Results/precision_vs_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 重置字体大小设置
    plt.rcParams.update(plt.rcParamsDefault)
    print("已生成精度与误差关系图像: Golden Section Search & 3-Point Quadratic Interpolation Results/precision_vs_error.png")

# 绘制迭代过程可视化图
def plot_iteration_process(golden_history, quad_history, improved_history=None):
    """绘制两种算法的迭代过程
    Args:
        golden_history: 黄金分割法迭代历史
        quad_history: 二次插值法迭代历史
        improved_history: 改进黄金分割法迭代历史（可选）
    """
    # 设置字体大小
    plt.rcParams.update({
        'font.size': 16,  # 全局字体大小
        'axes.titlesize': 20,  # 标题字体大小
        'axes.labelsize': 18,  # 坐标轴标签字体大小
        'xtick.labelsize': 14,  # x轴刻度字体大小
        'ytick.labelsize': 14,  # y轴刻度字体大小
        'legend.fontsize': 14,  # 图例字体大小
    })
    
    plt.figure(figsize=(10, 6))  # 调整图片大小，减少空白
    
    # 提取迭代次数和区间长度
    golden_iters = [item['iter'] for item in golden_history]
    golden_lengths = [item['interval_length'] for item in golden_history]
    
    quad_iters = [item['iter'] for item in quad_history]
    quad_lengths = [item['interval_length'] for item in quad_history]
    
    # 绘制黄金分割法和二次插值法的收敛过程
    plt.plot(golden_iters, golden_lengths, 'ro-', label='Golden Section Method', linewidth=3, markersize=10)
    plt.plot(quad_iters, quad_lengths, 'go-', label='Quadratic Interpolation Method', linewidth=3, markersize=10)
    
    # 如果有改进的黄金分割法结果，也绘制出来
    if improved_history:
        improved_iters = [item['iter'] for item in improved_history]
        improved_lengths = [item['interval_length'] for item in improved_history]
        plt.plot(improved_iters, improved_lengths, 'bo--', label='Improved Golden Section Method', linewidth=3, markersize=8)
    
    plt.yscale('log')
    plt.title('Convergence Process of Optimization Algorithms')
    plt.xlabel('Iteration Count')
    plt.ylabel('Interval Length (log scale)')
    plt.grid(True, which='both', ls='--', linewidth=1.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig('Golden Section Search & 3-Point Quadratic Interpolation Results/convergence_process.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n已生成收敛过程图像: Golden Section Search & 3-Point Quadratic Interpolation Results/convergence_process.png")
    
    # 绘制函数值收敛过程
    plt.figure(figsize=(10, 6))  # 调整图片大小，减少空白
    
    # 计算每次迭代的最佳函数值
    golden_best_values = []
    for item in golden_history:
        a, b = item['a'], item['b']
        p, q = item['p'], item['q']
        f_p, f_q = item['f_p'], item['f_q']
        # 取当前区间内已知点的最小函数值
        best_val = min(f_p, f_q)
        golden_best_values.append(best_val)
    
    quad_best_values = []
    for item in quad_history:
        x1, x2, x3 = item['x1'], item['x2'], item['x3']
        f1, f2, f3 = item['f1'], item['f2'], item['f3']
        best_val = min(f1, f2, f3)
        quad_best_values.append(best_val)
    
    plt.plot(golden_iters, golden_best_values, 'ro-', label='Golden Section Method', linewidth=3, markersize=10)
    plt.plot(quad_iters, quad_best_values, 'go-', label='Quadratic Interpolation Method', linewidth=3, markersize=10)
    
    plt.title('Function Value Convergence Process')
    plt.xlabel('Iteration Count')
    plt.ylabel('Best Function Value Found')
    plt.grid(True, linewidth=1.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig('Golden Section Search & 3-Point Quadratic Interpolation Results/function_value_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 重置字体大小设置
    plt.rcParams.update(plt.rcParamsDefault)
    print("已生成函数值收敛过程图像: Golden Section Search & 3-Point Quadratic Interpolation Results/function_value_convergence.png")

# 主函数
def main():
    """主函数，使用两种算法求解目标函数的极小值点并比较结果"""
    print("\n=== 使用0.618法和三点二次插值法求解目标函数极小值点 ===\n")
    print("目标函数: φ(α) = 1 − αe^{-α^2}")
    print("初始区间: [0, 1]")
    print("精度要求: ε = 0.01\n")
    # 设置参数
    a, b = 0.0, 1.0
    tol = 0.01
    
    # --------------------------
    # 1. 基本算法比较（标准黄金分割法和三点二次插值法）
    # --------------------------
    print("1. 正在使用黄金分割法（0.618法）求解...")
    golden_min, golden_iter, golden_history = golden_section_search(
        objective_function, a, b, tol
    )
    golden_min_value = objective_function(golden_min)
    print(f"   黄金分割法完成: 极小值点 = {golden_min:.6f}, 迭代次数 = {golden_iter}")
    
    print("\n2. 正在使用三点二次插值法求解...")
    quad_min, quad_iter, quad_history = quadratic_interpolation_search(
        objective_function, a, b, tol
    )
    quad_min_value = objective_function(quad_min)
    print(f"   三点二次插值法完成: 极小值点 = {quad_min:.6f}, 迭代次数 = {quad_iter}")
    
    # 输出两种算法结果比较
    print(f"\n=== 两种算法结果比较 ===")
    print(f"{'算法':<30} {'极小值点':<15} {'极小值':<15} {'迭代次数':<10}")
    print("-" * 70)
    print(f"{'黄金分割法':<30} {golden_min:<15.6f} {golden_min_value:<15.6f} {golden_iter:<10}")
    print(f"{'三点二次插值法':<30} {quad_min:<15.6f} {quad_min_value:<15.6f} {quad_iter:<10}")
    
    # 验证极小值点的正确性
    theoretical_min = 1 / math.sqrt(2)  # 理论极小值点
    theoretical_min_value = objective_function(theoretical_min)
    print("\n\n=== 验证算法正确性 ===")
    print(f"理论极小值点: α = {theoretical_min:.6f}")
    print(f"理论极小值: φ(α) = {theoretical_min_value:.6f}")
    print(f"黄金分割法结果与理论值的误差: |{golden_min - theoretical_min:.6f}| = {abs(golden_min - theoretical_min):.6f}")
    print(f"三点二次插值法结果与理论值的误差: |{quad_min - theoretical_min:.6f}| = {abs(quad_min - theoretical_min):.6f}")
    
    # --------------------------
    # 2. 探索不同初始区间的影响
    # --------------------------
    print("\n\n=== 探索不同初始区间的影响 ===")
    intervals = [(0.0, 1.0), (0.0, 5.0), (0.5, 2), (0.7, 0.8)]
    print("3. 正在探索不同初始区间...")
    try:
        explore_initial_intervals(objective_function, intervals, tol)
        print("   初始区间探索完成")
    except Exception as e:
        print(f"   初始区间探索时出错: {e}")
    
    # --------------------------
    # 3. 探索不同精度要求的影响
    # --------------------------
    print("\n\n=== 探索不同精度要求的影响 ===")
    precisions = [0.1, 0.05, 0.01, 0.005, 0.001]
    print("4. 正在探索不同精度要求...")
    try:
        explore_precision(objective_function, a, b, precisions)
        print("   精度要求探索完成")
    except Exception as e:
        print(f"   精度要求探索时出错: {e}")
    
    # --------------------------
    # 4. 改进的黄金分割法
    # --------------------------
    print("\n\n=== 改进的黄金分割法 ===")
    print("5. 正在使用改进的黄金分割法求解...")
    improved_min, improved_iter, improved_history = improved_golden_section_search(
        objective_function, a, b, tol
    )
    improved_min_value = objective_function(improved_min)
    print(f"   改进的黄金分割法完成: 极小值点 = {improved_min:.6f}, 迭代次数 = {improved_iter}")
    
    # 输出改进前后的结果比较
    print(f"\n=== 黄金分割法改进前后比较 ===")
    print(f"{'算法':<30} {'极小值点':<15} {'极小值':<15} {'迭代次数':<10} {'与理论值误差':<15}")
    print("-" * 95)
    print(f"{'标准黄金分割法':<30} {golden_min:<15.6f} {golden_min_value:<15.6f} {golden_iter:<10} {abs(golden_min - theoretical_min):<15.6f}")
    print(f"{'改进的黄金分割法':<30} {improved_min:<15.6f} {improved_min_value:<15.6f} {improved_iter:<10} {abs(improved_min - theoretical_min):<15.6f}")
    
    # 改进的黄金分割法与理论值的导数验证
    improved_derivative = objective_function_derivative(improved_min)
    print(f"\n导数验证:")
    print(f"理论极小值点导数: {objective_function_derivative(theoretical_min):.6f} (应为0)")
    print(f"改进的黄金分割法找到点的导数: {improved_derivative:.6f}")
    
    # --------------------------
    # 5. 绘制迭代过程可视化
    # --------------------------
    print("\n6. 正在绘制迭代过程可视化图...")
    try:
        plot_iteration_process(golden_history, quad_history, improved_history)
        print("   迭代过程可视化图绘制完成")
    except Exception as e:
        print(f"   绘制迭代过程可视化图时出错: {e}")
    
    # --------------------------
    # 6. 绘制基本结果图像
    # --------------------------
    print("\n7. 正在绘制基本结果图像...")
    try:
        plot_results(objective_function, 
                    (golden_min, golden_iter, golden_history), 
                    (quad_min, quad_iter, quad_history), 
                    a, b)
        print("   基本结果图像绘制完成")
    except Exception as e:
        print(f"   绘制基本结果图像时出错: {e}")
    
    print("\n=== 所有探索和分析已完成 ===")
    print(f"\n所有图像已保存到文件夹: Golden Section Search & 3-Point Quadratic Interpolation Results")

if __name__ == "__main__":
    main()