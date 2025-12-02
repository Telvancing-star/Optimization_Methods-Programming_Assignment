import numpy as np
import time
import matplotlib.pyplot as plt

# 设置中文支持和字体放大
plt.rcParams.update({
    # 中文支持，使用多种备选字体
    'font.family': ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    # 字体放大一倍
    'font.size': 20,  # 全局字体大小
    'axes.titlesize': 28,  # 标题字体大小
    'axes.labelsize': 24,  # 坐标轴标签字体大小
    'xtick.labelsize': 20,  # x轴刻度字体大小
    'ytick.labelsize': 20,  # y轴刻度字体大小
    'legend.fontsize': 20,  # 图例字体大小
    'figure.titlesize': 32  # 图表标题字体大小
})

# 全局计数器，用于统计函数调用次数
function_calls = 0
gradient_calls = 0
hessian_calls = 0

# 残差函数 r_i(x)
def residual_function(x, i, m, n):
    if i == m - 1:
        return x[0]
    elif i == m:
        if len(x) >= 2:
            return x[1] - x[0]**2 - 1
        else:
            return x[0]**2 + 1  # 退化情况处理
    else:
        t_i = i / (m - 2)
        sum1 = 0.0
        sum2 = 0.0
        for j in range(2, min(n+1, len(x)+1)):
            sum1 += (j - 1) * x[j-1] * (t_i ** (j - 2))
        for j in range(1, min(n+1, len(x)+1)):
            sum2 += x[j-1] * (t_i ** (j - 1))
        return sum1 - sum2**2 - 1

# 目标函数 f(x) = sum_{i=1}^m r_i(x)^2
def objective_function_least_squares(x, m, n):
    global function_calls
    function_calls += 1
    total = 0.0
    for i in range(1, m + 1):
        r = residual_function(x, i, m, n)
        total += r ** 2
    return total

# 目标函数的梯度 g(x)
def gradient_function_least_squares(x, m, n):
    global gradient_calls
    gradient_calls += 1
    n_dim = len(x)
    grad = np.zeros(n_dim)
    for i in range(1, m + 1):
        r_i = residual_function(x, i, m, n)
        r_grad = np.zeros(n_dim)
        if i == m - 1:
            if n_dim > 0:
                r_grad[0] = 1.0
        elif i == m:
            if n_dim > 0:
                r_grad[0] = -2 * x[0]
            if n_dim > 1:
                r_grad[1] = 1.0
        else:
            t_i = i / (m - 2)
            sum2 = 0.0
            for j in range(1, min(n+1, n_dim+1)):
                sum2 += x[j-1] * (t_i ** (j - 1))
            # 计算每个x_j的梯度分量
            for j in range(1, min(n+1, n_dim+1)):
                if j == 1:
                    r_grad[j-1] = -2 * (t_i ** 0) * sum2
                else:
                    term1 = (j-1) * (t_i ** (j-2))
                    term2 = -2 * (t_i ** (j-1)) * sum2
                    r_grad[j-1] = term1 + term2
        # 累加到总梯度（利用链式法则：∇f = 2 * sum(r_i * ∇r_i)）
        grad += 2 * r_i * r_grad
    return grad

# 精确线搜索 - 黄金分割法
def golden_section_search(f, df, x, d, m, n, eps=1e-6, max_iter=100):
    a = 0.0
    b = 1.0
    gr = (np.sqrt(5) + 1) / 2
    while True:
        f_a = f(x + a * d, m, n)
        f_b = f(x + b * d, m, n)
        if f_b < f_a:
            a, b = b, 2 * b
        else:
            break
    # 黄金分割搜索
    for _ in range(max_iter):
        if b - a < eps:
            break
        c = b - (b - a) / gr
        d_point = a + (b - a) / gr
        f_c = f(x + c * d, m, n)
        f_d = f(x + d_point * d, m, n)
        if f_c < f_d:
            b = d_point
        else:
            a = c
    return (a + b) / 2

# Armijo条件线搜索
def armijo_line_search(f, df, x, d, m, n, beta=0.5, sigma=0.01, max_iter=100):
    alpha = 1.0
    f_x = f(x, m, n)
    grad_x = df(x, m, n)
    # 计算梯度在搜索方向上的投影
    grad_proj = np.dot(grad_x, d)
    if grad_proj >= 0:
        raise ValueError("搜索方向不是下降方向")
    # Armijo条件: f(x + alpha*d) <= f(x) + sigma*alpha*grad_x^T*d
    for _ in range(max_iter):
        f_x_alpha = f(x + alpha * d, m, n)
        if f_x_alpha <= f_x + sigma * alpha * grad_proj:
            return alpha
        alpha *= beta
    return alpha

# Wolfe条件线搜索
def wolfe_line_search(f, df, x, d, m, n, beta=0.5, sigma=0.01, rho=0.9, max_iter=100):
    alpha = 1.0  # 初始步长
    f_x = f(x, m, n)
    grad_x = df(x, m, n)
    # 计算梯度在搜索方向上的投影
    grad_proj = np.dot(grad_x, d)
    if grad_proj >= 0:
        raise ValueError("搜索方向不是下降方向")
    for _ in range(max_iter):
        # Armijo条件: f(x + alpha*d) <= f(x) + sigma*alpha*grad_x^T*d
        f_x_alpha = f(x + alpha * d, m, n)
        if f_x_alpha > f_x + sigma * alpha * grad_proj:
            alpha *= beta
            continue
        # 曲率条件: grad(x + alpha*d)^T*d >= rho*grad_x^T*d
        grad_x_alpha = df(x + alpha * d, m, n)
        grad_proj_alpha = np.dot(grad_x_alpha, d)
        if grad_proj_alpha >= rho * grad_proj:
            return alpha
        alpha *= beta
    return alpha

# SR1（对称秩1）算法
def sr1_method(f, df, x0, m, n, line_search='wolfe', eps=1e-6, max_iter=100):
    line_search_functions = {
        'exact': golden_section_search,
        'armijo': armijo_line_search,
        'wolfe': wolfe_line_search
    }
    line_search_func = line_search_functions.get(line_search, wolfe_line_search)
    # 初始化
    x = np.copy(x0)
    n_dim = len(x)
    H = np.eye(n_dim)
    g = df(x, m, n)
    g_norm = np.linalg.norm(g)
    iter_count = 0
    
    # 保存迭代历史数据
    history = {
        'function_values': [f(x, m, n)],
        'gradient_norms': [g_norm],
        'iterates': [x.copy()]
    }
    
    # 迭代主循环
    while g_norm > eps and iter_count < max_iter:
        # 计算搜索方向
        d = -np.dot(H, g)
        if np.dot(g, d) >= 0:
            H = np.eye(n_dim)
            d = -g
        # 执行线搜索
        alpha = line_search_func(f, df, x, d, m, n)
        x_new = x + alpha * d
        g_new = df(x_new, m, n)
        s = x_new - x
        y = g_new - g    
        
        # 保存当前迭代数据
        history['function_values'].append(f(x_new, m, n))
        history['gradient_norms'].append(np.linalg.norm(g_new))
        history['iterates'].append(x_new.copy())
        
        # SR1更新公式
        Hy = np.dot(H, y)
        s_minus_Hy = s - Hy
        denominator = np.dot(s_minus_Hy, y)
        if abs(denominator) > 1e-10 * np.linalg.norm(s_minus_Hy) * np.linalg.norm(y):
            numerator = np.outer(s_minus_Hy, s_minus_Hy)
            H = H + numerator / denominator
        x = x_new
        g = g_new
        g_norm = np.linalg.norm(g)
        iter_count += 1
    
    return x, iter_count, f(x, m, n), g_norm, history

# DFP（Davidon-Fletcher-Powell）算法
def dfp_method(f, df, x0, m, n, line_search='wolfe', eps=1e-6, max_iter=100):
    line_search_functions = {
        'exact': golden_section_search,
        'armijo': armijo_line_search,
        'wolfe': wolfe_line_search
    }
    line_search_func = line_search_functions.get(line_search, wolfe_line_search)
    # 初始化
    x = np.copy(x0)
    n_dim = len(x)
    H = np.eye(n_dim)
    g = df(x, m, n)
    g_norm = np.linalg.norm(g)
    iter_count = 0
    
    # 保存迭代历史数据
    history = {
        'function_values': [f(x, m, n)],
        'gradient_norms': [g_norm],
        'iterates': [x.copy()]
    }
    
    # 迭代主循环
    while g_norm > eps and iter_count < max_iter:
        # 计算搜索方向 d
        d = -np.dot(H, g)
        if np.dot(g, d) >= 0:
            H = np.eye(n_dim)
            d = -g
        # 执行线搜索
        alpha = line_search_func(f, df, x, d, m, n)
        x_new = x + alpha * d
        g_new = df(x_new, m, n)
        s = x_new - x
        y = g_new - g
        
        # 保存当前迭代数据
        history['function_values'].append(f(x_new, m, n))
        history['gradient_norms'].append(np.linalg.norm(g_new))
        history['iterates'].append(x_new.copy())
        
        sTy = np.dot(s, y)
        if sTy <= 1e-10:
            x = x_new
            g = g_new
            g_norm = np.linalg.norm(g)
            iter_count += 1
            continue
        # DFP更新公式
        Hy = np.dot(H, y)
        yTHy = np.dot(y, Hy)
        term1 = np.outer(s, s) / sTy
        term2 = np.outer(Hy, Hy) / yTHy
        H = H + term1 - term2
        x = x_new
        g = g_new
        g_norm = np.linalg.norm(g)
        iter_count += 1
    
    return x, iter_count, f(x, m, n), g_norm, history

# BFGS（Broyden-Fletcher-Goldfarb-Shanno）算法
def bfgs_method(f, df, x0, m, n, line_search='wolfe', eps=1e-6, max_iter=100):
    line_search_functions = {
        'exact': golden_section_search,
        'armijo': armijo_line_search,
        'wolfe': wolfe_line_search
    }
    line_search_func = line_search_functions.get(line_search, wolfe_line_search)
    # 初始化
    x = np.copy(x0)
    n_dim = len(x)
    H = np.eye(n_dim)
    g = df(x, m, n)
    g_norm = np.linalg.norm(g)
    iter_count = 0
    
    # 保存迭代历史数据
    history = {
        'function_values': [f(x, m, n)],
        'gradient_norms': [g_norm],
        'iterates': [x.copy()]
    }
    
    # 迭代主循环
    while g_norm > eps and iter_count < max_iter:
        # 计算搜索方向
        d = -np.dot(H, g)
        if np.dot(g, d) >= 0:
            H = np.eye(n_dim)
            d = -g
        # 执行线搜索
        alpha = line_search_func(f, df, x, d, m, n)
        x_new = x + alpha * d
        g_new = df(x_new, m, n)
        s = x_new - x
        y = g_new - g
        
        # 保存当前迭代数据
        history['function_values'].append(f(x_new, m, n))
        history['gradient_norms'].append(np.linalg.norm(g_new))
        history['iterates'].append(x_new.copy())
        
        sTy = np.dot(s, y)
        if sTy <= 1e-10:
            x = x_new
            g = g_new
            g_norm = np.linalg.norm(g)
            iter_count += 1
            continue
        # BFGS更新公式：
        yTHy = np.dot(np.dot(y, H), y)
        rho = 1.0 / sTy
        term1 = (1.0 + yTHy * rho) * np.outer(s, s) * rho
        term2 = np.dot(H, np.outer(y, s)) * rho + np.dot(np.outer(s, y), H) * rho
        H = H + term1 - term2
        x = x_new
        g = g_new
        g_norm = np.linalg.norm(g)
        iter_count += 1
    
    return x, iter_count, f(x, m, n), g_norm, history

# 运行算法并收集性能统计
def run_algorithm(algorithm, f, df, x0, m, n, line_search='wolfe', eps=1e-6, max_iter=100):
    reset_counters()
    start_time = time.time()
    x, iterations, f_value, g_norm, history = algorithm(f, df, x0, m, n, line_search, eps, max_iter)
    end_time = time.time()
    stats = {
        'x': x,
        'iterations': iterations,
        'function_value': f_value,
        'gradient_norm': g_norm,
        'function_calls': function_calls,
        'gradient_calls': gradient_calls,
        'cpu_time': end_time - start_time,
        'converged': g_norm <= eps,
        'history': history  # 添加历史数据
    }
    return stats

# 比较不同算法和线搜索策略的性能
def compare_algorithms(x0, m, n, line_searches=['wolfe', 'armijo', 'exact'], eps=1e-6, max_iter=100):
    algorithms = {
        'SR1': sr1_method,
        'DFP': dfp_method,
        'BFGS': bfgs_method
    }
    results = {}
    for algo_name, algorithm in algorithms.items():
        results[algo_name] = {}
        for ls in line_searches:
            print(f"正在运行 {algo_name} 算法，使用 {ls} 线搜索策略...")
            stats = run_algorithm(
                algorithm, 
                objective_function_least_squares, 
                gradient_function_least_squares, 
                x0, m, n, ls, eps, max_iter
            )
            results[algo_name][ls] = stats
            print(f"  {algo_name} + {ls}:")
            print(f"    迭代次数: {stats['iterations']}")
            print(f"    函数调用次数: {stats['function_calls']}")
            print(f"    梯度调用次数: {stats['gradient_calls']}")
            print(f"    CPU时间: {stats['cpu_time']:.6f} 秒")
            print(f"    函数值: {stats['function_value']:.8e}")
            print(f"    梯度范数: {stats['gradient_norm']:.8e}")
            print(f"    收敛状态: {'是' if stats['converged'] else '否'}")
            print()
    return results

# 格式化输出性能比较结果
def print_comparison_results(results):
    print("=== 拟牛顿方法性能比较结果 ===")
    print()
    metrics = [
        ('迭代次数', 'iterations'),
        ('函数调用次数', 'function_calls'),
        ('梯度调用次数', 'gradient_calls'),
        ('CPU时间 (秒)', 'cpu_time'),
        ('函数值', 'function_value'),
        ('梯度范数', 'gradient_norm')
    ]
    for metric_name, metric_key in metrics:
        print(f"--- {metric_name} 比较 ---")
        algorithms = list(results.keys())
        line_searches = list(results[algorithms[0]].keys()) if algorithms else []
        header = f"{'算法':<10}"
        for ls in line_searches:
            header += f"{ls:<20}"
        print(header)
        print("-" * (10 + 20 * len(line_searches)))
        for algo in algorithms:
            row = f"{algo:<10}"
            for ls in line_searches:
                value = results[algo][ls][metric_key]
                if isinstance(value, float):
                    if metric_key in ['cpu_time']:
                        row += f"{value:<20.6f}"
                    else:
                        row += f"{value:<20.8e}"
                else:
                    row += f"{value:<20}"
            print(row)
        print()

# 重置计数器
def reset_counters():
    global function_calls, gradient_calls, hessian_calls
    function_calls = 0
    gradient_calls = 0
    hessian_calls = 0

# 可视化函数

def create_visualization_dir():
    """创建可视化图表保存目录"""
    import os
    os.makedirs('Quasi-Newton Methods', exist_ok=True)
    # 设置中文支持和字体大小
    plt.rcParams.update({
        # 中文支持
        'font.family': ['SimHei', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        # 字体放大一倍
        'font.size': 20,  # 全局字体大小
        'axes.titlesize': 28,  # 标题字体大小
        'axes.labelsize': 24,  # 坐标轴标签字体大小
        'xtick.labelsize': 20,  # x轴刻度字体大小
        'ytick.labelsize': 20,  # y轴刻度字体大小
        'legend.fontsize': 20,  # 图例字体大小
        'figure.titlesize': 32  # 图表标题字体大小
    })

# 绘制迭代次数对比图
def plot_iterations_comparison(results, test_case_name):
    """绘制不同算法在不同线搜索策略下的迭代次数对比图"""
    plt.figure(figsize=(12, 8))
    algorithms = list(results.keys())
    line_searches = list(results[algorithms[0]].keys())
    
    width = 0.25  # 柱状图宽度
    x = np.arange(len(line_searches))
    
    for i, algo in enumerate(algorithms):
        iterations = [results[algo][ls]['iterations'] for ls in line_searches]
        plt.bar(x + i*width, iterations, width, label=algo)
    
    plt.xlabel('线搜索策略')
    plt.ylabel('迭代次数')
    plt.title(f'{test_case_name} - 迭代次数对比')
    plt.xticks(x + width, line_searches)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Quasi-Newton Methods/{test_case_name}_iterations_comparison.png', dpi=300)
    plt.close()

# 绘制CPU时间对比图
def plot_cpu_time_comparison(results, test_case_name):
    """绘制不同算法的CPU时间对比图"""
    plt.figure(figsize=(12, 8))
    algorithms = list(results.keys())
    line_searches = list(results[algorithms[0]].keys())
    
    width = 0.25  # 柱状图宽度
    x = np.arange(len(line_searches))
    
    for i, algo in enumerate(algorithms):
        cpu_times = [results[algo][ls]['cpu_time'] for ls in line_searches]
        plt.bar(x + i*width, cpu_times, width, label=algo)
    
    plt.xlabel('线搜索策略')
    plt.ylabel('CPU时间 (秒)')
    plt.title(f'{test_case_name} - CPU时间对比')
    plt.xticks(x + width, line_searches)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Quasi-Newton Methods/{test_case_name}_cpu_time_comparison.png', dpi=300)
    plt.close()

# 绘制函数值收敛图
def plot_function_value_convergence(results, test_case_name, line_search):
    """绘制不同算法的函数值收敛图"""
    plt.figure(figsize=(12, 8))
    algorithms = list(results.keys())
    
    for algo in algorithms:
        function_values = results[algo][line_search]['history']['function_values']
        plt.semilogy(range(len(function_values)), function_values, label=algo)
    
    plt.xlabel('迭代次数')
    plt.ylabel('函数值 (对数刻度)')
    plt.title(f'{test_case_name} - 函数值收敛 ({line_search}线搜索)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'Quasi-Newton Methods/{test_case_name}_function_value_convergence_{line_search}.png', dpi=300)
    plt.close()

# 绘制梯度范数收敛图
def plot_gradient_norm_convergence(results, test_case_name, line_search):
    """绘制不同算法的梯度范数收敛图"""
    plt.figure(figsize=(12, 8))
    algorithms = list(results.keys())
    
    for algo in algorithms:
        gradient_norms = results[algo][line_search]['history']['gradient_norms']
        plt.semilogy(range(len(gradient_norms)), gradient_norms, label=algo)
    
    plt.xlabel('迭代次数')
    plt.ylabel('梯度范数 (对数刻度)')
    plt.title(f'{test_case_name} - 梯度范数收敛 ({line_search}线搜索)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'Quasi-Newton Methods/{test_case_name}_gradient_norm_convergence_{line_search}.png', dpi=300)
    plt.close()

# 绘制性能雷达图
def plot_performance_radar(results, test_case_name):
    """绘制算法性能雷达图"""
    algorithms = list(results.keys())
    line_searches = list(results[algorithms[0]].keys())
    
    # 选择wolfe线搜索的结果进行雷达图比较
    ls = 'wolfe'
    
    # 性能指标
    metrics = [
        'iterations',
        'function_calls',
        'gradient_calls',
        'cpu_time'
    ]
    
    # 归一化指标值
    normalized_data = {}
    max_values = {}
    
    # 计算每个指标的最大值
    for metric in metrics:
        max_values[metric] = max(results[algo][ls][metric] for algo in algorithms)
    
    # 归一化数据
    for algo in algorithms:
        normalized_data[algo] = [results[algo][ls][metric]/max_values[metric] for metric in metrics]
    
    # 绘制雷达图
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    for algo in algorithms:
        values = normalized_data[algo]
        values += values[:1]  # 闭合雷达图
        ax.plot(angles, values, linewidth=2, label=algo)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric for metric in metrics])
    ax.set_title(f'{test_case_name} - 性能雷达图 ({ls}线搜索)', size=20, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(f'Quasi-Newton Methods/{test_case_name}_performance_radar.png', dpi=300)
    plt.close()

# 绘制所有线搜索策略下的收敛对比图
def plot_all_convergence(results, test_case_name):
    """绘制所有线搜索策略下的收敛对比图"""
    algorithms = list(results.keys())
    line_searches = list(results[algorithms[0]].keys())
    
    # 函数值收敛对比
    plt.figure(figsize=(12, 8))
    for algo in algorithms:
        for ls in line_searches:
            function_values = results[algo][ls]['history']['function_values']
            plt.semilogy(range(len(function_values)), function_values, label=f'{algo} ({ls})')
    
    plt.xlabel('迭代次数')
    plt.ylabel('函数值 (对数刻度)')
    plt.title(f'{test_case_name} - 所有算法和线搜索策略的函数值收敛对比')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'Quasi-Newton Methods/{test_case_name}_all_function_value_convergence.png', dpi=300)
    plt.close()

# 生成所有可视化图表
def generate_visualizations(results, test_case_name):
    """生成所有可视化图表"""
    create_visualization_dir()
    
    # 生成比较图表
    plot_iterations_comparison(results, test_case_name)
    plot_cpu_time_comparison(results, test_case_name)
    plot_performance_radar(results, test_case_name)
    plot_all_convergence(results, test_case_name)
    
    # 生成收敛过程图表
    line_searches = list(results[list(results.keys())[0]].keys())
    for ls in line_searches:
        plot_function_value_convergence(results, test_case_name, ls)
        plot_gradient_norm_convergence(results, test_case_name, ls)

def main():
    print("拟牛顿方法 SR1、DFP、BFGS 的性能比较")
    print("=" * 50)
    print()
    # 设置不同的问题规模进行测试
    test_cases = [
        {'m': 10, 'n': 5, 'name': '小规模问题 (m=10, n=5)'},
        {'m': 20, 'n': 18, 'name': '中等规模问题 (m=20, n=18)'},
        # {'m': 50, 'n': 20, 'name': '大规模问题 (m=50, n=20)'}
    ]
    eps = 1e-6
    max_iter = 200
    # 要比较的线搜索方法
    line_searches = ['wolfe', 'armijo', 'exact']
    for test_case in test_cases:
        m = test_case['m']
        n = test_case['n']
        print(f"\n测试 {test_case['name']}")
        print("-" * 40)
        # 生成初始点 x0
        np.random.seed(42)
        x0 = np.random.randn(n) * 0.5
        print(f"初始点 x0: {x0}")
        print()
        results = compare_algorithms(
            x0, 
            m, 
            n, 
            line_searches=line_searches,
            eps=eps,
            max_iter=max_iter
        )
        print_comparison_results(results)
        
        # 生成可视化图表
        test_case_name = test_case['name'].replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
        generate_visualizations(results, test_case_name)
        print(f"\n已生成可视化图表，保存至 Quasi-Newton Methods 目录")
        
        # 分析每种线搜索策略下三个方法的表现
        print("\n--- 各线搜索策略下的最佳算法分析 ---")
        for ls in line_searches:
            print(f"线搜索策略: {ls}")
            # 比较三种算法在该线搜索策略下的表现，综合考虑迭代次数和函数值
            best_algo = min(
                results.keys(),
                key=lambda algo: (results[algo][ls]['iterations'], results[algo][ls]['function_value'])
            )
            print(f"  最佳算法: {best_algo}")
            print(f"    迭代次数: {results[best_algo][ls]['iterations']}")
            print(f"    函数值: {results[best_algo][ls]['function_value']:.8e}")
            print(f"    梯度范数: {results[best_algo][ls]['gradient_norm']:.8e}")
            print(f"    CPU时间: {results[best_algo][ls]['cpu_time']:.6f} 秒")
        
        # 整体分析三个方法哪个最好
        print("\n--- 整体最佳算法分析 ---")
        # 综合考虑所有线搜索策略下的表现，计算每个算法的综合评分
        algo_scores = {}
        for algo_name in results.keys():
            # 计算该算法在所有线搜索策略下的平均迭代次数、平均函数值、平均CPU时间
            total_iterations = sum(results[algo_name][ls]['iterations'] for ls in line_searches)
            avg_iterations = total_iterations / len(line_searches)
            
            total_function_value = sum(results[algo_name][ls]['function_value'] for ls in line_searches)
            avg_function_value = total_function_value / len(line_searches)
            
            total_cpu_time = sum(results[algo_name][ls]['cpu_time'] for ls in line_searches)
            avg_cpu_time = total_cpu_time / len(line_searches)
            
            # 计算综合评分（考虑迭代次数、函数值、CPU时间）
            # 权重可调整，这里假设三个指标同等重要
            score = avg_iterations + avg_function_value * 1e6 + avg_cpu_time * 1e3
            algo_scores[algo_name] = {
                'avg_iterations': avg_iterations,
                'avg_function_value': avg_function_value,
                'avg_cpu_time': avg_cpu_time,
                'score': score
            }
        
        # 找出整体最佳算法
        overall_best_algo = min(algo_scores.keys(), key=lambda algo: algo_scores[algo]['score'])
        print(f"整体最佳算法: {overall_best_algo}")
        print(f"  平均迭代次数: {algo_scores[overall_best_algo]['avg_iterations']:.2f}")
        print(f"  平均函数值: {algo_scores[overall_best_algo]['avg_function_value']:.8e}")
        print(f"  平均CPU时间: {algo_scores[overall_best_algo]['avg_cpu_time']:.6f} 秒")
        
        print("\n" + "=" * 50)
    print("\n测试完成！")

if __name__ == "__main__":
    main()