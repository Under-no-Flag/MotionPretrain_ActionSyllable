import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 范德波尔方程参数
mu = 1.0

def vanderpol(t, y):
    y1, y2 = y
    return [y2, -y1 + mu * (1 - y1**2) * y2]

# 时间设置
t_max = 100
t_eval = np.linspace(0, t_max, 5000)

# 初始条件（不同的起始点）
initial_conditions = [
    (0.5, 0.5),
    (2.0, 2.0),
    (3.0, 0.0),
    (0.1, 0.1)
]

# 创建图形
plt.figure(figsize=(10, 8))

# 生成网格以绘制向量场
x = np.linspace(-3, 3, 10)
y = np.linspace(-5, 5, 10)
X, Y = np.meshgrid(x, y)

# 计算向量场的导数
dX = Y
dY = -X + mu * (1 - X**2) * Y

# 绘制向量场（灰色箭头）
plt.streamplot(X, Y, dX, dY, color='black', density=0.5, linewidth=1.5, arrowsize=2)

# 绘制每个初始条件的轨迹
for y0 in initial_conditions:
    sol = solve_ivp(vanderpol, (0, t_max), y0, t_eval=t_eval, method='RK45')
    # 提取后1/4时间的数据（稳态部分）
    start_idx = int(0.75 * len(sol.t))
    plt.plot(sol.y[0][start_idx:], sol.y[1][start_idx:], linewidth=2.5)
    # 标记初始点
    plt.plot(y0[0], y0[1], 'ko', markersize=3)

# 图形装饰
# plt.title(f'Van der Pol Oscillator Limit Cycle (μ={mu})')
plt.xlabel(r"v",fontsize=26)
plt.ylabel(r"$ \rho $",fontsize=26)
plt.xlim([-3, 3])
plt.ylim([-5, 5])
# plt.grid(True)
plt.show()


