import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt  
import matplotlib.patches as patches
import dccp  


def find_max_radius_in_sector(center:np.ndarray, radius:float, start_angle:float, end_angle:float, n_circle:int=20, seed:int=134, warm_start=True):
    """
    在不大于180度的扇形内找到最大数量的不重叠圆的最大半径
    Args:
        center (np.ndarray): 圆心
        radius (float): 扇形半径
        start_angle (float): 起始角度，弧度制
        end_angle (float): 终止角度，弧度制
        n_circle (int, optional): 放置圆的数量. Defaults to 20.
        seed (int, optional): 随机种子，用于随机化算法. Defaults to 134.
    Returns:
        半径：最大半径
        所有圆的圆心坐标
    """
    np.random.seed(seed)
    n = n_circle
    r = cp.Variable()
    c = cp.Variable((n, 2))
    # theta_i = np.linspace(start_angle,end_angle,n)
    constr = []
    for i in range(n - 1):
        constr.append(cp.norm(cp.reshape(c[i, :], (1, 2)) - c[i + 1: n, :], 2, axis=1) >= 2 * r)  

    for i in range(n):
        constr.append(cp.norm(c[i, :]-center)<=radius-r-(np.random.rand(1)*radius/10 if warm_start else 0))
        constr.append((c[:,0]-center[0])*np.sin(start_angle)-(c[:,1]-center[1])*np.cos(start_angle)<=-r)
        constr.append((c[:,0]-center[0])*np.sin(end_angle)-(c[:,1]-center[1])*np.cos(end_angle)>=r)
    prob = cp.Problem(cp.Maximize(r), constr)
    prob.solve(method="dccp", solver="ECOS", ep=1e-6, max_slack=1e-2)
    
    if warm_start:
        for i in range(n):
            constr[n-1+3*i] = cp.norm(c[i, :]-center)<=radius-r
        prob = cp.Problem(cp.Maximize(r), constr)
        prob.solve(method="dccp", solver="ECOS", ep=1e-6, max_slack=1e-2, warm_start=True)
    return r.value, [(c[i, 0].value, c[i,1].value) for i in range(n)]


# 扇形描述（不大于180度）
center = np.array([0, 0])
radius = 10
start_angle = np.pi / 6
end_angle = 2*np.pi / 3

# 圆的个数
n_circle = 30

# plot
fig, ax = plt.subplots(2, 1, figsize=(8, 8))
fig.set_tight_layout(True)
def plot_sector(axi, circle_centers, max_r):
    for i in range(n_circle):
        circ = plt.Circle((circle_centers[i][0], circle_centers[i][1]), max_r, color='b', alpha=0.3)
        axi.add_artist(circ)

    wedge = patches.Wedge(center, radius, start_angle * 180 / np.pi, end_angle * 180 / np.pi, fill=False, ec='r')
    axi.add_patch(wedge)
    axi.set_xlim([-radius-1, radius + 1])  
    axi.set_ylim([0, radius + 1])

# 没有暖启动，容易陷入局部最优
max_r, circle_centers = find_max_radius_in_sector(warm_start=False,center=center, radius=radius, start_angle=start_angle, end_angle=end_angle, n_circle=n_circle, seed=153)
print("最大圆的半径为：", max_r)
print("所有圆的圆心坐标为：", circle_centers)
plot_sector(ax[0], circle_centers, max_r)
ax[0].set_title('cold start')

# 暖启动，可以缓解局部最优
max_r, circle_centers = find_max_radius_in_sector(warm_start=True, center=center, radius=radius, start_angle=start_angle, end_angle=end_angle, n_circle=n_circle, seed=153)
print("最大圆的半径为：", max_r)
print("所有圆的圆心坐标为：", circle_centers)
plot_sector(ax[1], circle_centers, max_r)
ax[1].set_title('warm start')
plt.show()