import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt  
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import dccp  

def find_max_length_in_sector(center:np.ndarray, radius:float, start_angle: float, end_angle: float, ratio:float, n_rec:int=20, seed:int=134):
    """
    在不大于180度的扇形内找到不重叠矩形的最大长度
    Args:
        center (np.ndarray): 扇形对应圆心
        radius (float): 扇形半径
        start_angle (float): 起始角度，弧度制
        end_angle (float): 终止角度，弧度制
        ratio (float): 给定长宽比
        n_rec (int, optional): 放置矩形的数量.
        seed (int, optional): 随机种子，用于随机化算法.
    Returns:
        长：最大矩形长度
        所有矩形的中心坐标
    """
    np.random.seed(seed)
    n = n_rec
    l = cp.Variable()
    c = cp.Variable((n, 2))
    
    def get_4_vertex(rec_center):
        return (rec_center[0]+l/2, rec_center[1]+l/2/ratio),\
            (rec_center[0]+l/2, rec_center[1]-l/2/ratio),\
            (rec_center[0]-l/2, rec_center[1]+l/2/ratio),\
            (rec_center[0]-l/2, rec_center[1]-l/2/ratio)
            
    constr = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            constr.append(cp.maximum(cp.abs(c[i,0]-c[j,0]), cp.abs(c[i,1]-c[j,1])*ratio)>=l)
            # constr.append(cp.maximum(c[i,0]-c[j,0])**2+ (c[i,1]-c[j,1])**2>=l**2)
    for i in range(n):
        ps = get_4_vertex(rec_center=c[i,:])
        for p in ps:
            constr.append((p[0]-center[0])**2+(p[1]-center[1])**2<=radius**2)
            constr.append((p[0]-center[0])*np.sin(start_angle)-(p[1]-center[1])*np.cos(start_angle)<=0)
            constr.append((p[0]-center[0])*np.sin(end_angle)-(p[1]-center[1])*np.cos(end_angle)>=0)
    prob = cp.Problem(cp.Maximize(l), constr)
    prob.solve(method="dccp", solver="ECOS", ep=1e-6, max_slack=1e-2)
    return l.value, c.value
    

# 扇形描述（不大于180度）
center = np.array([0, 0])
radius = 10
start_angle = np.pi / 6
end_angle = 2*np.pi / 3
ratio = 1.5

# 矩形的个数
n_rec = 10

# plot
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
fig.set_tight_layout(True)
def plot_sector(axi, rec_centers):
    for i in range(n_rec):
        rec = Rectangle((rec_centers[i][0]-max_l/2, rec_centers[i][1]-max_l/2/ratio), max_l, max_l/ratio, color='b', alpha=0.3)
        axi.add_artist(rec)

    wedge = patches.Wedge(center, radius, start_angle * 180 / np.pi, end_angle * 180 / np.pi, fill=False, ec='r')
    axi.add_patch(wedge)
    axi.set_xlim([-radius-1, radius + 1])  
    axi.set_ylim([0, radius + 1])

# 没有暖启动，容易陷入局部最优
max_l, rec_centers = find_max_length_in_sector(center=center, radius=radius, start_angle=start_angle, end_angle=end_angle, ratio=ratio, n_rec=n_rec, seed=153)
print("最大矩形的长度为：", max_l)
print("所有矩形中心坐标为：", rec_centers)
plot_sector(ax, rec_centers)
plt.show()


# # 暖启动，可以缓解局部最优
# max_r, rec_centers = find_max_radius_in_sector(warm_start=True, center=center, radius=radius, start_angle=start_angle, end_angle=end_angle, n_circle=n_circle, seed=153)
# print("最大圆的半径为：", max_r)
# print("所有圆的圆心坐标为：", rec_centers)
# plot_sector(ax[1], rec_centers, max_r)
# ax[1].set_title('warm start')
