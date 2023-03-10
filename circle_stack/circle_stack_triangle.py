"""_summary_
!pip install dccp
"""
import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import dccp



def find_max_radius_in_triangle(points:np.ndarray, seed:int=134, n_circle:int=20):
    """_summary_

    Args:
        points (np.ndarray): 逆时针的三个点
        seed (int, optional): 由于是非凸优化，随机数可能对结果有影响 Defaults to 134.
        n_circle (int, optional): 需要放置的圆的数量

    Returns:
        半径：最大半径
        所有圆的圆心坐标
    """
    np.random.seed(seed)
    n = n_circle
    r = cvx.Variable()
    c = cvx.Variable((n, 2))
    constr = []
    for i in range(n - 1):
        constr.append(cvx.norm(cvx.reshape(c[i, :], (1, 2)) - c[i + 1: n, :], 2, axis=1) >= 2 * r)
    constraint = []
    for p1, p2 in [[0,1],[1,2],[2,0]]:
        (x0,y0) = points[p1]
        (x1,y1) = points[p2]
        l = np.sqrt((x0-x1)**2+(y0-y1)**2)
        x_coef = (y0-y1)/l
        y_coef = (x1-x0)/l
        constant = (x0*y1-x1*y0)/l
        constraint.append((x_coef,y_coef,constant))
        constr.append(c[:, 0]*x_coef+c[:,1]*y_coef+constant>=r)
    prob = cvx.Problem(cvx.Maximize(r), constr)
    prob.solve(method="dccp", solver="ECOS", ep=1e-6, max_slack=1e-2)
    return r.value, [(c[i, 0].value, c[i,1].value) for i in range(n)]

# 逆时针摆放三个点描述任意三角形
points = np.array([[-5,0],[5,1],[0,8]])

# 大了就算得慢了,100需要两分钟左右
n_circle = 100
max_r, circle_centers = find_max_radius_in_triangle(points, n_circle=n_circle)
print("最大可能半径：",max_r)

# plot
cic_points= np.concatenate([points, points[:1,:]],axis=0)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fig.set_tight_layout(True)
for i in range(n_circle):
    circ = plt.Circle((circle_centers[i][0], circle_centers[i][1]), max_r, color='b', alpha=0.3)
    ax.add_artist(circ)

ax.plot(cic_points[:,0], cic_points[:,1], "g")
ax.set_xlim([-6, 6])
ax.set_ylim([-2, 10])
plt.show()
