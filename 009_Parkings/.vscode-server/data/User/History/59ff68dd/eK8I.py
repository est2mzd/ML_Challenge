import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import imageio
from matplotlib.patches import Rectangle

# ==== パラメータ設定 ====
L = 4.7  # 車両の全長
W = 2.0  # 車両の全幅
T = 0.2  # シミュレーション時間刻み
N = 100   # 予測ホライゾン

# 自車（AV）の初期位置
x_init, y_init, phi_init, v_init = -10, 0, 0, 0

# 目標駐車位置
x_goal, y_goal, phi_goal = 0, 5, np.pi/2

# 障害物（他車）の位置
obstacles = [
    (0, 7),  # 車1
    (0, 3)   # 車2
]

# ==== NMPC 最適化設定 ====
opti = ca.Opti()
X = opti.variable(4, N+1)  # 状態変数 [x, y, phi, v]
U = opti.variable(2, N)    # 制御入力 [delta, a]

# 初期状態
opti.subject_to(X[:, 0] == ca.vertcat(x_init, y_init, phi_init, v_init))

# モデルダイナミクス制約
for k in range(N):
    x_k = X[0, k]
    y_k = X[1, k]
    phi_k = X[2, k]
    v_k = X[3, k]
    delta_k = U[0, k]
    a_k = U[1, k]
    x_next = x_k + T * v_k * ca.cos(phi_k)
    y_next = y_k + T * v_k * ca.sin(phi_k)
    phi_next = phi_k + T * (v_k / L) * ca.tan(delta_k)
    v_next = v_k + T * a_k
    opti.subject_to(X[:, k+1] == ca.vertcat(x_next, y_next, phi_next, v_next))

# 目標状態へのコスト関数
Q = np.diag([10, 10, 1, 1])  # 状態の重み
R = np.diag([1, 1])  # 制御入力の重み
cost = 0
for k in range(N):
    cost += ca.mtimes((X[:, k] - ca.vertcat(x_goal, y_goal, phi_goal, 0)).T, Q) @ (X[:, k] - ca.vertcat(x_goal, y_goal, phi_goal, 0))
    cost += ca.mtimes(U[:, k].T, R) @ U[:, k]
opti.minimize(cost)

# 制御変数の制約
opti.subject_to(opti.bounded(-np.pi/6, U[0, :], np.pi/6))  # ステアリング角制約
opti.subject_to(opti.bounded(-1, U[1, :], 1))  # 加速度制約

# ソルバー設定
opti.solver('ipopt')
sol = opti.solve()
X_opt = sol.value(X)
U_opt = sol.value(U)

# ==== アニメーション生成 ====
frames = []
fig, ax = plt.subplots(figsize=(6, 6))
for i in range(N+1):
    ax.clear()
    ax.set_xlim(-12, 8)
    ax.set_ylim(-2, 10)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    # 自車描画
    rect = Rectangle((X_opt[0, i] - L/2, X_opt[1, i] - W/2), L, W, np.degrees(X_opt[2, i]), color='blue', alpha=0.7)
    ax.add_patch(rect)
    
    # 障害物描画
    for ox, oy in obstacles:
        ax.add_patch(Rectangle((ox - L/2, oy - W/2), L, W, color='red', alpha=0.7))
    
    # 目標位置
    ax.scatter(x_goal, y_goal, color='green', marker='x', s=100)
    
    # フレーム保存
    fig.canvas.draw()
    image = np.array(fig.canvas.renderer.buffer_rgba())
    frames.append(image)

# GIF作成
gif_path = "./parking_simulation.gif"
imageio.mimsave(gif_path, frames, duration=0.2)
print(f"GIF saved at {gif_path}")
