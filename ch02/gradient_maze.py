# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def softmax_convert_into_pi_from_theta(theta):

    beta= 1.0
    [m,n] = theta.shape
    pi = np.zeros((m,n))
    exp_theta = np.exp(beta*theta)

    for i in range(0,m):
        pi[i,:] = exp_theta[i,:] / np.nansum(exp_theta[i,:])
    
    pi = np.nan_to_num(pi)

    return pi

def get_action_and_next_s(pi, s):
    direction = ["up","right","down","left"]

    next_direction = np.random.choice(direction, p=pi[s,:])
    
    if next_direction == "up":
        action = 0
        s_next = s - 3
    elif next_direction == "right":
        action = 1
        s_next = s + 1
    elif next_direction == "down":
        action = 2
        s_next = s + 3
    elif next_direction == "left":
        action = 3
        s_next = s - 1
        
    return [action, s_next]

def goal_maze_ret_a(pi):
    s = 0
    # s_a_history = np.array([[0,np.nan]])
    s_a_history = [[0,np.nan]]

    while True:
        [action, next_s] = get_action_and_next_s(pi, s)
        s_a_history[-1][1] = action

        # s_a_history = np.append(s_a_history,[next_s, np.nan])
        s_a_history.append([next_s, np.nan])

        if next_s == 8:
            break
        else:
            s = next_s
    return s_a_history

def update_theta(theta, pi, s_a_history):
    eta = 0.1
    T = len(s_a_history) - 1 
    [m,n] = theta.shape
    delta_theta = theta.copy()
    for i in range(0,m):
        for j in range(0,n):
            if not(np.isnan(theta[i,j])):

                SA_i = [SA for SA in s_a_history if SA[0] == i]

                SA_ij = [SA for SA in s_a_history if SA == [i,j]]

                N_i = len(SA_i)
                N_ij = len(SA_ij)
                delta_theta[i,j] = (N_ij - pi[i,j] * N_i) / T

    new_theta = theta + eta * delta_theta
    return new_theta


# =================================================================================================================

# 初期の方策を決定するパラメータtheta_0を設定

# 行は状態0～7、列は移動方向で↑、→、↓、←を表す
theta_0 = np.array([[np.nan, 1, 1, np.nan],  # s0
                    [np.nan, 1, np.nan, 1],  # s1
                    [np.nan, np.nan, 1, 1],  # s2
                    [1, 1, 1, np.nan],  # s3
                    [np.nan, np.nan, 1, 1],  # s4
                    [1, np.nan, np.nan, np.nan],  # s5
                    [1, np.nan, np.nan, np.nan],  # s6
                    [1, 1, np.nan, np.nan],  # s7、※s8はゴールなので、方策はなし
                    ])

pi_0 = softmax_convert_into_pi_from_theta(theta_0)
# print(pi_0)

#チェック用コード
# #実行
# s_a_history = goal_maze_ret_a(pi_0)
# print(s_a_history)
# print("number of steps:" + str(len(s_a_history)-1))
# #方策の更新
# new_theta = update_theta(theta_0, pi_0, s_a_history)
# pi = softmax_convert_into_pi_from_theta(new_theta)
# print(pi_0)
# print(pi)

stop_epsilon = 10**-4
theta = theta_0
pi = pi_0

is_continue = True
count = 1
steps = []

while is_continue:
    s_a_history = goal_maze_ret_a(pi)
    new_theta = update_theta(theta, pi, s_a_history)
    new_pi = softmax_convert_into_pi_from_theta(new_theta)
    count += 1

    print(np.sum(np.abs(new_pi - pi)))
    print("number of steps:" + str(len(s_a_history)-1))
    steps.append(len(s_a_history))

    if np.sum(np.abs(new_pi-pi)) < stop_epsilon:
        is_continue = False
    else:
        theta = new_theta
        pi = new_pi

print(count)
np.set_printoptions(precision=3, suppress=True)
print(pi)


# エージェントの移動の様子を可視化します
# 参考URL http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-notebooks/
from matplotlib import animation
from IPython.display import HTML


def init():
    '''背景画像の初期化'''
    line.set_data([], [])
    return (line,)


def animate(i):
    '''フレームごとの描画内容'''
    state = s_a_history[i][0]  # 現在の場所を描く
    x = (state % 3) + 0.5  # 状態のx座標は、3で割った余り+0.5
    y = 2.5 - int(state / 3)  # y座標は3で割った商を2.5から引く
    line.set_data(x, y)
    return (line,)

fig = plt.figure(figsize = (5,5))
mngr = plt.get_current_fig_manager()
# to put it into the upper left corner for example:
mngr.window.setGeometry(800,100,500,500) #(X(左から),Y(上から),W,H)
ax = plt.gca()
plt.plot([1,1],[0,1],color='red',linewidth=2)
plt.plot([1,2],[2,2],color='red',linewidth=2)
plt.plot([2,2],[2,1],color='red',linewidth=2)
plt.plot([2,3],[1,1],color='red',linewidth=2)

plt.text(0.5,2.5,'S0',size=14,ha='center')
plt.text(1.5,2.5,'S1',size=14,ha='center')
plt.text(2.5,2.5,'S2',size=14,ha='center')
plt.text(0.5,1.5,'S3',size=14,ha='center')
plt.text(1.5,1.5,'S4',size=14,ha='center')
plt.text(2.5,1.5,'S5',size=14,ha='center')
plt.text(0.5,0.5,'S6',size=14,ha='center')
plt.text(1.5,0.5,'S7',size=14,ha='center')
plt.text(2.5,0.5,'S8',size=14,ha='center')
plt.text(0.5,2.3,'START',ha='center')
plt.text(2.5,0.3,'GOAL',ha='center')

# 描画範囲の設定と目盛りを消す設定
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
plt.tick_params(axis='both', which='both', bottom='off', top='off',labelbottom='off', right='off', left='off', labelleft='off')

# 現在地S0に緑丸を描画する
line, = ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)

#　初期化関数とフレームごとの描画関数を用いて動画を作成する
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(\
    s_a_history), interval=200, repeat=False)

# # HTML(anim.to_jshtml())

fig = plt.figure("number of steps for each iteration")
mngr = plt.get_current_fig_manager()
# to put it into the upper left corner for example:
mngr.window.setGeometry(1800,100,500,500) #(X(左から),Y(上から),W,H)
x = np.arange(len(steps))
plt.plot(x,steps,label="steps")
plt.ylim(0,max(steps)+10)
plt.xlabel("iteration")
plt.ylabel("number of steps")
plt.show()