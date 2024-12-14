import carla
import torch
import torch.nn as nn

# 定数の定義
D     = 128  # エンコーディングの次元数 (特徴ベクトルの次元数)
N_A   = 10   # 動的エージェントの数 (他の車両や歩行者など)
N_S   = 5    # 静的エージェントの数 (信号機、標識など)
N_P   = 20   # ポリラインの数 (道路やレーンの線形情報)
B     = 2    # バッチサイズ


# CARLAからデータを取得してエンコード情報を生成する関数
def get_encoded_data_from_carla(world, ego_vehicle):
    # エージェントのエンコード情報を取得
    agents = world.get_actors().filter('vehicle.*')
    E_A = []
    for agent in agents[:N_A]:
        location = agent.get_location()
        velocity = agent.get_velocity()
        # エージェントの位置、速度、加速度などを用いてエンコード
        E_A.append([location.x, location.y, location.z, velocity.x, velocity.y, velocity.z])
    E_A = torch.tensor(E_A).view(N_A, B, D)

    # 障害物のエンコード情報を取得
    obstacles = world.get_actors().filter('static.*')
    E_O = []
    for obstacle in obstacles[:N_S]:
        location = obstacle.get_location()
        # 障害物の位置を用いてエンコード
        E_O.append([location.x, location.y, location.z])
    E_O = torch.tensor(E_O).view(N_S, B, D)

    # ポリラインのエンコード情報を取得
    waypoints = world.get_map().generate_waypoints(distance=2.0)
    E_P = []
    for waypoint in waypoints[:N_P]:
        location = waypoint.transform.location
        # ポリラインの座標を用いてエンコード
        E_P.append([location.x, location.y, location.z])
    E_P = torch.tensor(E_P).view(N_P, B, D)

    # 自律車両の状態エンコード情報を取得
    location = ego_vehicle.get_location()
    velocity = ego_vehicle.get_velocity()
    acceleration = ego_vehicle.get_acceleration()
    E_AV = torch.tensor([[location.x, location.y, location.z, velocity.x, velocity.y, velocity.z, acceleration.x, acceleration.y, acceleration.z]])
    E_AV = E_AV.view(1, B, D)  # (1, B, D) の形状に整形

    # 全てのエンコード情報を返す
    return E_A, E_O, E_P, E_AV

#================================================#
#================================================#

# CARLAクライアントの初期化
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# 自車両（エゴビークル）の取得
ego_vehicle = world.get_actors().filter('vehicle.*')[0]

# CARLAからデータを取得してエンコード情報を生成
E_A, E_O, E_P, E_AV = get_encoded_data_from_carla(world, ego_vehicle)

