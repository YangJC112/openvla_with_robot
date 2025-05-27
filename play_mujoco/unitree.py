import mujoco
import mujoco.viewer

# 加载宇树机器人模型
model = mujoco.MjModel.from_xml_path(r"F:\td4\Sii\code\play_mujoco\unitree_rl_gym\resources\robots\g1_description\scene.xml")
data = mujoco.MjData(model)

# 启动查看器
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()