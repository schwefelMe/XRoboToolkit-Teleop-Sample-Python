# G1D Teleopration
这个仓库是海信基于[XRoboToolkit-Teleop-Sample-Python](https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python)开发的，使用PICO VR遥操作G1D机器人。


## 遥操使用方式

### 1. PC端
打开应用程序[XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service)  ！！！！！

PC端要和PICO端处在同一个无线局域网（PICO Ultra4 企业版可以用USB有线连接，更稳定；普通版只能无线连接,）。

PC端要和机器人Orin处在同一个局域网下。

确保PC电脑已关闭防火墙`sudo ufw disable`
### 2. PICO端
PICO端需要[开启开发者模式](https://developer.picoxr.com/zh/document/unreal/test-and-build/)，同时需要把【开发者选项-企业设置-系统设置】中的【灭屏 和 系统休眠】都改为永不，防止遥操过程中系统休眠。
![替代文字](media/pico1.png "可选标题")

打开应用程序[XRoboToolkit-PICO-1.1.1.apk](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk)，跟PC端建立连接后，根据需求勾选发送手柄、灵巧手、体感追踪器的选项。
PICO眼镜USB连接电脑并勾选Shraed network（connect USB first）后PICO眼镜会跳出IP地址，点击连接
![替代文字](media/apk_ui.jpg "可选标题")
此时电脑端有有线和PICO两个连接
![alt text](image.png)
如果是PICO 4U Ultra4 企业版，建议用数据线连接笔记本，以获得更稳定的位姿获取
USB以太网下PICO会在pico有线连接笔记本时出现
如果是非企业版，需要将pico wifi设置为和笔记本同一个局域网下，然后在XRoboToolkit-PICO-1.1.1程序界面中点击Enter笔记本对应的ip地址

如果STATUS显示为Working，则表示pico和笔记本连接成功

### 3. 实机遥操

    

第一步，确认机器人和笔记本网络联通成功：
  将笔记本和机器人连接到同一个局域网下，建议都用网线连接，以获得更稳定的数据传输效果
  在笔记本端执行：
  
  ’’’
  ping 192.168.50.196（机器人对应ip，也可能是192.168.50.201）,ping通则连通成功
  '''
  
第二步，ssh连接机器人：
SSH 192.168.50.196或201 连接机器人（建议使用termius点击对应ip连接）密码123，连接机器人之后，终端开头显示
    
  
![alt text](image-1.png)
![alt text](image-2.png)

第三步：启动智元夹爪及转换节点：
  以下三条命令在orin端执行
  在一个终端上执行：
  sudo chmod 777 -R /dev/ttyUSB*
  /home/unitree/unitree_eai_environment/service/omnipicker/build/example_dds --interface wlan0

  如果执行正常，则终端显示：![alt text](image-5.png)
    

  在另一个终端上执行
  python /home/unitree/unitree_sdk2_python/example/g1/transform_node_chassis.py

  如果执行正常，则终端显示;![alt text](image-6.png)

如执行命令时出现：


![alt text](image-3.png)

说明夹爪和机器人没有连接，找硬件大哥确认

第四步：本地（5090 推理机端）启动：

G1D 遥操和采集入口使用 `g1_teleop` 环境，并把仓库内的 Unitree SDK 加入 Python 搜索路径：

 conda activate g1_teleop
 cd /home/ysm/coderepo/XRoboToolkit-Teleop-Sample-Python
 export PYTHONPATH="$PWD/unitree_sdk2_python:$PWD/scripts/hardware:$PYTHONPATH"

 原有机器人继续使用下面的默认命令；不传 `--gripper-type` 时仍为 Omni Hand，原 Omni URDF 和控制器保持不变：

 python /home/ysm/coderepo/XRoboToolkit-Teleop-Sample-Python/scripts/hardware/teleop_G1D_hardware_omnihand.py

 新 G1D 使用原装平动夹爪时必须显式选择内置夹爪后端。该分支会自动连接 D435 RGB 图像服务，不要再单独启动 `image_client_g1.py`：

 python /home/ysm/coderepo/XRoboToolkit-Teleop-Sample-Python/scripts/hardware/teleop_G1D_hardware_omnihand.py --gripper-type unitree_builtin --camera-host 192.168.123.164

 若执行成功，则会弹出 Placo 网页和已启用的摄像头画面。可在图像窗口按 `q` 或 `Esc` 结束遥操。![alt text](image-8.png)
 
 新 G1D 的有线图像地址默认为 `192.168.123.164`。机器人地址变化时可传入 `--camera-host`，或者在启动前设置环境变量：

 XR_TELEOP_ROBOT_IP=192.168.123.164 python /home/ysm/coderepo/XRoboToolkit-Teleop-Sample-Python/scripts/hardware/teleop_G1D_hardware_omnihand.py --gripper-type unitree_builtin

 `start_new_g1d_d435_image_client.sh` 仅保留为单独排查图像服务时的诊断工具，正常遥操不启动它。

 如果运行内置数据采集程序，显式传入同一个相机地址。原有 Omni 机器人仍可不传夹爪参数：

 python /home/ysm/coderepo/XRoboToolkit-Teleop-Sample-Python/scripts/hardware/teleop_recorder.py --camera-host 192.168.123.164

 新 G1D 的 D435 与原装平动夹爪组合使用：

 python /home/ysm/coderepo/XRoboToolkit-Teleop-Sample-Python/scripts/hardware/teleop_recorder.py --camera-host 192.168.123.164 --gripper-type unitree_builtin

`unitree_builtin` 后端从 `rt/lowstate` 的 31/33 号电机读取左右夹爪状态，并通过与双臂共享的 `rt/lowcmd` 写入器发送夹爪目标；实机已确认 `rt/arm_sdk` 在这台 G1D 上没有订阅者，不能用于原装夹爪。该后端不依赖 `/dev/ttyUSB*`，也不启动外置 `dex1_1_service`。不要同时运行其他 `rt/lowcmd` 写入程序。

 `unitree_builtin` 的 Pico 夹爪映射为：松开手指扣机=夹爪闭合，按下扣机=夹爪打开。普通遥操和录制程序共享同一个双臂微弯初始姿态；长按 `A+X` 后会持续下发该目标，并等待实测关节进入容差后再恢复 IK。Omni 分支不使用这两项新逻辑。

 以上 `unitree_builtin` 配置仅用于新 G1D；其他机器人继续使用各自原有启动入口和默认 Omni 配置。

此时遥操已经启动，以下是调试配置

  夹爪测试：
  在遥操作软件无法控制夹爪时，可以先在测试夹爪是否正常，
    在orin端执行：
  /home/unitree/unitree_eai_environment/service/omnipicker/build/example_dds_client --gripper 1 --demo
  gripper 1 对应右手夹爪   gripper 2 对应左手夹爪 
  如果夹爪有相应动作，则排查遥操软件问题：
  否则找硬件相关人员排查夹爪硬件问题
  
  图像测试：
  在orin端执行：
  ps aux | grep image
  root        1107 67.1  1.2 2302484 196964 ?      Ssl  09:56  28:28
    /home/unitree/miniconda3/envs/tv/bin/python3.10 /home/unitree/miniconda3/envs/tv/bin/teleimager-server
  如果没有运行，则启动图像服务：
  

  机器人Orin切换网络（推荐网线直连情况下切换）：
  第一种方法：
   ssh连接上机器人之后，执行 bash no_machine.sh
   打开笔记本端nomachnie.sh，修改对应的ip（最好是直连ip）
   在图形界面直接切换网络

  第二种方法：如遇到问题，请咨询ai
    查看wifi是否使能：sudo nmcli radio wifi
    查看wifi列表：sudo nmcli device wifi list
    连接wifi：sudo nmcli device wifi connect <ssid> password <password>
```
机器人速度调整：
teleop_G1D_hardware_omnihand.py 中154行
ctrl_chassis[0] = 0.7* round(right_joystick_y) # lift 控制升降速度
ctrl_chassis[1] = 0.3* round(left_joystick_y) # v_x  控制前进后退速度
ctrl_chassis[3] = -0.5* round(left_joystick_x) # v_yaw 控制转弯速度
