import numpy as np
import threading
import time
from enum import IntEnum

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import ( LowCmd_  as hg_LowCmd, LowState_ as hg_LowState) # idl for g1, h1_2
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

import logging_mp
logger_mp = logging_mp.get_logger(__name__, logging_mp.WARNING)
ROBOT_NODE = 3 
 #  无线遥控的时候配置为2  有线遥控配置  3
if ROBOT_NODE == 1: 
    kTopicLowCommand_Debug  = "lowcmd_e"
    kTopicLowCommand_Motion = "arm_sdk_e"
    kTopicLowState = "lowstate_e"
    kTopicChassisCommand = "chasscmd_e"
elif ROBOT_NODE == 2:
    kTopicLowCommand_Debug  = "lowcmd_r"
    kTopicLowCommand_Motion = "arm_sdk_r"
    kTopicLowState = "lowstate_r"
    kTopicChassisCommand = "chasscmd_r"
elif ROBOT_NODE == 3:  # 直连机器人，不过转发节点
    kTopicLowCommand_Debug  = "rt/lowcmd"
    kTopicLowCommand_Motion = "rt/arm_sdk"
    kTopicLowState = "rt/lowstate"
    kTopicChassisCommand = "chasscmd_r"
# kTopicLowCommand_Debug  = "rt/lowcmd"
# kTopicLowCommand_Motion = "rt/arm_sdk"
# kTopicLowState = "rt/lowstate"

G1_29_Num_Motors = 35
G1_23_Num_Motors = 35
H1_2_Num_Motors = 35
H1_Num_Motors = 20
 

class MotorState:
    def __init__(self):
        self.q = None
        self.dq = None
        self.tau_est = None

class G1_29_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_29_Num_Motors)]

class G1_23_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_23_Num_Motors)]

class H1_2_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(H1_2_Num_Motors)]

class H1_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(H1_Num_Motors)]

class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data

class G1_29_ArmController:
    def __init__(
        self,
        motion_mode=False,
        simulation_mode=False,
        demain_id=0,
        network=None,
        waist_go_home_on_init=True,
        initial_q_target=None,
        enable_internal_gripper=False,
        internal_gripper_kp=2.0,
        internal_gripper_kd=0.05,
        hold_current_arm_on_init=False,
    ):
        logger_mp.info("Initialize G1_29_ArmController...")
        self.enable_internal_gripper = bool(enable_internal_gripper)
        if self.enable_internal_gripper and motion_mode:
            raise ValueError(
                "G1D built-in grippers require motion_mode=False because "
                "motors 31/33 accept commands on rt/lowcmd, not rt/arm_sdk"
            )
        self.internal_gripper_kp = float(internal_gripper_kp)
        self.internal_gripper_kd = float(internal_gripper_kd)
        self.hold_current_arm_on_init = bool(hold_current_arm_on_init)
        self.internal_gripper_target = None
        self.internal_gripper_action = None
        self.q_target = np.zeros(17) if initial_q_target is None else np.array(initial_q_target, dtype=float).copy() # 手臂 14 + 腰部 3
        if self.q_target.shape != (17,):
            raise ValueError(f"initial_q_target must have shape (17,), got {self.q_target.shape}")
        self.waist_yaw_position_target = None
        self.chassis_target = np.zeros(4) # 升降 lift + v_x + v_y + v_yaw
        self.tauff_target = np.zeros(17)
        self.motion_mode = motion_mode
        self.simulation_mode = simulation_mode
        self.kp_high = 300.0
        self.kd_high = 3.0
        self.kp_low = 150.0
        self.kd_low = 5.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.waist_velocity_limit = 5.0
        # self.control_dt = 1.0 / 250.0
        self.control_dt = 1.0 / 250.0
        self.control_chassis_dt = 1.0 / 20.0
        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None
        self._reset_flag = False # 是否需要重置目标位置
        self.actions = {"heart": (3.0,[np.array([-0.06004, 2.23009, 1.30004, 0.03002, -1.52003, -0.10996, -0.85004,
                                -0.00995, -2.22007, -1.39004, 0.02007, 1.24006, -0.060039, 0.79008])])}  # 执行时间和动作目标关节，支持复合动作
        # initialize lowcmd publisher and lowstate subscriber
        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            if network is not None:
                ChannelFactoryInitialize(demain_id, network)
            else:
                ChannelFactoryInitialize(demain_id) 

        if self.motion_mode:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
        else:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        
        self.chassis_cmd_publisher = ChannelPublisher(kTopicChassisCommand, WirelessController_) # 直接复用类似结构体
        self.chassis_cmd_publisher.Init()
        
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[G1_29_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[G1_29_ArmController] Subscribe dds ok.")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()
        
        self.chassis_msg = unitree_go_msg_dds__WirelessController_()

        self.all_motor_q = self.get_current_motor_q()
        self._initialize_arm_target_from_state()
        if self.enable_internal_gripper:
            self.internal_gripper_target = self.all_motor_q[[31, 33]].copy()
            self.internal_gripper_action = self.internal_gripper_target.copy()
        logger_mp.debug(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.debug(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")

        arm_indices = set(member.value for member in G1_29_JointArmIndex)
        internal_gripper_indices = {31, 33} if self.enable_internal_gripper else set()
        for id in G1_29_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in internal_gripper_indices:
                self.msg.motor_cmd[id].kp = self.internal_gripper_kp
                self.msg.motor_cmd[id].kd = self.internal_gripper_kd
            elif id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
        logger_mp.info("Lock OK!")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()
        
        self.publish_chassis_thread = threading.Thread(target=self._ctrl_chassis_state)
        self.ctrl_chassis_lock = threading.Lock()
        self.publish_chassis_thread.daemon = True
        self.publish_chassis_thread.start()

        if waist_go_home_on_init:
            self.ctrl_dual_arm_waist_go_home()

        logger_mp.info("Initialize G1_29_ArmController OK!")

    def _initialize_arm_target_from_state(self):
        if self.hold_current_arm_on_init:
            self.q_target[:14] = self.get_current_dual_arm_q()

    def _subscribe_motor_state(self):
        cnt = 0
        pre_time = time.time()
        while True:

            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                cnt += 1
                if cnt%100 ==0:
                    cur_time = time.time()
                    delta_t = cur_time - pre_time
                    logger_mp.info(f"[G1_29_ArmController] Subscribe dds rate: {100.0/delta_t} Hz")
                    # logger_mp.warning(f'waist.q: {msg.motor_state[G1_29_JointIndex.kWaistYaw].q}')
                    pre_time = cur_time
                lowstate = G1_29_LowState()
                # logger_mp.warning(f"waist pitch q: {msg.motor_state[G1_29_JointIndex.kWaistPitch].q}; waist yaw q: {msg.motor_state[G1_29_JointIndex.kWaistYaw].q}")
                for id in range(G1_29_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                    lowstate.motor_state[id].tau_est = getattr(msg.motor_state[id], "tau_est", 0.0)

                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q[:len(current_q)] - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        logger_mp.info(f'current_elbow.q:{current_q[-4]}, delta:{delta[-4]}, cliped_arm_q_target:{cliped_arm_q_target[-4]},velocity_limit:{velocity_limit}')
        # logger_mp.warning(f" cliped_arm_q_target: {cliped_arm_q_target}, \n current____________: {current_q}, \n delta: {delta}, motion_scale: {motion_scale}")
        return cliped_arm_q_target

    def clip_waist_q_target(self, increment, velocity_limit, position_target=None):
        current_q = self.get_current_waist_yaw_q()
        if position_target is not None:
            delta = float(position_target) - current_q
            motion_scale = abs(delta) / (velocity_limit * self.control_dt)
            return current_q + delta / max(motion_scale, 1.0)

        if self._reset_flag:
            increment = 1 if current_q < 0 else -1
        cliped_waist_q_target = current_q + float(increment) * (velocity_limit * self.control_dt)
        
        return cliped_waist_q_target
    
    def _ctrl_motor_state(self):
        if self.motion_mode:
            self.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = 1.0

        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target     = self.q_target[:14]
                waist_q_target = self.q_target[14] #腰部yaw自由度
                waist_yaw_position_target = self.waist_yaw_position_target
                arm_tauff_target = self.tauff_target[:14]
                internal_gripper_target = (
                    None
                    if self.internal_gripper_target is None
                    else self.internal_gripper_target.copy()
                )

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)
            
            for idx, id in enumerate(G1_29_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]

            cliped_waist_q_target = self.clip_waist_q_target(
                waist_q_target,
                velocity_limit = self.waist_velocity_limit,
                position_target = waist_yaw_position_target,
            )
            self.msg.motor_cmd[G1_29_JointIndex.kWaistYaw].q = cliped_waist_q_target
            self.msg.motor_cmd[G1_29_JointIndex.kWaistYaw].dq = 0
            self.msg.motor_cmd[G1_29_JointIndex.kWaistYaw].tau = 0

            if internal_gripper_target is not None:
                current_gripper_q = self.get_current_dual_internal_gripper_q()
                actual_gripper_action = np.clip(
                    internal_gripper_target,
                    current_gripper_q - 0.18,
                    current_gripper_q + 0.18,
                )
                actual_gripper_action = np.clip(actual_gripper_action, 0.0, 5.4)
                for motor_index, target in zip(
                    (31, 33), actual_gripper_action, strict=True
                ):
                    self.msg.motor_cmd[motor_index].q = float(target)
                    self.msg.motor_cmd[motor_index].dq = 0.0
                    self.msg.motor_cmd[motor_index].tau = 0.0
                with self.ctrl_lock:
                    self.internal_gripper_action = actual_gripper_action.copy()

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))
            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)


    def _ctrl_chassis_state(self):

        while True:
            start_time = time.time()

            with self.ctrl_chassis_lock:
                chassis_target = self.chassis_target

            self.chassis_msg.rx = chassis_target[0] # lift -0.5-0.5
            self.chassis_msg.lx = chassis_target[1] # v_x
            self.chassis_msg.ly = 0
            self.chassis_msg.ry = chassis_target[3]
            # logger_mp.warning(f'vx:{}')
            self.chassis_cmd_publisher.Write(self.chassis_msg)
            
            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_chassis_dt - all_t_elapsed))
            time.sleep(sleep_time)
            
    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors. 增加腰部3个自由度 一共17维数据'''
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target
            if len(q_target) > 14 and abs(float(q_target[14])) > 1e-6:
                self.waist_yaw_position_target = None

    def ctrl_dual_internal_gripper(self, q_target):
        if not self.enable_internal_gripper:
            raise RuntimeError("G1D built-in gripper control is not enabled")
        target = np.asarray(q_target, dtype=float)
        if target.shape != (2,) or not np.all(np.isfinite(target)):
            raise ValueError("built-in gripper q_target must contain two finite values")
        with self.ctrl_lock:
            self.internal_gripper_target = np.clip(target, 0.0, 5.4).copy()

    def ctrl_chassis(self, chassis_target):
        '''Set control target value q of the waist motor.'''
        with self.ctrl_chassis_lock:
            self.chassis_target = chassis_target

    def ctrl_waist_yaw(self, q_target):
        '''Set absolute target value q of the waist yaw motor.'''
        with self.ctrl_lock:
            self.waist_yaw_position_target = float(q_target)
            self.q_target[14] = 0.0

    def ctrl_waist_yaw_delta(self, q_delta):
        '''Set relative target value q of the waist yaw motor.'''
        with self.ctrl_lock:
            current_target = self.waist_yaw_position_target
            base_q = self.get_current_waist_yaw_q() if current_target is None else current_target
            self.waist_yaw_position_target = base_q + float(q_delta)
            self.q_target[14] = 0.0

    def execute_pose(self, actions_name):
        if actions_name not in self.actions:
            logger_mp.warning(f'unsupported action: {actions_name}')
            return
        logger_mp.warning(f'执行：{actions_name}动作，请注意周围空间人员及物品，请不要执行其他遥操作！！')
        duration_time = self.actions[actions_name][0]
        transition_steps = round(duration_time/self.control_dt)
        actions = self.actions[actions_name][1]

        cur_q = self.get_current_dual_arm_q() # 14 维度
        q_target = np.zeros(17)
        tauff_target = np.zeros(17)
        if actions_name != "restore":
            self.actions["restore"] = (duration_time, [cur_q])
        for action in actions:
            for i in range(transition_steps):
                phase = i / transition_steps
                smooth_phase = 3.0*phase*phase -2.0*phase*phase*phase
                q_target[:14] =  action[:14]*smooth_phase + cur_q[:14]*(1-smooth_phase)
                # for j in range(len(G1_29_JointArmIndex)): # 14个关节，暂时写死了
                #     q_target[i] = action[i]*smooth_phase - cur_q[i]*(1-smooth_phase) # 平滑目标
                self.ctrl_dual_arm(q_target, tauff_target)
                time.sleep(self.control_dt)

        logger_mp.warning(f'{actions_name}动作执行完成')


    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        return self.lowstate_subscriber.Read().mode_machine
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointIndex])

    def get_current_dual_internal_gripper_q(self):
        state = self.lowstate_buffer.GetData().motor_state
        return np.array([state[31].q, state[33].q], dtype=float)

    def get_current_dual_internal_gripper_dq(self):
        state = self.lowstate_buffer.GetData().motor_state
        return np.array([state[31].dq, state[33].dq], dtype=float)

    def get_current_dual_internal_gripper_tau_est(self):
        state = self.lowstate_buffer.GetData().motor_state
        return np.array([state[31].tau_est, state[33].tau_est], dtype=float)

    def get_current_dual_internal_gripper_action(self):
        if not self.enable_internal_gripper:
            raise RuntimeError("G1D built-in gripper control is not enabled")
        with self.ctrl_lock:
            return self.internal_gripper_action.copy()

    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in G1_29_JointArmIndex])
    
    def get_current_waist_pitch_q(self):
        '''Return current state q of the waist pitch motor.'''
        id = G1_29_JointIndex.kWaistPitch.value
        return self.lowstate_buffer.GetData().motor_state[id].q
    def get_current_waist_pitch_dq(self):
        '''Return current state dq of the waist pitch motor.'''
        id = G1_29_JointIndex.kWaistPitch.value
        return self.lowstate_buffer.GetData().motor_state[id].dq
    def get_current_waist_yaw_q(self):
        '''Return current state q of the waist yaw motor.'''
        id = G1_29_JointIndex.kWaistYaw.value
        return self.lowstate_buffer.GetData().motor_state[id].q
    def get_current_waist_yaw_dq(self):
        '''Return current state dq of the waist yaw motor.'''
        id = G1_29_JointIndex.kWaistYaw.value
        return self.lowstate_buffer.GetData().motor_state[id].dq
    def ctrl_dual_arm_go_home_slow(self):
        cur_q = self.get_current_dual_arm_q()
        action = np.zeros(17)
        tauff_target = np.zeros(17)
        q_target = np.zeros(17)
        transition_steps = 3.0/self.control_dt
        for i in range(transition_steps):
                phase = i / transition_steps
                smooth_phase = 3.0*phase*phase -2.0*phase*phase*phase
                q_target[:14] =  action[:14]*smooth_phase + cur_q[:14]*(1-smooth_phase)
                # for j in range(len(G1_29_JointArmIndex)): # 14个关节，暂时写死了
                #     q_target[i] = action[i]*smooth_phase - cur_q[i]*(1-smooth_phase) # 平滑目标
                self.ctrl_dual_arm(q_target, tauff_target)
                time.sleep(self.control_dt)
        return

    def ctrl_dual_arm_go_home(self, max_attempts_cnt=100, g_target=None):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        logger_mp.info("[G1_29_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        # record_vel = self.arm_velocity_limit
        # self.arm_velocity_limit = self.arm_velocity_limit/delay_radio
        # record_speed_set = self._speed_gradual_max
        # self._speed_gradual_max = False
        # self.arm_velocity_limit = 10.0
        with self.ctrl_lock:
            self.q_target = np.zeros(17) if g_target is None else g_target
        return
            # self.tauff_target = np.zeros(14)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        current_q = self.get_current_dual_arm_q()
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            # logger_mp.warning(f"[G1_29_ArmController] Current dual arm q: {current_q}")
            if np.all(np.abs(current_q) < tolerance):
                if self.motion_mode:
                    for weight in np.linspace(1, 0, num=101):
                        self.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = weight
                        time.sleep(0.02)
                logger_mp.error("[G1_29_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)
        if not np.all(np.abs(current_q) < tolerance) :
            logger_mp.error("[G1_29_ArmController] Warning: both arms failed to reach the home position within the maximum number of attempts.")
            logger_mp.error(f"[G1_29_ArmController] Current dual arm q: {current_q}")
        # self._speed_gradual_max = record_speed_set
        # self.arm_velocity_limit = record_vel
    def ctrl_left_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        logger_mp.info("[G1_29_ArmController] ctrl_left_arm_go_home start...")
        with self.ctrl_lock:
            self.q_target[0:7] = 0.0  # Left arm joints

    def ctrl_right_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        logger_mp.info("[G1_29_ArmController] ctrl_right_arm_go_home start...")
        with self.ctrl_lock:
            self.q_target[7:14] = 0.0  # Right arm joints
    def ctrl_dual_arm_waist_go_home(self, *, log_success_as_info=None):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        logger_mp.info("[G1_29_ArmController] ctrl_dual_arm_waist_go_home start...")
        max_attempts = 100
        current_attempts = 0
        self.ctrl_waist_yaw(0.0)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        current_q = self.get_current_waist_yaw_q()
        while current_attempts < max_attempts:
            current_q = self.get_current_waist_yaw_q()
            # logger_mp.warning(f"[G1_29_ArmController] Current dual arm q: {current_q}")
            if np.all(np.abs(current_q) < tolerance):

                if log_success_as_info is None:
                    log_success_as_info = getattr(
                        self, "enable_internal_gripper", False
                    )
                success_logger = (
                    logger_mp.info if log_success_as_info else logger_mp.error
                )
                success_logger(
                    "[G1_29_ArmController] waist have reached the home position."
                )
                break
            current_attempts += 1
            time.sleep(0.05)
        if not np.all(np.abs(current_q) < tolerance) :
            logger_mp.error("[G1_29_ArmController] Warning: waist failed to reach the home position within the maximum number of attempts.")
            logger_mp.error(f"[G1_29_ArmController] Current waist q: {current_q}")

    def ctrl_dual_arm_waist_go_right(self):
        '''Move waist yaw to the right-facing 90 degree position.'''
        logger_mp.info("[G1_29_ArmController] ctrl_dual_arm_waist_go_right start...")
        max_attempts = 100
        current_attempts = 0
        target_q = -np.pi / 2.0
        self.ctrl_waist_yaw(target_q)
        tolerance = 0.05
        current_q = self.get_current_waist_yaw_q()
        while current_attempts < max_attempts:
            current_q = self.get_current_waist_yaw_q()
            if abs(current_q - target_q) < tolerance:
                logger_mp.error("[G1_29_ArmController] waist has reached the right-facing position.")
                break
            current_attempts += 1
            time.sleep(0.05)
        if abs(current_q - target_q) >= tolerance:
            logger_mp.error("[G1_29_ArmController] Warning: waist failed to reach the right-facing position within the maximum number of attempts.")
            logger_mp.error(f"[G1_29_ArmController] Current waist q: {current_q}")

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            G1_29_JointIndex.kLeftAnklePitch.value,
            G1_29_JointIndex.kRightAnklePitch.value,
            # Left arm
            G1_29_JointIndex.kLeftShoulderPitch.value,
            G1_29_JointIndex.kLeftShoulderRoll.value,
            G1_29_JointIndex.kLeftShoulderYaw.value,
            G1_29_JointIndex.kLeftElbow.value,
            # Right arm
            G1_29_JointIndex.kRightShoulderPitch.value,
            G1_29_JointIndex.kRightShoulderRoll.value,
            G1_29_JointIndex.kRightShoulderYaw.value,
            G1_29_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            G1_29_JointIndex.kLeftWristRoll.value,
            G1_29_JointIndex.kLeftWristPitch.value,
            G1_29_JointIndex.kLeftWristyaw.value,
            G1_29_JointIndex.kRightWristRoll.value,
            G1_29_JointIndex.kRightWristPitch.value,
            G1_29_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors

class G1_29_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28
    
    # not used
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34
