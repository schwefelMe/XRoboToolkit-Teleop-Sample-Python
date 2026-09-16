from robot_arm import G1_29_ArmController
import time
if __name__ == "__main__":
    arm = G1_29_ArmController(demain_id=2,network='enp131s0')
    # arm.ctrl_dual_arm_go_home()
    for i in range(100):

        current_lr_arm_q  = arm.get_current_dual_arm_q()
        current_lr_arm_dq = arm.get_current_dual_arm_dq()
        print("Current Left Arm Q: ", current_lr_arm_q)
        print("Current Left Arm DQ: ", current_lr_arm_dq)
        time.sleep(0.1)