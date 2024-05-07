'''
Author: FANG Zhiwei
Email: zwfang@link.cuhk.edu.hk/ afjsliny@gmail.com
Date: 2024-03-18 16:20:57
LastEditTime: 2024-04-23 20:41:54
Description: Set the communication between the scripts and the VREP simulator.
'''

import sys
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# CartPole simulation model for VREP
class skullbotSimModel():

    def __init__(self, name='skullbot'):

        # super(self.__class__, self).__init__()
        self.name = name
        self.client_ID = None
        self.sim = None

        self.world_handle = None
        self.Rotor1_joint_handle = None
        self.Slider1_joint_handle = None
        self.Rotor2_joint_handle = None
        self.Slider2_joint_handle = None
        self.needle_driver_joint_handle = None
        self.sim_OCT_vision_sensor_handle = None

        self.object1_handle = None
        self.object2_handle = None
        self.object3_handle = None
        self.object4_handle = None

        self.needleDummy_handle = None
        # self.sim.boolparam_display_enabled = False

    def initializeSimModel(self, sim):

        self.sim = sim
        # sim.setBoolParam(sim.boolparam_display_enabled,False)

        self.Rotor1_joint_handle = self.sim.getObjectHandle('Rotor1_joint')
        print('get object Rotor1_joint ok.')
        self.Slider1_joint_handle = self.sim.getObjectHandle('Slider1_joint')
        print('get object Slider1_joint ok.')
        self.Rotor2_joint_handle = self.sim.getObjectHandle('Rotor2_joint')
        print('get object Rotor2_joint ok.')
        self.Slider2_joint_handle = self.sim.getObjectHandle('Slider2_joint')
        print('get object Slider2_joint ok.')

        self.needle_driver_joint_handle = self.sim.getObjectHandle('needle_driver_joint')
        print('get object needle_driver_joint ok.')

        self.sim_OCT_vision_sensor_handle = self.sim.getObjectHandle('sim_OCT_vision_sensor')
        print('get object sim_OCT_vision_sensor ok.')
        # Get the joint position
        # q = vrep_sim.simxGetJointPosition(self.client_ID, self.prismatic_joint_handle, vrep_sim.simx_opmode_streaming)
        # q = vrep_sim.simxGetJointPosition(self.client_ID, self.revolute_joint_handle, vrep_sim.simx_opmode_streaming)

        # Set the initialized position for each joint
        # self.setJointTorque(0)
        self.setJointPosition('Rotor1_joint', 0)
        self.setJointPosition('Slider1_joint', 0)
        self.setJointPosition('Rotor2_joint', 0)
        self.setJointPosition('Slider2_joint', 0)
        self.setJointPosition('needle_driver_joint', 0)

        '''for random object setting'''
        # self.object1_handle = self.sim.getObjectHandle('object1')
        # self.object2_handle = self.sim.getObjectHandle('object2')
        # self.object3_handle = self.sim.getObjectHandle('object3')
        # self.object4_handle = self.sim.getObjectHandle('object4')
        # self.object_handles = [self.object1_handle, self.object2_handle, self.object3_handle, self.object4_handle]

        self.needleDummy_handle = self.sim.getObjectHandle('needleDummy')
    

    def getJointPosition(self, joint_name):
        '''
        description: 
        param {*} self
        param {*} joint_name : string for the joint
        return {*}
        '''
        q = 0
        if joint_name == 'Rotor1_joint':
            q = self.sim.getJointPosition(self.Rotor1_joint_handle)
        elif         joint_name == 'Slider1_joint':
            q = self.sim.getJointPosition(self.Slider1_joint_handle)
        elif         joint_name == 'Rotor2_joint':
            q = self.sim.getJointPosition(self.Rotor2_joint_handle)
        elif         joint_name == 'Slider2_joint':
            q = self.sim.getJointPosition(self.Slider2_joint_handle)   
        elif         joint_name == 'needle_driver_joint':
            q = self.sim.getJointPosition(self.needle_driver_joint_handle)

        else:
            print('Error: joint name: \' ' + joint_name + '\' can not be recognized.')

        return q
    
    def getDummyPosition(self, dummy_handle):
        dummy = self.sim.getObjectPosition(dummy_handle, relativeToObjectHandle = self.sim.handle_world)
        dummy_posi = dummy[:3]
        return dummy_posi
    
    def setObjectPosition(self, object_handle, translation):
        pass

    def getVisionSensorCharImage(self, vision_sensor_name):
        if vision_sensor_name == 'sim_OCT_vision_sensor':
            image, resX, resY = self.sim.getVisionSensorCharImage(self.sim_OCT_vision_sensor_handle)
            # imageMatrix = np.frombuffer(image, dtype=np.uint8).reshape([reso[1], reso[0], 3])
            # imageMatrix = cv2.flip(imageMatrix, 0)
            # new_image = Image.fromarray(imageMatrix)
            # new_image = np.array(new_image)
            # gray_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
            return image, resX, resY
        else:
            print('Error: vision sensor handle: \' ' + vision_sensor_name + '\' can not be recognized.')

    def setObjVisible(self, visible_object_handle):

        for handle in self.object_handles:
            self.sim.setObjectIntParameter(handle, self.sim.sim_objintparam_visibility_layer, 0)
        self.sim.setObjectIntParameter(visible_object_handle, self.sim.sim_objintparam_visibility_layer, 1)

    def setObjectPosition(self, object_handle, translation):
        '''
        translation: a [x, y] list, representing the x and y translation w.r.t. the world frame 
        '''
        self.sim.setObjectPosition(object_handle, translation,  relativeToObjectHandle = self.sim.handle_world)


    def zeroingJoints(self):
        self.skullbot_sim_model.setJointPosition('Rotor1_joint', 0)
        self.skullbot_sim_model.setJointPosition('Slider1_joint', 0)
        self.skullbot_sim_model.setJointPosition('Rotor2_joint', 0)
        self.skullbot_sim_model.setJointPosition('Slider2_joint', 0)
        self.skullbot_sim_model.setJointPosition('needle_driver_joint', 0)
        return None









    def getJointVelocity(self, joint_name):
        """
        :param: joint_name: string
        """
        v = 0
        if joint_name == 'Rotor1_joint':
            v = self.sim.getJointVelocity(self.Rotor1_joint_handle)
        elif joint_name == 'Slider1_joint':
            v = self.sim.getJointVelocity(self.Slider1_joint_handle)
        elif joint_name == 'Rotor2_joint':
            v = self.sim.getJointVelocity(self.Rotor2_joint_handle)
        elif joint_name == 'Slider2_joint':
            v = self.sim.getJointVelocity(self.Slider2_joint_handle)
        elif joint_name == 'needle_driver_joint':
            v = self.sim.getJointVelocity(self.needle_driver_joint_handle)
        else:
            print('Error: joint name: \' ' + joint_name + '\' can not be recognized.')

        return v

    def setJointPosition(self, joint_name, pos):
        """
        :param: joint_name: string
        """
        if joint_name == 'Rotor1_joint':
            self.sim.setJointPosition(self.Rotor1_joint_handle, pos)
        elif joint_name == 'Slider1_joint':
            self.sim.setJointPosition(self.Slider1_joint_handle, pos)
        elif joint_name == 'Rotor2_joint':
            self.sim.setJointPosition(self.Rotor2_joint_handle, pos)
        elif joint_name == 'Slider2_joint':
            self.sim.setJointPosition(self.Slider2_joint_handle, pos)
        elif joint_name == 'needle_driver_joint':
            self.sim.setJointPosition(self.needle_driver_joint_handle, pos)
        else:
            print('Error: joint name: \' ' + joint_name + '\' can not be recognized.')

        return 0


    def setJointTorque(self, torque):
        print('no torque here')
        # if torque >= 0:
        #     self.sim.setJointTargetVelocity(self.prismatic_joint_handle, 1000)
        # else:
        #     self.sim.setJointTargetVelocity(self.prismatic_joint_handle, -1000)

        # self.sim.setJointMaxForce(self.prismatic_joint_handle, abs(torque))
