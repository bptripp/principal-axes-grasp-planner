import numpy as np
from scipy.spatial.qhull import QhullError
import mesh_classes as mc

class ReFlex:
    def __init__(self):
        self.make_new_shell()
        self.finger_points = np.vstack([np.zeros(120),np.linspace(0,12,120)*1e-2,np.zeros(120)]).T

        self.H_finger_1 = np.zeros((4,4))
        self.H_finger_1[:3,:3] = np.eye(3)
        self.H_finger_1[:3,3] = np.array([-2.5,4.5,0])*1e-2
        self.H_finger_1[3,3] = 1

        self.H_finger_2 = np.zeros((4,4))
        self.H_finger_2[:3,:3] = np.eye(3)
        self.H_finger_2[:3,3] = np.array([2.5,4.5,0])*1e-2
        self.H_finger_2[3,3] = 1

        self.H_finger_3 = np.zeros((4,4))
        self.H_finger_3[:3,:3] = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        self.H_finger_3[:3,3] = np.array([0,-4.5,0])*1e-2
        self.H_finger_3[3,3] = 1

        self.orientation = np.eye(3)
        self.position = np.zeros(3)

    def make_new_shell(self, apply_pose=False):
        gripper_points = np.array([[0, 0, 0],
                                   [-4.5,6,0],
                                   [4.5,6,0],
                                   [4.5,-6,0],
                                   [-4.5,-6,0],
                                   [-4,4.5,-9],
                                   [4,4.5,-9],
                                   [4,-4.5,-9],
                                   [-4,-4.5,-9]])*1e-2

        if apply_pose:
            gripper_points = (self.orientation @ gripper_points.T + self.position).T
        self.shell = mc.PointCloud(gripper_points, form_hull=True)


    def finger_flex(self, theta):
        """
        Takes in a value of theta in radians

        Returns a finger centered at the origin flexed at the appropriate degree
        """
        RX = np.array([[1,0,0],[0,np.cos(theta), -np.sin(theta)],[0,np.sin(theta), np.cos(theta)]])
        flexed = self.finger_points @ RX.T
        return flexed

    def set_gripper_pose(self, R, p):
        """
        Set gripper pose with respect to the origin (table coordinates)
        """
        self.orientation = R
        self.position = p
        self.pose = np.zeros((4,4))
        self.pose[:3,:3] = R
        self.pose[:3,3] = p[:,0]
        self.pose[3,3] = 1
        return

    def check_valid_grasp(self, object_cloud):
        try:
            self.make_new_shell(apply_pose=True)
        except QhullError:
            print('Something wrong with forming shell')
            return False

        gripper_on_object = self.shell.check_collision(object_cloud.points)
        object_on_gripper = object_cloud.check_collision(self.shell.points)

        if np.sum(gripper_on_object) or np.sum(object_on_gripper):
            print('Collision between object and gripper')
            return False
        elif np.sum(self.shell.points[:,2]<=0):
            print('Gripper shell collides with table')
            return False

        dummy = np.ones([120,4])
        finger_contacts = [False, False, False]
        self._final_thetas = [0,0,0]
        for theta in np.deg2rad(np.linspace(0,120,120)):
            flexed = self.finger_flex(theta)
            dummy[:,:3] = flexed

            # Check if any of the fingers are under the table
            fingers = []
            transforms = [self.H_finger_1, self.H_finger_2, self.H_finger_3]
            for i,contact in enumerate(finger_contacts):
                # only check if the finger still han't reach the object.
                if not contact:
                    finger = (self.pose @ (transforms[i] @ dummy.T)).T
                    if np.sum(finger[:,2]<=0):
                        print('Finger {} collided with table before reaching object'.format(i))
                        return False
                fingers.append(finger)

            # Check if the finer has reached the object
            for i,finger in enumerate(fingers):
                if not finger_contacts[i]:
                    object_on_finger = object_cloud.check_collision(finger[:,:3])
                    if np.sum(object_on_finger):
                        if theta==0:
                            print("Fingers can't start inside object")
                            return False
                        else:
                            print('Finger {} touched object at {} radians'.format(i, theta))
                            finger_contacts[i] = True
                            self._final_thetas[i] = theta

        if not(finger_contacts[2] and (finger_contacts[0] or finger_contacts[1])):
            print('Fingers did not touch object')
            return False

        if not self._final_thetas[0]:
            self._final_thetas[0] = self._final_thetas[1]
        elif not self._final_thetas[1]:
            self._final_thetas[1] = self._final_thetas[0]
        return True
