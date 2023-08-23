#!/usr/bin/env python
import datetime
import time
import sys
import rospy
import numpy as np
from copy import copy
from std_msgs.msg import Header #Need this for SunInfo? 2/9
from std_msgs.msg import String
from  basic_led_strip_proxy import BasicLedStripProxy
from basic_led_strip import BasicLedStrip
from basic_led_strip_ros.msg import StripLEDInfo
from basic_led_strip_ros.msg import SunInfo
from collections import Iterable

#from datetime import datetime

class ShuffleSun:


    def __init__(self):

        rospy.init_node('shuffle_sun')
        rospy.on_shutdown(self.sunset)
        self.sun_position_pub = rospy.Publisher('sun_position', SunInfo, queue_size=10)
        rate = rospy.Rate(10) #Rate we want to publish message in Hz #added 1/20 #12/7/22 changed from 5 12/7
        self.led_strip = BasicLedStripProxy(use_thread=True) #True? False?
        self.led_index_list = range(144)
        ######################################### 
        self.sun_positions = [19, 57, 93, 129]  #sun position in terms of actual LED number 19(135)18, 57(45)56, 93(-45)92, 129(-135)128
        #########################################
        self.current_sun_position = 0
        np.random.shuffle(self.sun_positions) #!!!comment out if you need the same sun(and put the same sun position in the very front)
        self.sun_positions*=2
        print(self.sun_positions)
        #####


    # def rotate_sun(self):
    #     self.sun_positions1 = [19,93]
    #     self.sun_positions2 = [57, 129]
    #     self.set_sun_positions = []
    #     sun = np.random.choice(self.sun_positions)
    #     self.set_sun_positions.append(sun)
    #     print('set_sun_positions:',sl)
    #     if self.set_sun_positions[-1] == 19 or 93:
    #         self.set_sun_positions.append(np.random.sample(self.sun_positions2, 1))
    #     else:
    #         self.set_sun_positions.append(np.random.sample(self.sun_positions1,1))

        
    def sunset(self):
        print('Sun is setting')
        self.led_strip.set_all(( 0, 0, 0))
        sys.exit()

    def publish_sun_position(self):
        sun_msg = SunInfo()
        sun_msg.header.stamp = rospy.Time.now()
        sun_msg.sun_position = self.current_sun_position
        #rospy.loginfo('Sun Info:' + str(sun_msg)) #shows what is being published as a message
        self.sun_position_pub.publish(sun_msg)
    
    def dark(self):
        self.led_strip.set_all((  0,   0,   0))
        self.current_sun_position = 0

###original######
    def sun_sample_no_replacement(self):

        for sun_position in self.sun_positions:
            self.current_sun_position = sun_position
            self.led_strip.set_led(self.current_sun_position,(0,8,0)) #original (0,128,0) -lowered brightness
            yield sun_position


    def run(self):

        while not rospy.is_shutdown():

            """
            This experiment consists of the following procedure:
                1) Dark period                                                        - 30 seconds
                2) Random light from positions [19(135), 57(45), 93(-45), 129(-135)]  - 5 minute
                3) Random light from positions [19, 57, 93, 129] w/o replacement      - 5 minute
                4) Dark period                                                        - 30 seconds
            
            """
            sun_sampling = self.sun_sample_no_replacement()
  
            procedure = [

                        # #control group (1 sun position)
                        #     [self.dark,30],
                        #     [sun_sampling,1200],
                            
                            

                        # #for 6 suns
                            [self.dark,30],

                            [sun_sampling,300], 
           
                            [sun_sampling,300] ,
  
                            [sun_sampling,300],  
                     
                            [sun_sampling,300] 
                             
                            # [sun_sampling,300],   
                    
                            # [sun_sampling,300]


                        ]
            self.experiment(procedure)
            self.sunset() # sys.exit()

    def experiment(self,procedure):
        for step, timestep in procedure:
            if isinstance(step,Iterable):
                try:
                    next(step)
                except StopIteration:
                    print("step {} is no longer iterable".format(step.__name__))
            else:
                step()
            self.publish_sun_position()
            print('sun_position: ' + str(self.current_sun_position))
            time.sleep(timestep)


#####original#####
    # def experiment(self,procedure):
    #     for step, timestep in procedure:
    #         if isinstance(step,Iterable):
    #             if any(step):
    #                 next(step)
    #             else:
    #                 print("step {} is no longer iterable".format(step.__name__))
    #         else:
    #             step()
    #         self.publish_sun_position()
    #         print('sun_position: ' + str(self.current_sun_position))
    #         time.sleep(timestep)

#-------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    node = ShuffleSun()
    node.run()


