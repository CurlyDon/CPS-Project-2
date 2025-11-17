########################################################################
# ROS system filler implementation for ECE430/ECE8 
# To use, import rospy with: import rospy-shim as rospy
# 
# DO NOT MODIFY THE FILE, OR THE RESULTS FOR YOUR CODE WILL DIFFER WHEN
# SUBMITTED AND AUTO-GRADED BY THE SYSTEM
#######################################################################

import subprocess
import time
import sys
import requests
import json
import threading

# Wrapper for ROS primitive types
# mapping to their msg specification
class Std_msgs():
    def __init__(self):
        self.name = ""

    def String(self):
        return "std_msgs/String"

    def Bool(self):
        return "std_msgs/Bool"

    def Int64(self):
        return "std_msgs/Int64"

    def Float64(self):
        return "std_msgs/Float64"

    def Uint16(self):
        return "std_msgs/Uint16"

    def Empty(self):
        return "std_msgs/Empty"

std_msgs = Std_msgs()

# Wrapper for ROS geometric primitives like
# points, vectors, and poses
class Geometry_msgs():
    def __init__(self):
        self.name = ""

    def Vector3(self):
        return "geometry_msgs/Vector3"

    def Twist(self):
        return "geometry_msgs/Twist"
    
    def TwistWithCovariance(self):
        return "geometry_msgs/TwistWithCovariance"

    def Point(self):
        return "geometry_msgs/Point"

    def PoseWithCovariance(self):
        return "geometry_msgs/PoseWithCovariance"

    def Quaternion(self):
        return "geometry_msgs/Quaternion"


geometry_msgs = Geometry_msgs()

# Wrapper for ROS messages for commonly
# used sensors
class Sensor_msgs():
    def __init__(self):
        self.name = ""

    def LaserScan(self):
        return "sensor_msgs/LaserScan"

    def Imu(self):
        return "sensor_msgs/Imu"

    def JointState(self):
        return "sensor_msgs/JointState"


sensor_msgs = Sensor_msgs()

# Wrapper for ROS messages for commonly
# used sensors
class Nav_msgs():
    def __init__(self):
        self.name = ""

    def Odometry(self):
        return "nav_msgs/Odometry"

    def Delivery(self):
        return "nav_msgs/Delivery"

nav_msgs = Nav_msgs()

# Wrapper for ROS messages for commonly
# used sensors
class Turtlebot3_msgs():
    def __init__(self):
        self.name = ""

    def VersionInfo(self):
        return "turtlebot3_msgs/VersionInfo"

    def SensorState(self):
        return "turtlebot3_msgs/SensorState"

turtlebot3_msgs = Turtlebot3_msgs()

# ROS publisher 
class PublisherWrap():
    def __init__(self, topic, msg_type,parentNode):
        self.topic = topic
        self.msg_type = msg_type()
        self.parent = parentNode
        self.NON_JSON = "std_msgs"
        
    #send the data over to the topic
    def publish(self, message):
        #send the message to the publishers for this topic
        try:
            if self.NON_JSON not in self.msg_type:
                r = requests.post(self.parent.hostname + self.topic,
                    params={
                        "{}".format(self.msg_type): json.dumps(message)
                    }
                )
            else:
                r = requests.post(self.parent.hostname + self.topic,
                    params={
                        "{}".format(self.msg_type): message
                    }
                )
            return r.json()[self.msg_type]
        except Exception as e:
            print("ERROR PROCESSING REQUEST: " + str(e))
            return False

# ROS subscriber to a topic
class SubscriberWrap():
    #initialization 
    #will start a thread for the callback fn
    def __init__(self, topic, msg_type, callback, parentNode, sr=0.01):
        self.topic = topic
        self.rate = sr
        self.msg_type = msg_type()
        self.parent = parentNode
        self.callback = callback
        self.isRegistered = True

        #create thread
        self.thread = threading.Thread(target=self.perform, args=())
        self.thread.daemon = True
        self.thread.start() #start the execution

    #update the read rate
    def updateRate(self, sr):
        self.rate = sr

    #perform the action while registered
    def perform(self):
        while self.isRegistered:
            #print(self.topic)
            try:
                r = requests.get(self.parent.hostname + self.topic)
                self.callback(r.json()[self.msg_type])
                time.sleep(self.rate)
            except Exception as e:
                print("ERROR PROCESSING REQUEST: " + str(e))
                time.sleep(self.rate)
                self.unregister()

    #unregister the subscription
    def unregister(self):
        self.isRegistered = False
    
# ROSPY library wrapper
class Rospy():
    

    #initialize the ROS instance to connect to the simulator
    #requires that the simulation program is already running
    def __init__(self):
        self.subscriptions = {}
        self.publishers = {}
        self.topics = {}
        self.node_name = ""
        self.hostname = "http://localhost:3000"

        try:
            #get the topics from the other node
            r = requests.get(self.hostname)
            self.topics = r.json()
        except:
            print("No simulation program found running!")
            sys.exit(1)

    # create a new publisher object and register with the node
    # @param topic
    # @param msgType
    def Publisher(self, topic, msgType):
        if topic in self.topics:
            print("WARNING: Topic %s is already defined"%topic)

        pub = PublisherWrap(topic, msgType, self)
        self.topics[topic] = msgType()
        return pub
        
    #create a new subscription object and register with the node
    # @param topic
    # @param msgType
    # @param callback
    def Subscriber(self, topic, msgType, callback):
        self.subscriptions[topic] = SubscriberWrap(topic, msgType, callback, self)
        return self.subscriptions[topic]

    # ROS sleep function for provided duration
    # @param time Integer seconds to sleep
    def sleep(self, duration):
        time.sleep(duration)

    #change the read rate of subscribers, in Hz
    def Rate(self, sr):
        for tp, ob in self.publishers.items():
            ob.updateRate(1/float(sr))

    # ROS function to view an array of 
    # arrays for all topics and msg types
    # @return [[str, str]]
    def get_published_topics(self):
        res = []
        for tp, ob in self.topics.items():
            res.append([tp, ob])
        return res
        
    # ROS function for initialization of the node
    def init_node(self, node_name):
        print("Node Started for %s"% node_name)
        self.node_name = node_name


rospy = Rospy()