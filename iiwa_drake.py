import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import meshcat
import meshcat.geometry as g

###----------------
from robot_plans import JointSpacePlan
from drake import lcmt_iiwa_status
from iiwa_plan_runner import IiwaPlanRunner
from plan_utils import *
from pydrake.examples.manipulation_station import (
  ManipulationStationHardwareInterface)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, AbstractValue
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer, \
  MeshcatContactVisualizer
from pydrake.systems.primitives import LogOutput
###----------------
from pydrake.multibody.tree import FrameIndex, BodyIndex
from pydrake.multibody.tree import ModelInstanceIndex
from ik_utils import InverseKinPointwise
from robot_plans import JointSpacePlan
from pydrake.trajectories import PiecewisePolynomial
### ------------------
from pydrake.systems.perception import PointCloudConcatenation
from pydrake.perception import DepthImageToPointCloud, BaseField
from pydrake.systems.sensors import PixelType
from pydrake.systems.meshcat_visualizer import (MeshcatVisualizer,
  MeshcatPointCloudVisualizer, AddTriad)
from pydrake.common import FindResourceOrThrow, GetDrakePath
from pydrake.perception import DepthImageToPointCloud, BaseField
from file_utils import LoadCameraConfigFile
from pydrake.systems.sensors import RgbdSensor, PixelType
from pydrake.geometry.render import (CameraProperties, DepthCameraProperties,
                   MakeRenderEngineVtk, RenderEngineVtkParams)
# use drake visualizer
from pydrake.geometry import ConnectDrakeVisualizer
from pydrake.lcm import DrakeLcm
# generate manipulands
from pydrake.examples.manipulation_station import CreateManipulationClassYcbObjectList
import glob
from pydrake.common.eigen_geometry import Quaternion, AngleAxis
from pydrake.multibody.math import SpatialVelocity
###----------------
from IPython import embed
import pdb
import copy
# rotation matrix interpolation
from scipy.spatial.transform import Rotation as scipyR
from scipy.spatial.transform import Slerp
from scipy.stats import norm


def random_quaternion():
  """
  use scipy, or alternatively
  generate a random quaternion by 
  https://stackoverflow.com/questions/31600717/how-to-generate-a-random-quaternion-quickly
  definition of the 4 resultant numbers are assumed not matter (TODO need verify)
  """ 
  r = scipyR.random()
  return r.as_quat()

def QuaternionToRotationMatrix(quat : np.array):
  """
  use pydrake's internal function to 
  convert quaternion to rotation matrix
  """
  qt = Quaternion(np.asarray(quat))
  return RotationMatrix(qt.rotation())

def random_rigid_transform(p=[0.46, 0, 0.3]):
  """
  place an object directly under gripper, but randomize object shape and orientation
  pose of the object relative to gripper is fixed
  """
  ### define fixed location 
  loc = np.asarray(p)
  ### random orientation
  rotation_matrix = QuaternionToRotationMatrix(random_quaternion())
  rand_rf = RigidTransform(R=rotation_matrix, p=loc)
  return rand_rf

def init_noisy_manipuland_pose():
  """
  slightly vary/perturb the init pose of the manipuland 
  from the given pose
  """
  ## predefined tranform with noise
  noisy_rf = np.eye(4)
  noisy_rf[0:3, 0:3] = np.array([[0, 1, 0],  [1, 0, 0], [0, 0, -1]], dtype=float).T
  noisy_rf[0:3, 3] = [0.5, 0, 0.3] # init from air and drop down to bin
  # add noise on orientation
  # for soup can case, rotate about z axis for a random angle
  noisy_ax = AngleAxis(angle=norm(0, np.pi/4).rvs(), axis=[0,0,1])
  noisy_rf[0:3, 0:3] = np.dot(noisy_rf[0:3, 0:3], noisy_ax.rotation())
  rand_rf = RigidTransform(noisy_rf)
  return rand_rf

class ManipulationStationSimulator:
  def __init__(self, X_WObject=None, 
         mode='class', 
         use_palm_cam = False, 
         use_tactile_cam = False,
         visualizer='meshcat', 
         vis_contact = False):
    """
    args:
    --- input ---
    visualizer = 'drake', 'meshcat', None
    vis_contact: boolean for whether to draw contact vectors
    tips for drake visualizer:
    1. need to launch in advance
    2. shift + mouse = transition, ctrl + mouse = in/out, mouse = rotation
    Todo: 
    4. add workspace boundary calc by IK, object out of reach are considered failure
    6. breakdown the grasping into pre-grasp planning stage (push, grasp) 
       and the post-grasp stage (early release, slowly retract)
       simplify the game to test the possibility
       (1) only close the loop at post-grasp stage?
       (2) trigger "wandering mode" before grasp, allow learning of local pushing 
    7. get height map
    8. manually take a look of the palm cam info when pushing 
    heightmap --> action --> ... --> liftup while monitoring
    future add push while monitoring as well
    """
    # Finalize manipulation station by adding manipuland.
    self.mode = mode
    self.visualizer = visualizer
    self.vis_contact = vis_contact
    # an instance of pydrake.examples.manipulation_station import ManipulationStation
    self.station = ManipulationStation()
    if mode == 'clutter':
      self.station.SetupClutterClearingStation()
    else:
      self.station.SetupManipulationClassStation()

    self.frames_to_draw = {"iiwa": {"iiwa_link_7"}}
    # self.frames_to_draw = {"gripper": {"body"}}
    # self.frames_to_draw = {"gripper": {"right_finger"}}
    self.default_render_name = "manip_station_renderer" ## hardcoded in associated cpp file
    # self.add_single_rand_manipuland()
    ## add wall
    # tmp_T = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0.5, 0.0, 0.25, 1]]).T
    # self.wall_rf0 = RigidTransform(tmp_T)
    # self.station.AddManipulandFromFile('drake/examples/manipulation_station/models/plate_wall.sdf', self.wall_rf0)
    ## add plate
    tmp_T = np.array([[1,0,0,0], [0,0,-1,0], [0,1,0,0], [0.5, -0.2, 0.1, 1]]).T
    self.plate_rf0 = RigidTransform(tmp_T)

    # self.station.AddManipulandFromFile('drake/examples/manipulation_station/models/simple_cylinder.sdf', self.plate_rf0)

    #### camera related
    self.plant = self.station.get_mutable_multibody_plant()
    self.simulator = None
    self.plan_runner = None
    self.nq = 7
    self.camera_name_list = self.station.get_camera_names()
    self.get_frame_names()
    self.get_model_instances()
    self.world_frame = self.plant.world_frame()
    self.l7_frame = self.plant.GetFrameByName("iiwa_link_7")
    self.left_finger_frame = self.plant.GetFrameByName('left_finger')
    self.right_finger_frame = self.plant.GetFrameByName('right_finger')

    # note on rgbd sensor registration
    # existing camera (particularly the default ones) can be 
    # overwritten by new camera(s) with the same name
    if use_palm_cam: ## register palm camera
      # plam camera instrinsics
      width = 480
      height = 360
      fov_y = np.pi / 2
      focal_y = height / 2 / np.tan(fov_y / 2)
      focal_x = focal_y
      center_x = width / 2 - 0.5
      center_y = height / 2 - 0.5
      intrinsic_matrix = np.array([
        [focal_x, 0, center_x],
        [0, focal_y, center_y],
        [0, 0, 1]])
      # Depth camera properties
      self.cam_properties_palm = DepthCameraProperties(
        width=width, height=height, fov_y=fov_y,
        renderer_name=self.default_render_name, z_near=0.05, z_far=0.1)
      # relative transform
      self.gripper_body_frame = self.plant.GetFrameByName('body')
      self.gripper_body_R_palm_cam = RotationMatrix(np.array([[1,0,0], [0,0,-1], [0,1,0]]).T)
      self.gripper_body_RT_palm_cam = RigidTransform(R=self.gripper_body_R_palm_cam, p=np.array([0.0, 0.045, 0.0]))
      self.station.RegisterRgbdSensor('palm', self.gripper_body_frame,\
                      self.gripper_body_RT_palm_cam, \
                      self.cam_properties_palm)
    if use_tactile_cam: ## register tactile sensor
      # plam camera instrinsics
      width = 480
      height = 360
      fov_y = np.pi / 2
      focal_y = height / 2 / np.tan(fov_y / 2)
      focal_x = focal_y
      center_x = width / 2 - 0.5
      center_y = height / 2 - 0.5
      intrinsic_matrix = np.array([
        [focal_x, 0, center_x],
        [0, focal_y, center_y],
        [0, 0, 1]])
      # Depth camera properties
      self.cam_properties_tactile = DepthCameraProperties(
        width=width, height=height, fov_y=fov_y,
        renderer_name=self.default_render_name, z_near=0.01, z_far=0.02) ## very short furthest depth
      self.gripper_left_finger_frame = self.plant.GetFrameByName("left_finger")
      self.gripper_right_finger_frame = self.plant.GetFrameByName("right_finger")
      # add left finger tactile
      self.gripper_left_finger_R_tactile = RotationMatrix(np.array([[0,0,-1], [0,1,0], [1,0,0]]).T)
      self.gripper_left_finger_RT_tact = RigidTransform(R=self.gripper_left_finger_R_tactile, p=np.array([0.01, 0.02, 0.0]))
      self.station.RegisterRgbdSensor('left_tactile', self.gripper_left_finger_frame,\
                      self.gripper_left_finger_RT_tact, \
                      self.cam_properties_tactile)
      # add right finger tactile
      self.gripper_right_finger_R_tactile = RotationMatrix(np.array([[0,0,1], [0,1,0], [-1,0,0]]).T)
      self.gripper_right_finger_RT_tact = RigidTransform(R=self.gripper_right_finger_R_tactile, p=np.array([-0.01, 0.02, 0.0]))
      self.station.RegisterRgbdSensor('right_tactile', self.gripper_right_finger_frame,\
                      self.gripper_right_finger_RT_tact, \
                      self.cam_properties_tactile)
    ### build diagram
    self.__build_diagram__()

  def get_accessible_manipulands(self):
    """
    read availale manipulands in pydrake/share directory
    models are placed under
    drake/lib/../share/drake/manipulation/models/ycb/sdf
    """
    accessible_manipulands = []
    drake_path = GetDrakePath()
    model_folder = os.path.join(drake_path, 'manipulation/models/ycb/sdf')
    sdf_files = glob.glob(model_folder + '/*.sdf')
    for it in sdf_files:
      crop_index = it.find('share/')
      accessible_manipulands.append(it[crop_index+6:])
    return accessible_manipulands

  def add_cluttered_manipulands(self):
    self.manipulands = CreateManipulationClassYcbObjectList()
    for path, pose in self.manipulands:
      self.station.AddManipulandFromFile(path, pose)

  def add_single_rand_manipuland(self, sdf_path=None, add_bin = True):
    """
    randomize object selection as well
    """
    if add_bin:
      bin_path = 'drake/examples/manipulation_station/models/bin.sdf'
      bin_loc = [0.5, 0, 0.1]
      bin_rf = RigidTransform(p=np.array(bin_loc))
      self.station.AddManipulandFromFile(bin_path, bin_rf)

    sdf_path = 'drake/manipulation/models/ycb/sdf/005_tomato_soup_can.sdf'
    if sdf_path is None:
      accessible_manipulands = self.get_accessible_manipulands()
      sdf_path = np.random.choice(accessible_manipulands)
    ## complete random transform
    # rand_rf = random_rigid_transform()
    ## predefined tranform with noise
    rand_rf = init_noisy_manipuland_pose()
    self.station.AddManipulandFromFile(sdf_path, rand_rf)

  def __build_diagram__(self):
    """
    Constructs a Diagram that sends commands to ManipulationStation.
    @param plan_list: A list of Plans to be executed.
    @param gripper_setpoint_list: A list of gripper setpoints. Each setpoint corresponds to a Plan.
    @param extra_time: the amount of time for which commands are sent,
      in addition to the sum of the durations of all plans.
    """
    ### finalize call its internal builder to build station's internal diagram
    self.station.Finalize()

    builder = DiagramBuilder()
    builder.AddSystem(self.station)

    # Add plan runner.
    ### use an empty lists to initialize IiwaPlanRunner,
    ### send plans later to re-use IiwaPlanRunner
    ### otherwise use default settings
    plan_runner = IiwaPlanRunner(
      iiwa_plans=[],  
      gripper_setpoint_list=[])
    self.plan_runner = plan_runner

    ### hook up ports and loggers
    ### these logs accumulate automatically, one second includes
    ### ~600 simulation steps
    self.ConnectPortsAndAddLoggers(builder, self.station, plan_runner)

    plant_state_log = LogOutput(
      self.station.GetOutputPort("plant_continuous_state"), builder)
    plant_state_log.DeclarePeriodicPublish(0.01)
    self.plant_state_log = plant_state_log

    scene_graph = self.station.get_mutable_scene_graph()
    self.scene_graph = scene_graph
    if not (self.visualizer is None):
      # Add meshcat visualizer
      if self.visualizer == 'meshcat':
        self.meshcat_vis = meshcat.Visualizer()
        viz = MeshcatVisualizer(scene_graph,
                    zmq_url="tcp://127.0.0.1:6000",
                    frames_to_draw=self.frames_to_draw,
                    frames_opacity=0.8)
        self.viz = builder.AddSystem(viz)
        builder.Connect(self.station.GetOutputPort("pose_bundle"),
                viz.GetInputPort("lcm_visualization"))

      # Add drake visualizer
      if self.visualizer == 'drake':
        lcm = DrakeLcm()
        lcm_publisher_sys = ConnectDrakeVisualizer(builder=builder, \
                scene_graph=scene_graph, \
                pose_bundle_output_port=self.station.GetOutputPort("pose_bundle"), \
                lcm = lcm)
        self.drake_vis_lcm = lcm
      if self.vis_contact:
      ### visualize contact results
        contact_viz = MeshcatContactVisualizer(
          meshcat_viz=viz, plant=self.plant)
        builder.AddSystem(contact_viz)
        builder.Connect(
          self.station.GetOutputPort("pose_bundle"),
          contact_viz.GetInputPort("pose_bundle"))
        builder.Connect(
          self.station.GetOutputPort("contact_results"),
          contact_viz.GetInputPort("contact_results"))
    
    # build diagram
    self.diagram = builder.Build()

  def reset_single_object_env(self, pose:RigidTransform=None):
    """
    reset the object pose in single object learning world
    """
    if pose is None:
      pose = init_noisy_manipuland_pose()
    self.set_obj_pose_vel(pose)
    self.sim_duration(5.0)

  def get_body_by_frame_name(self, frame_name:str):
    obj_frame = self.plant.GetFrameByName(frame_name)
    obj_body = self.plant.GetBodyByName(obj_frame.body().name())
    return obj_body

  def set_obj_pose_vel(self, pose: RigidTransform, vel=None):
    """
    hardset object pose and velocity
    """
    obj_frame_name = 'plate_base'
    obj_body = self.get_body_by_frame_name(obj_frame_name)
    if vel is None:
      zero_vel = SpatialVelocity.Zero()
      self.plant.SetFreeBodySpatialVelocity(body=obj_body, V_WB=zero_vel, context=self.plant_context)
    self.plant.SetFreeBodyPose(context=self.plant_context, body=obj_body, X_WB=pose)

  def get_body_vel_in_world(self, frame_name:str):
    """
    convenient wrapper to get body velocity in world
    """
    body = self.get_body_by_frame_name(frame_name)
    vel = self.plant.EvalBodySpatialVelocityInWorld(
      context = self.plant_context,
      body = body
      )
    # vel_trans = vel.translational()
    # vel_rot = vel.rotational()
    vel_all = vel.get_coeffs() # first 3 rot, last 3 trans
    return vel_all

  def init_simulator(self, real_time_rate=0):
    """
    @param real_time_rate: 1.0 means realtime; 0 means as fast as possible.
    @param q0_iiwa: initial configuration of the robot.
    @param is_visualizing: if true, adds MeshcatVisualizer to the Diagram. It should be set to False
      when running tests.
    @param sim_duration: the duration of simulation in seconds. If unset, it is set to the sum of the durations of all
      plans in plan_list plus extra_time.
    @return: logs of robot configuration and MultibodyPlant, generated by simulation.
      Logs are SignalLogger systems, whose data can be accessed by SignalLogger.data().

    """
    # construct simulator
    simulator = Simulator(self.diagram)
    self.simulator = simulator

    context = self.diagram.GetMutableSubsystemContext(
      self.station, simulator.get_mutable_context())
    self.station_context = context

    # set initial state of the robot
    # initial joint angles of the robot
    # q0_iiwa = np.array([0., 0.6, 0., -1.75, 0., 1., 0.])
    # q0_iiwa = np.array([0.0,  0.5656,  0.0, -1.8045, 0.0,  0.7856,  0.0])
    q0_iiwa = np.array([ 1.40666193e-05,  1.56461165e-01, -3.82761069e-05, -1.32296976e+00, -6.29097287e-06,  1.61181157e+00, -2.66900985e-05])
    self.station.SetIiwaPosition(context, q0_iiwa)
    self.station.SetIiwaVelocity(context, np.zeros(7))
    self.station.SetWsgPosition(context, 0.05)
    self.station.SetWsgVelocity(context, 0)
    # set initial hinge angles of the cupboard.
    # setting hinge angle to exactly 0 or 90 degrees will result in
    # intermittent contact with small contact forces between the door and
    # the cupboard body.
    if self.mode == 'class':
      door_angle = np.pi/2.0*3.0
      left_hinge_joint = self.plant.GetJointByName("left_door_hinge")
      left_hinge_joint.set_angle(
        context=self.station.GetMutableSubsystemContext(
          self.plant, context),
        angle=-door_angle)

      right_hinge_joint = self.plant.GetJointByName("right_door_hinge")
      right_hinge_joint.set_angle(
        context=self.station.GetMutableSubsystemContext(self.plant,
                                context),
        angle=door_angle)

    simulator.set_publish_every_time_step(False)

    simulator.set_target_realtime_rate(real_time_rate)
    # calculate starting time for all plans.
    # t_plan = GetPlanStartingTimes(plan_list)
    # sim_duration = t_plan[-1] + extra_time
    # if is_visualizing:
    #     print("simulation duration", sim_duration)
    #     print("plan starting times\n", t_plan)

    self.SetInitialPlanRunnerState(self.plan_runner, simulator, self.diagram)
    simulator.Initialize()
    # clone the init simulator context for reset
    self.simulator_init_context = self.simulator.get_mutable_context().Clone()
    self.plant_context = self.diagram.GetMutableSubsystemContext(
      self.plant, self.simulator.get_mutable_context())

    ### get and visualize camera poses in world
    self.static_camera_pose_in_world = {}
    for cam_name in self.camera_name_list: 
      self.static_camera_pose_in_world[cam_name] = self.station.GetStaticCameraPosesInWorld()[cam_name].matrix()

    ### for camera image collection
    self.camera_time = 0.0
    self.pose_home = self.get_arm_pose()
    self.reset_recorded_sensor_port()
    self.reset_recorded_pose_data()
    self.reset_event_history()
    self.reset_recorded_vel_data()

  def reset_recorded_sensor_port(self, sensor_port_names:list=None):
    """
    set sensor ports to be recorded
    list of strings, port names can be read from the system diagram
    """
    if sensor_port_names is None:
      self.sensor_port_names = [
        'camera_0_rgb_image', \
        # 'camera_0_depth_image', \
        # 'camera_1_rgb_image', \
        # 'camera_1_depth_image', \
        # 'camera_2_rgb_image', \
        # 'camera_2_depth_image', \
        # 'camera_palm_rgb_image', \
        # 'camera_palm_depth_image', \
        # 'camera_left_tactile_rgb_image', \
        # 'camera_left_tactile_depth_image', \
        # 'camera_right_tactile_rgb_image', \
        # 'camera_right_tactile_depth_image', \
        'wsg_state_measured',
        'wsg_force_measured',
        'iiwa_torque_measured',
        'iiwa_position_measured',
        'iiwa_velocity_estimated',
        'iiwa_torque_external',
        # 'geometry_poses',
        # 'plant_continuous_state'
        ]
    else:
      self.sensor_port_names = sensor_port_names
    # initialize the sensor data container
    self.sensor_data = {}
    for sensor_name in self.sensor_port_names:
      self.sensor_data[sensor_name] = []
    self.sensor_data['time'] = []

  def reset_recorded_pose_data(self, pose_names:list=None):
    """
    same as the sensor data recording
    """
    self.pose_data = {}
    if pose_names is None:
      self.pose_names = [
        'plate_base',
        'iiwa_link_7'
      ]
    else:
      self.pose_names = pose_names
    for pose_frame in self.pose_names:
      self.pose_data[pose_frame] = []
    self.pose_data['time'] = []

  def reset_recorded_vel_data(self, frame_names:list=None):
    """
    
    """
    self.vel_data = {}
    if frame_names is None:
      self.vel_names = [
      'plate_base',
      'iiwa_link_7',
      ]
    else:
      self.vel_names = frame_names
    for obj in self.vel_names:
      self.vel_data[obj] = []
    self.vel_data['time'] = []

  def append_event(self, name:str='generic', has_data:bool=False):
    """
    record a dispatched event
    """
    self.event_history['action'].append(name)
    self.event_history['time'].append(
      self.station_context.get_time()
      )
    self.event_history['has_data'].append(has_data)

  def reset_event_history(self):
    """
    reset the dispatched event record
    """
    # dispatch time for each notable action
    self.event_history = {
    'action': [],
    'time': [],
    'has_data': []
    }
  
  def draw_frame(self, frame_name: str):
    """
    draw frame if exist
    frame_name cannot be empty string
    """
    if isinstance(frame_name, str) and len(frame_name) > 0 and frame_name in self.frame_names:
      tmp_frame = self.plant.GetFrameByName(frame_name)
      world_T_frame = self.get_relative_frame_transform(self.world_frame, tmp_frame).matrix()
      self.draw_pose_in_world(world_T_frame, prefix='', name=frame_name)
    else:
      print('error: {} does not exist'.format(frame_name))

  def draw_pose_in_world(self, transform:np.array, prefix = 'tests', name = 'test', length=0.15, radius=0.006):
    """
    draw a transform in 3D space
    if the same prefix and name is used in existing visualization frame, the frame
    will be removed and replaced by the frame defined by the current call

    arg
    transform: a numpy array or matrix of size 4x4
    """
    if self.visualizer == 'meshcat':
      AddTriad(self.viz.vis, name=name, prefix=prefix, length=length, radius=radius)
      self.viz.vis[prefix][name].set_transform(transform)
    else:
      print("drake visualiser does not support drawing axes")

  def interact(self, **args):
    """ 
    enter an interactive session
    """
    print('entered iiwa drake interactive embed session')
    embed()
    print('left iiwa drake interactive embed session')

  def get_frame_names(self):
    self.frame_names = []
    for i in range(self.plant.num_frames()):
      self.frame_names.append(self.plant.get_frame(FrameIndex(i)).name())
    return self.frame_names

  def get_body_frameid(self):
    self.bodyframeid = []
    for i in range(self.plant.num_bodies()):
      self.bodyframeid.append(self.plant.GetBodyFrameIdIfExists(BodyIndex(i)))

  def get_model_instances(self):
    ### check model instances
    self.model_names = []
    for i in range(self.plant.num_model_instances()):
      name = self.plant.GetModelInstanceName(ModelInstanceIndex(i))
      if name == 'gripper':
        self.model_instance_gripper_index = int(i)
      self.model_names.append(name)
    self.model_instances = []
    for model_name in self.model_names:
      self.model_instances.append(self.plant.GetModelInstanceByName(model_name))
    self.model_instances_gripper = self.model_instances[self.model_instance_gripper_index]
    self.body_indices_gripper = self.plant.GetBodyIndices(self.model_instances_gripper)
    self.frameid_gripper = self.plant.GetBodyFrameIdIfExists(self.body_indices_gripper[0])
    return self.model_names, self.model_instances

  def get_relative_frame_transform(self, frame_A, frame_B):
    """
    frames are pydrake.multibody.tree.BodyFrame_[float]
    Seems like frame_A is the parent frame
    Seems like frame_B is the child frame
    ----
    return 4x4 np.array
    """
    return self.plant.CalcRelativeTransform(
            self.plant_context,
            frame_A=frame_A,
            frame_B=frame_B)

  def sim_to_time(self, stop_time):
    """
    sim until an absolute time point
    """
    ### advance to sim to an absolute time stamp
    self.simulator.AdvanceTo(stop_time)

  def append_sensor_data(self):
    """
    append sensor data for analysis
    """
    self.sensor_data['time'].append(self.station_context.get_time())
    for sensor_name in self.sensor_port_names:
      if 'rgb' in sensor_name:
        self.sensor_data[sensor_name].append(copy.deepcopy(self.eval_station_port_value(sensor_name).data[:,:,0:3]))
      elif 'depth' in sensor_name:
        self.sensor_data[sensor_name].append(copy.deepcopy(self.eval_station_port_value(sensor_name).data[:,:,0]))
      else:
        self.sensor_data[sensor_name].append(copy.deepcopy(self.eval_station_port_value(sensor_name)))

  def get_sensor_data(self):
    """
    get sensor data of the current time step, useful for RL training
    """
    sensor_data = {}
    for sensor_name in self.sensor_port_names:
      if 'rgb' in sensor_name:
        sensor_data[sensor_name].append(copy.deepcopy(self.eval_station_port_value(sensor_name).data[:,:,0:3]))
      elif 'depth' in sensor_name:
        sensor_data[sensor_name].append(copy.deepcopy(self.eval_station_port_value(sensor_name).data[:,:,0]))
      else:
        sensor_data[sensor_name].append(copy.deepcopy(self.eval_station_port_value(sensor_name)))
    return sensor_data

  def append_pose_data(self, parent_frame:str='WorldBody'):
    """
    append pose of a selected frame
    """
    frameA = self.plant.GetFrameByName(parent_frame)
    self.pose_data['time'].append(self.station_context.get_time())
    for pose_frame in self.pose_names:
      frameB = self.plant.GetFrameByName(pose_frame)
      pose = self.get_relative_frame_transform(frameA, frameB).matrix()
      self.pose_data[pose_frame].append(pose)

  def append_vel_data(self):
    """
    append velocity in world frame, (only available in world frame)
    """
    self.vel_data['time'].append(self.station_context.get_time())
    for vel_name in self.vel_names:
      vel = self.get_body_vel_in_world(vel_name)
      self.vel_data[vel_name].append(vel)

  def sim_duration(self, duration, sensor_dt = None):
    """
    simulate through the duration
    sensor_dt: if not None, will record sensor data
    """
    stop_time = self.station_context.get_time() + duration
    if sensor_dt is None:
      self.simulator.AdvanceTo(stop_time)
    else:
      intermediate_times = np.arange(self.station_context.get_time(), stop_time, step=sensor_dt, dtype=float)
      for cur_t in intermediate_times:
        self.simulator.AdvanceTo(cur_t)
        # camera time are tracked separately to avoid 
        # repeated frames
        if self.camera_time < cur_t:
          # print("image taken at t = {}".format(cur_t))
          self.camera_time = cur_t
          self.append_sensor_data()
          self.append_pose_data()
          self.append_vel_data()

  def eval_station_port_value(self, port_name):
    tmpport = self.station.GetOutputPort(port_name)
    return tmpport.Eval(self.station_context)

  def get_arm_pose(self):
    return copy.deepcopy(self.get_relative_frame_transform(self.world_frame, self.l7_frame).matrix())

  def get_gripper_distance(self):
    """
    return the distance as a scalar
    """
    distance = self.get_relative_frame_transform(self.left_finger_frame, self.right_finger_frame).translation()[0]
    return distance

  def plan_to_T(self, world_T_goal:np.array, duration=3.0, num_knot_points=3.0):
    """
    plan to a 4x4 transform in world frame
    input:
    world_T_goal should be a numpy array/matrix of size 4x4
    """
    pos_start = self.get_arm_pose()[0:3, 3]
    pos_end = world_T_goal[0:3 ,3]
    R_start = self.get_arm_pose()[0:3, 0:3]
    R_goal = world_T_goal[0:3, 0:3]
    q0 = self.eval_station_port_value('iiwa_position_measured')
    try:
      poly, q_knots = InverseKinPointwise(GetInterploatePosition(pos_start, pos_end), \
                        GetInterpolateOrientation(R_start, R_goal), q0, \
                        duration=duration, num_knot_points=3 )
      return JointSpacePlan(trajectory=poly)
    except:
      print('Drake sim warning: IK failed at T = {}'.format(world_T_goal))
      return None

  def move_to_T(self, world_T_goal:np.array, duration=3.0, num_knot_points=3.0, sensor_dt = None, event_name:str=None):
    if event_name is not None:
      self.append_event(event_name, has_data = sensor_dt is not None)
    plan = self.plan_to_T(world_T_goal, duration=3.0, num_knot_points=3.0)
    if plan is not None:
      self.plan_runner.kuka_plans_list.append(plan)
      self.sim_duration(plan.duration + 1.0, sensor_dt)
      return True
    return False

  def tilt_gripper(self, angle, axis, duration=10.0, num_knot_points=10, sensor_dt=None):
    """
    convenient function for tilting the gripper
    E.g.
    angle = -np.pi
    axis = [1,0,0] # local frame
    """
    Tnow = self.get_arm_pose()
    rot = AngleAxis(angle=angle, axis=axis)
    Tnow[0:3, 0:3] = np.dot(Tnow[0:3, 0:3], rot.rotation())
    self.move_to_T(Tnow, duration=duration, num_knot_points=num_knot_points, sensor_dt=sensor_dt)

  def pickup_plate(self, 
    frame_name = None, 
    vertical_offset=0.25, 
    duration=2.0,
    num_knot_points=10,
    sensor_dt=None):

    ### assist data labeling
    self.cur_label = None

    # return to home position, get ready for object placement
    self.move_to_T(self.pose_home, duration=2.0, num_knot_points=10.0)

    self.reset_single_object_env(self.plate_rf0)
    self.open_gripper()
    # get object location
    if frame_name is None:
      frame_name = 'plate_base'
    obj_frame = self.plant.GetFrameByName(frame_name)
    world_T_obj = self.get_relative_frame_transform(self.world_frame, obj_frame).matrix()
    fixed_grasp_pose = True
    if fixed_grasp_pose:
      T_goal = np.eye(4)
      T_goal[0:3, 0:3] = np.array([[-1,0,0], [0,1,0], [0,0,-1]]).T
      T_goal[0:3, 3] = world_T_obj[0:3, 3]
      T_goal[2, 3] += vertical_offset + 0.1
      self.move_to_T(T_goal, duration=duration, sensor_dt=sensor_dt) # r
      # self.draw_pose_in_world(T_goal)
      T_goal[2, 3] -=  0.1
      self.move_to_T(T_goal, duration=duration, sensor_dt = sensor_dt) # ready to grasp
      self.close_gripper(sensor_dt=sensor_dt) # close fingers
      T_goal[2, 3] += 0.1
      # self.draw_pose_in_world(T_goal)
      self.move_to_T(T_goal, duration=duration, sensor_dt=sensor_dt) # lift up
  
  def place_plate(self, 
    frame_name = None, 
    vertical_offset=0.25, 
    duration=10.0, 
    randomize=True,
    tilt_angle=-np.pi/6-np.pi/4, 
    tilt_duration=10.0, 
    push_distance=0.05, 
    num_knot_points=10,
    sensor_dt=None):
    """
    These labels are time sensitive as well. Need to take that into account
    tilt_angle < -1.5, sticking and flipping (potential wrist ground collision)
    tilt_angle ~ -1.4 to -1.2, good placing
    tilt_angle > -1.2, object hard collision with the ground, gripper may bounce off while retracting 
    push_distance > 0.05, hard push, bounce
    """
    tilt_angle = tilt_angle
    push_distance = push_distance
    if randomize:
      tilt_angle = norm.rvs(loc=-np.pi/6-np.pi/4, scale=np.pi/6)
      push_distance = norm.rvs(loc=0.05, scale=0.02)

    # print('tilt angle = {}'.format(tilt_angle))
    # print('push_distance = {}'.format(push_distance))
    # tilt gripper
    self.tilt_gripper(tilt_angle, [1,0,0], duration=tilt_duration, num_knot_points=num_knot_points, sensor_dt=sensor_dt)
    # lower the gripper close to the ground
    Tnow = self.get_arm_pose()
    Tnow[2, 3] -= 0.5
    self.move_to_T(Tnow, duration=10.0, num_knot_points=10, sensor_dt=sensor_dt)
    ###################################################################
    # start of data collection
    # lower the gripper a bit further for gentle placing
    Tnow = self.get_arm_pose()
    Tnow[2, 3] -= push_distance
    self.move_to_T(Tnow, duration=10.0, num_knot_points=10, sensor_dt=sensor_dt)
    # gently place the plate
    self.open_gripper(sensor_dt=sensor_dt)
    # retract the gripper sideway
    Tnow = self.get_arm_pose()
    Tnow[1, 3] -= 0.1
    self.move_to_T(Tnow, duration=10.0, num_knot_points=10, sensor_dt=sensor_dt)

  def openloop_grasp(self, frame_name = None, vertical_offset=0.2, duration=10.0):
    """
    single object grasping trial, fixed gripper orientation
    fixed grasp pose in object frame
    add a tray to constraint object from moving out of workspace
    """
    self.reset_single_object_env()
    # get object location
    if frame_name is None:
      frame_name = 'plate_base'
    obj_frame = self.plant.GetFrameByName(frame_name)
    world_T_obj = r
    fixed_grasp_pose = True
    if fixed_grasp_pose:
      T_goal = np.eye(4)
      T_goal[0:3, 0:3] = np.array([[-1,0,0], [0,1,0], [0,0,-1]]).T
      T_goal[0:3, 3] = world_T_obj[0:3, 3]
      T_goal[2, 3] += vertical_offset
      self.open_gripper()
      # self.draw_pose_in_world(T_goal)
      self.move_to_T(T_goal, duration=duration) # ready to grasp
      self.close_gripper() # close fingers
      T_goal[2, 3] += 0.3
      # self.draw_pose_in_world(T_goal)
      self.move_to_T(T_goal, duration=duration) # lift up
      self.open_gripper()


  def moveto_gripper_distance(self, dist:float, duration=5.0, sensor_dt=None):
    """
    move to desired gripper distance
    open ==> 0.1
    closed ==> 0.01
    """
    ### need add stationary/dummy arm traj for gripper action
    ### to be executed correctly
    q_knots = np.zeros((2, 7))
    q0 = self.eval_station_port_value('iiwa_position_measured')
    q_knots[0] = q0
    q_knots[1] = q0
    qtraj_still = PiecewisePolynomial.ZeroOrderHold([0, 1], q_knots.T)
    # gripper_setpoint_list = [0.1, 0.01]
    self.plan_runner.kuka_plans_list.append(JointSpacePlan(qtraj_still))


    self.plan_runner.gripper_setpoint_list.append(dist)
    self.sim_duration(duration, sensor_dt=sensor_dt)
    return

  def close_gripper(self, duration=3.0, sensor_dt = None, event_name:str=None):
    """
    close finger
    """
    if event_name is not None:
      self.append_event(event_name, has_data = sensor_dt is not None)
    self.moveto_gripper_distance(0.01, duration=duration, sensor_dt = sensor_dt)
    return

  def open_gripper(self, duration=3.0, sensor_dt = None, event_name:str=None):
    if event_name is not None:
      self.append_event(event_name, has_data = sensor_dt is not None)
    self.moveto_gripper_distance(0.07, duration=duration, sensor_dt = sensor_dt)

  def incremental_move(self, increment=0.1, direction = 'forth', duration=0.5, topdown=False):
    """
    all poses are in world frame
    direction: angle from the +x axis (ccw), angle = 0-360 deg
    """
    T_goal = self.get_arm_pose()
    if topdown:
      T_goal[0:3, 0:3] = np.array([[-1,0,0], [0,1,0], [0,0,-1]]).T
    if direction == 'forth':
      T_goal[0:3, 3] -= increment * T_goal[0:3, 0]
    elif direction == 'back':
      T_goal[0:3, 3] += increment * T_goal[0:3, 0]
    elif direction == 'left':
      T_goal[0:3 ,3] += increment * T_goal[0:3, 1]
    elif direction == 'right':
      T_goal[0:3 ,3] -= increment * T_goal[0:3, 1]
    elif direction == 'down':
      T_goal[0:3 ,3] += increment * T_goal[0:3, 2]
    elif direction == 'up':
      T_goal[0:3 ,3] -= increment * T_goal[0:3, 2]
    ## add rotation
    return self.move_to_T(T_goal, duration=duration, num_knot_points=2.0)

  def incremental_move_in_world(self, increment=0.1, direction = 'z', duration=0.5, topdown=False, sensor_dt=None, event_name=False):
    """
    all poses are in world frame
    direction: angle from the +x axis (ccw), angle = 0-360 deg
    """
    if event_name is not None:
      self.append_event(event_name, has_data = sensor_dt is not None)
    T_goal = self.get_arm_pose()
    if topdown:
      T_goal[0:3, 0:3] = np.array([[-1,0,0], [0,1,0], [0,0,-1]]).T
    if direction == 'x':
      T_goal[0, 3] += increment 
    elif direction == 'y':
      T_goal[1 ,3] += increment
    else:
      T_goal[2 ,3] += increment
    ## add rotation
    return self.move_to_T(T_goal, duration=duration, num_knot_points=2.0, sensor_dt=sensor_dt)

  def incremental_rotation(self, increment=5, duration=0.25):
    """
    topdown incremental rotation
    """
    c = np.cos(float(increment)/180.*np.pi)
    s = np.sin(float(increment)/180.*np.pi)
    rotation_matrix = np.eye(4)
    rotation_matrix[0:3, 0:3] = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1] ]).T

    T_goal = self.get_arm_pose()
    T_goal = np.dot(T_goal, rotation_matrix)

    return self.move_to_T(T_goal, duration=duration, num_knot_points=2.0)

  def keyboard_controller(self):
    """
    simple keyboard controller interface
    """
    while True:
      command = input('a/s/d/w/g/h/z/x/q (quit) \n')
      try:
        if command == 'a':
          self.incremental_move(direction='left')
        elif command == 's':
          self.incremental_move(direction='back')
        elif command == 'd':
          self.incremental_move(direction='right')
        elif command == 'w':
          self.incremental_move(direction='forth')
        elif command == 'z':
          self.incremental_move(direction='down')
        elif command == 'x':
          self.incremental_move(direction='up')
        elif command == 'j': # ccw rotation
          self.incremental_rotation(increment=5.0)
        elif command == 'k':
          self.incremental_rotation(increment=-5.0)
      except:
        print('ik failed')
      if command == 'g':
        self.close_gripper()
      elif command == 'h':
        self.open_gripper()
      if command == 'exit':
        break
  
  def clear_data_record(self):
    """
    clear the stored data, typically
    after having saved it to file
    """ 
    for key in self.pose_data:
      self.pose_data[key] = []
    for key in self.sensor_data:
      self.sensor_data[key] = []
    for key in self.vel_data:
      self.vel_data[key] = []
    self.reset_event_history()

  def save_data_to_file(self, fdir:str=None):
    if fdir is None:
      fdir = os.getcwd()
    # save pickle
    pickle.dump(self.sensor_data, file=open(os.path.join(fdir, 'sensor_data.p'), 'wb') )
    pickle.dump(self.pose_data, file=open(os.path.join(fdir, 'pose_data.p'), 'wb') )
    pickle.dump(self.event_history, file=open(os.path.join(fdir, 'dispatch_history.p'), 'wb'))

  def save_cam_images(self, fdir:str=None, cam_name='camera_0_rgb_image'):
    if fdir is None:
      fdir = os.getcwd()
    for t, img in enumerate(self.sensor_data[cam_name]):
      plt.imsave(os.path.join(fdir,'frame_{}.png'.format(t)), img)

  @staticmethod
  def GetCurrentJointAngles():
    import lcm
    q0 = np.zeros(7)

    def HandleIiwaStatusMessage(channel, data):
      msg = lcmt_iiwa_status.decode(data)
      q0[:] = msg.joint_position_measured

    lc = lcm.LCM()
    lc.subscribe("IIWA_STATUS", HandleIiwaStatusMessage)
    lc.handle()

    return q0

  @staticmethod
  def ConnectPortsAndAddLoggers(builder, station, plan_runner):
    builder.AddSystem(plan_runner)
    builder.Connect(plan_runner.GetOutputPort("gripper_setpoint"),
            station.GetInputPort("wsg_position"))
    builder.Connect(plan_runner.GetOutputPort("force_limit"),
            station.GetInputPort("wsg_force_limit"))
    builder.Connect(plan_runner.GetOutputPort("iiwa_position_command"),
            station.GetInputPort("iiwa_position"))
    builder.Connect(plan_runner.GetOutputPort("iiwa_torque_command"),
            station.GetInputPort("iiwa_feedforward_torque"))

    builder.Connect(station.GetOutputPort("iiwa_position_measured"),
            plan_runner.GetInputPort("iiwa_position"))
    builder.Connect(station.GetOutputPort("iiwa_position_commanded"),
            plan_runner.GetInputPort("iiwa_position_cmd"))
    builder.Connect(station.GetOutputPort("iiwa_velocity_estimated"),
            plan_runner.GetInputPort("iiwa_velocity"))
    builder.Connect(station.GetOutputPort("iiwa_torque_external"),
            plan_runner.GetInputPort("iiwa_torque_external"))

    # Add logger
    # iiwa_position_command_log = LogOutput(
    #   plan_runner.GetOutputPort("iiwa_position_command"), builder)
    # iiwa_position_command_log.DeclarePeriodicPublish(0.01)

    # iiwa_external_torque_log = LogOutput(
    #   station.GetOutputPort("iiwa_torque_external"), builder)
    # iiwa_external_torque_log.DeclarePeriodicPublish(0.01)

    # iiwa_position_measured_log = LogOutput(
    #   station.GetOutputPort("iiwa_position_measured"), builder)
    # iiwa_position_measured_log.DeclarePeriodicPublish(0.01)

    # iiwa_velocity_estimated_log = LogOutput(
    #   station.GetOutputPort("iiwa_velocity_estimated"), builder)
    # iiwa_velocity_estimated_log.DeclarePeriodicPublish(0.005)

    # return (iiwa_position_command_log, iiwa_position_measured_log,
    #     iiwa_external_torque_log, iiwa_velocity_estimated_log)

  @staticmethod
  def SetInitialPlanRunnerState(plan_runner, simulator, diagram):
    """
    Sets iiwa_position_command, part of the discrete state of plan_runner,
      to the initial state of the robot at t=0.
    Otherwise the position command at t=0 is 0, driving the robot to its
      upright position, usually resulting in huge velocity.
    Calling this function after simulator.Initialize() puts a column of
      zeros at the beginning of iiwa_position_command_log, but that
      zero command doesn't seem to be sent to the robot.
    """
    plan_runner_context = \
      diagram.GetMutableSubsystemContext(plan_runner,
                         simulator.get_mutable_context())
    iiwa_position_input_port = plan_runner.GetInputPort("iiwa_position")
    q0_iiwa = plan_runner.EvalVectorInput(
      plan_runner_context,
      iiwa_position_input_port.get_index()).get_value()
    for i in range(7):
      plan_runner_context.get_mutable_discrete_state(0).SetAtIndex(
        i, q0_iiwa[i])

  ###########################################################
  ## RL environment for single object post-grasp lift up
  ###########################################################
  def reset(self):
    NotImplementedError()

  @property
  def observation_space(self):
    """
    image space observation
    """
    # low = np.zeros([64, 64, 3], dtype=uint8)
    # high = np.ones([64, 64, 3], dtype=uint8)
    # spaces = {'image': gym.spaces.Box(low, high)}
    # return gym.spaces.Dict(spaces)
    return None

  @property
  def action_space(self):
    ### incremental line push or grasp or release
    ### always assume gripper to be in top-down orientation
    return None

  def step(self):
    return None

  def render(self):
    return None
  

def GetInterploatePosition(pos_start, pos_end):
  """
  tau is an interpolation factor ranging from 0 (start) to 1 (end)
  """
  # pos_start = T_start.matrix()[0:3, 3]
  # pos_end = T_goal.matrix()[0:3, 3]
  return lambda tau: pos_start + tau * (pos_end - pos_start)

def GetInterpolateOrientation(R_start:np.array, R_end:np.array):
  """
  tau is an interpolation factor ranging from 0 (start) to 1 (end)
  Using scipy.slerp function
  """
  R_range = scipyR.from_dcm([R_start, R_end])
  t_range = [0,1]
  slerp = Slerp(t_range, R_range)
  def interp_R(tau):
    rotation_cur = slerp([tau]).as_dcm()[0]
    return RotationMatrix(rotation_cur)
  return interp_R

if __name__ == '__main__':
  sim = ManipulationStationSimulator(visualizer='drake')
  sim.init_simulator()
  print('works fine before resetting the context')
  sim.sim_duration(2.)
  sim.incremental_move_in_world(increment=0.3, direction='y')
  print('now reset the context')
  sim.simulator.reset_context(sim.simulator_init_context)
  print('works fine for simple AdvanceTo with no commands')
  sim.sim_duration(2.0)
  print('segmentation fault if arm is commanded to move again')
  sim.incremental_move_in_world(direction='y')