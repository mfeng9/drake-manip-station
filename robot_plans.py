import numpy as np
from plan_utils import ConnectPointsWithCubicPolynomial

plan_type_strings = [
    "JointSpacePlan",
    "JointSpacePlanRelative",
    "JointSpacePlanGoToTarget",
]

PlanTypes = dict()
for plan_types_string in plan_type_strings:
    PlanTypes[plan_types_string] = plan_types_string


class PlanBase:
    def __init__(self,
                 type=None,
                 trajectory=None):
        self.type = type
        self.traj = trajectory
        self.traj_d = None
        self.duration = None
        self.start_time = None
        if trajectory is not None:
            self.traj_d = trajectory.derivative(1)
            self.duration = trajectory.end_time()

    def get_duration(self):
        return self.duration

    def CalcPositionCommand(self, q_iiwa, q_cmd, v_iiwa, tau_iiwa, t_plan,
                            control_period):
        pass

    def CalcTorqueCommand(self, q_iiwa, q_cmd, v_iiwa, tau_iiwa, t_plan, control_period):
        return np.zeros(7)


class JointSpacePlan(PlanBase):
    def __init__(self,
                 trajectory=None):
        PlanBase.__init__(self,
                          type=PlanTypes["JointSpacePlan"],
                          trajectory=trajectory)

    def CalcPositionCommand(self, q_iiwa, q_cmd, v_iiwa, tau_iiwa, t_plan,
                            control_period):
        return self.traj.value(t_plan).flatten()


class JointSpacePlanGoToTarget(PlanBase):
    """
    The robot goes to q_target from its configuration when this plan starts.
    """

    def __init__(self, duration, q_target):
        PlanBase.__init__(self,
                          type=PlanTypes["JointSpacePlanGoToTarget"],
                          trajectory=None)
        self.q_target = q_target
        self.duration = duration

    def UpdateTrajectory(self, q_start):
        self.traj = ConnectPointsWithCubicPolynomial(
            q_start, self.q_target, self.duration)
        self.traj_d = self.traj.derivative(1)

    def CalcPositionCommand(self, q_iiwa, q_cmd, v_iiwa, tau_iiwa, t_plan,
                            control_period):
        if self.traj is None:
            self.UpdateTrajectory(q_start=q_cmd)
        return self.traj.value(t_plan).flatten()


class JointSpacePlanRelative(PlanBase):
    """
    The robot goes from its configuration when this plan starts (q_current) by
    delta_q to reach the final configuration (q_current + delta_q).
    """

    def __init__(self, duration, delta_q):
        PlanBase.__init__(self,
                          type=PlanTypes["JointSpacePlanRelative"],
                          trajectory=None)
        self.delta_q = delta_q
        self.duration = duration

    def UpdateTrajectory(self, q_start):
        self.traj = ConnectPointsWithCubicPolynomial(
            q_start, self.delta_q + q_start, self.duration)
        self.traj_d = self.traj.derivative(1)

    def CalcPositionCommand(self, q_iiwa, q_cmd, v_iiwa, tau_iiwa, t_plan,
                            control_period):
        if self.traj is None:
            self.UpdateTrajectory(q_start=q_cmd)
        return self.traj.value(t_plan).flatten()
