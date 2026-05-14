import math

import habitat_sim
import numpy as np
import quaternion

from .simulation import SceneSimulator
from .visualization import display_env


class TimeAwareSceneSimulator(SceneSimulator):
    """Unordered multi-target simulator with a step budget.

    This keeps the original LH-VLN simulator untouched. Targets can be completed
    in any order: a stop action completes the nearest remaining target if it is
    within success distance.
    """

    def __init__(self, args, config, time_budget=None, target_values=None):
        super().__init__(args, config)
        self._set_episode_start_state()
        self.time_budget = int(time_budget if time_budget is not None else args.max_step)
        self.time_used = 0
        self.remaining_targets = set(range(self.target_num))
        self.completed_targets = []
        self.completion_order = []
        self.abandoned_targets = []
        self.target_values = target_values or [1.0 for _ in range(self.target_num)]
        self.failed_stops = 0
        self.nav_steps = []
        self.nav_errors = []
        self.successes = [False for _ in range(self.target_num)]
        self.oracle_successes = [False for _ in range(self.target_num)]
        self.gt_path = []
        self.info = self.get_time_aware_info()

    def _set_episode_start_state(self):
        start_pos = self.config.get("Start pos")
        if start_pos is None:
            return

        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(start_pos, dtype=np.float32)

        start_yaw = self.config.get("Start yaw")
        if start_yaw is not None:
            yaw = math.radians(float(start_yaw))
            agent_state.rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0])

        self.agent.set_state(agent_state)
        if not self.no_render:
            self.observations = self.sim.get_sensor_observations()

    @property
    def time_remaining(self):
        return max(self.time_budget - self.time_used, 0)

    def get_coord_by_index(self, target_index):
        target_positions = self.config.get("Target positions")
        if target_positions and target_positions[target_index] is not None:
            return [np.array(target_positions[target_index], dtype=np.float32)]

        obj_target = self.target[target_index]
        region_id = self.region[target_index]
        coord_list = []
        scene = self.sim.semantic_scene
        for region in scene.regions:
            if region.id[1:] != str(region_id):
                continue
            for obj in region.objects:
                if obj.category.name() == obj_target:
                    coord_list.append(obj.aabb.center)
        return coord_list

    def geodesic_distance_from_current(self, position_b_list):
        position_a, _ = self.return_state()
        geo_dis = math.inf
        coord = position_b_list[0]

        for position_b in position_b_list:
            path = habitat_sim.nav.ShortestPath()
            path.requested_start = np.array(position_a, dtype=np.float32)
            path.requested_end = np.array(position_b, dtype=np.float32)
            if self.pathfinder.find_path(path) and path.geodesic_distance < geo_dis:
                geo_dis = path.geodesic_distance
                coord = position_b
        return geo_dis, coord

    def get_target_info(self, target_index):
        coord_list = self.get_coord_by_index(target_index)
        if not coord_list:
            return {
                "target_index": target_index,
                "target": self.target[target_index],
                "target coord": None,
                "geo dis": math.inf,
            }
        snap_coord_list = [self.pathfinder.snap_point(coord) for coord in coord_list]
        geo_dis, snap_coord = self.geodesic_distance_from_current(snap_coord_list)
        return {
            "target_index": target_index,
            "target": self.target[target_index],
            "target coord": snap_coord,
            "geo dis": geo_dis,
        }

    def get_remaining_target_infos(self):
        return [self.get_target_info(index) for index in sorted(self.remaining_targets)]

    def select_nearest_target(self):
        infos = self.get_remaining_target_infos()
        if not infos:
            return None
        return min(infos, key=lambda item: item["geo dis"])

    def get_time_aware_info(self):
        position, rotation = self.return_state()
        remaining_infos = self.get_remaining_target_infos()
        nearest = min(remaining_infos, key=lambda item: item["geo dis"]) if remaining_infos else None
        return {
            "agent position": position,
            "agent rotation": rotation,
            "remaining_targets": remaining_infos,
            "nearest_target": nearest,
            "time_budget": self.time_budget,
            "time_used": self.time_used,
            "time_remaining": self.time_remaining,
            "completed_targets": list(self.completed_targets),
        }

    def _mark_oracle_successes(self):
        for info in self.get_remaining_target_infos():
            if info["geo dis"] < self.args.success_dis:
                self.oracle_successes[info["target_index"]] = True

    def _complete_target_if_possible(self, target_index):
        if target_index not in self.remaining_targets:
            return False, None

        target_info = self.get_target_info(target_index)
        self.nav_errors.append(target_info["geo dis"])
        if target_info["geo dis"] < self.args.success_dis:
            self.successes[target_index] = True
            self.remaining_targets.remove(target_index)
            self.completed_targets.append(target_index)
            self.completion_order.append(
                {
                    "target_index": target_index,
                    "target": self.target[target_index],
                    "time_used": self.time_used,
                    "navigation_error": target_info["geo dis"],
                }
            )
            former = sum(self.nav_steps)
            self.nav_steps.append(self.time_used - former)
            print(f"\n***** time-aware nav to {target_info['target']} success! *****\n")
            return True, target_info

        self.failed_stops += 1
        print(f"\n***** time-aware nav stop failed near {target_info['target']} *****\n")
        return False, target_info

    def _complete_nearest_if_possible(self):
        nearest = self.select_nearest_target()
        if nearest is None:
            return False, None
        return self._complete_target_if_possible(nearest["target_index"])

    def _abandon_target(self, target_index, reason):
        if target_index not in self.remaining_targets:
            return
        self.remaining_targets.remove(target_index)
        self.abandoned_targets.append(
            {
                "target_index": target_index,
                "target": self.target[target_index],
                "time_used": self.time_used,
                "reason": reason,
            }
        )
        print(f"\n***** time-aware abandon {self.target[target_index]}: {reason} *****\n")

    def actor(self, action, stop_target_index=None):
        if action == "stop":
            pass
        else:
            self.observations = self.sim.step(action)

        display_target = "all_targets"
        nearest = self.select_nearest_target()
        if nearest is not None:
            display_target = nearest["target"]
        if self.no_render:
            obs = self.observations
        else:
            obs = display_env(self.observations, action, self.save_path, self.step, display_target)

        if self.step == -1:
            self.step = 0
            self.info = self.get_time_aware_info()
            return obs, self.done, self.info

        print("action: %s, step: %d, time_remaining: %d" % (action, self.step, self.time_remaining))
        self.step += 1
        self.time_used += 1
        self._mark_oracle_successes()

        if action == "stop":
            if stop_target_index is None:
                self._complete_nearest_if_possible()
            else:
                self._complete_target_if_possible(stop_target_index)

        if not self.remaining_targets:
            self.done = True
            self.episode_over = True
            print("\n***** time-aware navigation over: all targets completed! *****\n")

        if self.time_used >= self.time_budget:
            self.episode_over = True
            print("\n***** time-aware navigation over: time budget exhausted! *****\n")

        self.info = self.get_time_aware_info()
        return obs, self.done, self.info

    def get_next_action_to_nearest_target(self):
        nearest = self.select_nearest_target()
        if nearest is None or nearest["target coord"] is None:
            return "stop"
        if math.isinf(nearest["geo dis"]):
            self._abandon_target(nearest["target_index"], "unreachable_geodesic")
            return "stop"
        try:
            return self.get_next_action(nearest["target coord"]) or "stop"
        except habitat_sim.errors.GreedyFollowerError:
            self._abandon_target(nearest["target_index"], "greedy_follower_error")
            return "stop"

    def get_next_action_to_target(self, target_index):
        target_info = self.get_target_info(target_index)
        if target_info["target coord"] is None:
            return "stop"
        if math.isinf(target_info["geo dis"]):
            self._abandon_target(target_index, "unreachable_geodesic")
            return "stop"
        try:
            return self.get_next_action(target_info["target coord"]) or "stop"
        except habitat_sim.errors.GreedyFollowerError:
            self._abandon_target(target_index, "greedy_follower_error")
            return "stop"

    def return_results(self):
        reward = sum(
            self.target_values[index]
            for index, success in enumerate(self.successes)
            if success
        )
        total_reward = sum(self.target_values)
        completion_rate = sum(self.successes) / len(self.successes) if self.successes else 0.0
        return {
            "successes": self.successes,
            "oracle_successes": self.oracle_successes,
            "navigation_steps": self.nav_steps,
            "navigation_errors": self.nav_errors,
            "gt_step": self.gt_step,
            "gt_path": self.gt_path,
            "time_budget": self.time_budget,
            "time_used": self.time_used,
            "time_remaining": self.time_remaining,
            "completed_targets": self.completed_targets,
            "completion_order": self.completion_order,
            "abandoned_targets": self.abandoned_targets,
            "completion_rate": completion_rate,
            "reward": reward,
            "total_reward": total_reward,
            "reward_rate": reward / total_reward if total_reward > 0 else 0.0,
            "success_at_budget": all(self.successes),
            "failed_stops": self.failed_stops,
        }
