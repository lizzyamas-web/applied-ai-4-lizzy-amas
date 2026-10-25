
#!/usr/bin/env python3
"""
Skeleton Optimization Agent - Template for students to implement optimization algorithms

IMPORTANT: This is NOT an RL agent - it's a template for 
students to implement their own optimization algorithms (greedy, Hungarian, etc.)

This agent currently makes terrible decisions:
- Random staffing decisions
- No layout optimization 
- Ignores order priorities
- No intelligent assignment logic

Students should replace the naive methods with proper optimization algorithms.
"""

import numpy as np
from typing import Dict, Optional
from .standardized_agents import BaselineAgent


class SkeletonOptimizationAgent(BaselineAgent):
    """
    Template for students to implement their own optimization algorithms.

    Students should replace the naive methods with:
    - Economic models for staffing decisions
    - Hungarian algorithm for order assignment
    - Greedy search for layout optimization
    - Multi-objective optimization techniques
    """

    def __init__(self, env):
        super().__init__(env)
        self.name = "StudentOptimization"

        # Enable unlimited hiring for true economic optimization
        self.env._unlimited_hiring = True

        # Students can add their own state tracking here
        self.performance_metrics = []
        self.decision_history = []

        # Week 1: Track warehouse layout efficiency over time
        self.layout_performance_history = []

        # Students can add algorithm parameters here
        # Example: Hungarian algorithm matrices, greedy search parameters, etc.
        self.staffing_parameters = {
            'hire_threshold_ratio': 3.0,
            'fire_threshold_ratio': 2.0,
            'profit_threshold': 0
        }

        # Students can implement adaptive parameters that change based on performance
        self.adaptive_optimization_enabled = False

    def reset(self):
        """Reset agent state - students should expand this"""
        self.action_history = []
        self.reward_history = []

        # Week 1: Reset layout performance history for each episode
        self.layout_performance_history = []
        self._last_staff_change = -1000

        # TODO: Reset any neural network states, replay buffers, etc.

    def get_action(self, observation: Dict) -> Dict:
        """
        Generate action based on observation.
        Current implementation is intentionally terrible - students should improve!
        """

        # Extract basic info from observation
        current_timestep = observation['time'][0]
        financial_state = observation['financial']  # [profit, revenue, costs, burn_rate]
        queue_info = observation['order_queue']
        employee_info = observation['employees']

        # Week 1: Track layout performance every 100 timesteps.
        if current_timestep > 0 and current_timestep % 100 == 0:
            self.track_layout_performance(current_timestep)

        action = {
            'staffing_action': self._get_naive_staffing_action(
                financial_state, employee_info
            ),
            'layout_swap': self._get_naive_layout_action(current_timestep),
            'order_assignments': self._get_naive_order_assignments(
                queue_info, employee_info
            )
        }

        # TODO: Students should implement proper action recording for optimization
        self.action_history.append(action.copy())

        return action

    def _get_naive_staffing_action(self, financial_state, employee_info) -> int:
        """Demand-responsive staffing with hysteresis and hiring cooldown."""
        workers = [e for e in self.env.employees if not getattr(e, 'is_manager', False)]
        managers = [e for e in self.env.employees if getattr(e, 'is_manager', False)]
        count = len(workers)
        backlog = len(self.env.order_queue.orders)
        profit = float(financial_state[0])
        timestep = int(getattr(self.env, 'current_timestep', 0))

        # Keep a manager to enable the existing layout strategy.
        if not managers and timestep < 1000 and count >= 2:
            return 3

        # Avoid changing workforce every timestep on transient demand spikes.
        last_change = getattr(self, '_last_staff_change', -1000)
        if timestep - last_change < 100:
            return 0

        if count < 2:
            action = 1
        elif backlog > 3.0 * count and profit > 1000 and count < 12:
            action = 1
        elif count > 2 and backlog < 1.5 * count and (profit < 0 or backlog == 0):
            action = 2
        else:
            action = 0

        if action:
            self._last_staff_change = timestep
        return action

    def _get_naive_layout_action(self, current_timestep) -> list:
        """
        Week 1: Frequency-based greedy warehouse layout optimization.
        Move popular items closer to the nearest truck bay.
        """
        from environment.employee import EmployeeState

        grid = self.env.warehouse_grid

        # Run layout optimization every 100 timesteps.
        if current_timestep < 100 or current_timestep % 100 != 0:
            return [0, 0]

        # A manager must be available to perform relocations.
        manager = next(
            (e for e in self.env.employees if e.is_manager),
            None
        )
        if manager is None or manager.state != EmployeeState.IDLE:
            return [0, 0]

        # Avoid interrupting operations when the order queue is busy.
        if len(self.env.order_queue.orders) > max(
            5, len(self.env.employees) * 3
        ):
            return [0, 0]

        # Every 200 timesteps, prioritize co-occurrence grouping.
        # If no suitable grouping swap exists, continue with
        # the original frequency-based optimization.
        if current_timestep % 200 == 0:
            grouping_swap = self._find_cooccurrence_swap()

            if grouping_swap is not None:
                print(
                    f"[CO-OCCURRENCE] Step {current_timestep}: "
                    f"Proposed swap {grouping_swap}"
                )
                return grouping_swap

        frequencies = grid.item_access_frequency
        active_frequencies = frequencies[frequencies > 0]

        if len(active_frequencies) == 0:
            return [0, 0]

        # Top 25% of accessed items are considered hot.
        threshold = np.percentile(active_frequencies, 75)

        hot_items = [
            item for item in range(len(frequencies))
            if frequencies[item] > 0
            and frequencies[item] >= threshold
        ]

        delivery_positions = grid.truck_bay_positions

        best_swap = None
        best_benefit = 0.0

        for item in hot_items:
            for current_pos in grid.find_item_locations(item):
                current_distance = min(
                    grid.manhattan_distance(current_pos, delivery)
                    for delivery in delivery_positions
                )

                if current_distance <= 3:
                    continue

                current_idx = (
                    current_pos[1] * grid.width + current_pos[0]
                )

                for y in range(grid.height):
                    for x in range(grid.width):
                        if grid.cell_types[y, x] != 1:
                            continue

                        if not grid.can_access_storage(x, y):
                            continue

                        candidate_pos = (x, y)

                        if candidate_pos == current_pos:
                            continue

                        candidate_distance = min(
                            grid.manhattan_distance(
                                candidate_pos, delivery
                            )
                            for delivery in delivery_positions
                        )

                        distance_saved = (
                            current_distance - candidate_distance
                        )

                        if distance_saved < 1:
                            continue

                        other_item = grid.get_item_at_position(x, y)
                        other_frequency = (
                            frequencies[other_item]
                            if other_item is not None
                            else 0
                        )

                        # Account for the product being displaced.
                        benefit = (
                            distance_saved
                            * (frequencies[item] - other_frequency)
                        )

                        if benefit > best_benefit:
                            best_benefit = benefit
                            candidate_idx = y * grid.width + x
                            best_swap = [current_idx, candidate_idx]

        if best_swap is not None:
            print(
                f"[LAYOUT] Step {current_timestep}: "
                f"Proposed swap {best_swap}, "
                f"estimated benefit {best_benefit:.2f}"
            )
            return best_swap

        return [0, 0]

    def _find_cooccurrence_swap(self):
        """
        Week 1: Group products frequently ordered together.

        Requirements:
        - Co-occurrence count must be at least 3.
        - Products must be at least 4 grid steps apart.
        - Proposed relocation must reduce their distance.
        - Prefer empty locations or locations with cold items.
        """
        grid = self.env.warehouse_grid
        cooccurrence = grid.item_cooccurrence
        frequencies = grid.item_access_frequency

        if cooccurrence.max() == 0:
            return None

        best_swap = None
        best_benefit = 0.0

        for item1 in range(grid.num_item_types):
            for item2 in range(item1 + 1, grid.num_item_types):

                count = cooccurrence[item1, item2]

                # Products must have been ordered together at least 3 times.
                if count < 3:
                    continue

                locations1 = grid.find_item_locations(item1)
                locations2 = grid.find_item_locations(item2)

                if not locations1 or not locations2:
                    continue

                for pos1 in locations1:
                    for pos2 in locations2:

                        current_distance = grid.manhattan_distance(
                            pos1, pos2
                        )

                        # Only optimize pairs at least 4 steps apart.
                        if current_distance < 4:
                            continue

                        # Search nearby storage cells around item2.
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:

                                x = pos2[0] + dx
                                y = pos2[1] + dy

                                # Check warehouse boundaries.
                                if not (
                                    0 <= x < grid.width
                                    and 0 <= y < grid.height
                                ):
                                    continue

                                # Destination must be a storage cell.
                                if grid.cell_types[y, x] != 1:
                                    continue

                                # Destination must be accessible.
                                if not grid.can_access_storage(x, y):
                                    continue

                                candidate = (x, y)

                                if candidate == pos1 or candidate == pos2:
                                    continue

                                new_distance = grid.manhattan_distance(
                                    candidate, pos2
                                )

                                improvement = current_distance - new_distance

                                # Require at least one step of improvement.
                                if improvement < 1:
                                    continue

                                existing_item = grid.get_item_at_position(
                                    x, y
                                )

                                # Prefer empty cells or cold items.
                                if existing_item is not None:
                                    if frequencies[existing_item] > 0:
                                        continue

                                benefit = count * improvement

                                if benefit > best_benefit:
                                    best_benefit = benefit

                                    source_idx = (
                                        pos1[1] * grid.width + pos1[0]
                                    )

                                    target_idx = y * grid.width + x

                                    best_swap = [
                                        source_idx,
                                        target_idx
                                    ]

        return best_swap

    def _calculate_layout_efficiency(self):
        """
        Week 1: Calculate frequency-weighted average distance
        from stored products to their nearest truck bay.

        Lower average distance indicates a more efficient layout
        under this metric.
        """
        grid = self.env.warehouse_grid
        frequencies = grid.item_access_frequency

        total_weighted_distance = 0.0
        total_frequency = 0.0

        for item_type, frequency in enumerate(frequencies):
            if frequency <= 0:
                continue

            locations = grid.find_item_locations(item_type)

            if not locations:
                continue

            # Use the nearest stored copy of each product.
            nearest_distance = min(
                grid.manhattan_distance(
                    location,
                    grid.get_nearest_truck_bay_position(location)
                )
                for location in locations
            )

            total_weighted_distance += frequency * nearest_distance
            total_frequency += frequency

        if total_frequency == 0:
            return 0.0

        return total_weighted_distance / total_frequency

    def track_layout_performance(self, current_timestep):
        """
        Week 1: Record warehouse layout performance over time.
        """
        average_distance = self._calculate_layout_efficiency()

        relocations = getattr(self.env, 'relocations_history', [])

        completed_relocations = sum(
            1 for relocation in relocations
            if relocation.get('status') == 'completed'
        )

        record = {
            'timestep': int(current_timestep),
            'average_weighted_distance': float(average_distance),
            'completed_relocations': completed_relocations
        }

        self.layout_performance_history.append(record)

        print(
            f"[LAYOUT PERFORMANCE] Step {current_timestep}: "
            f"Weighted distance={average_distance:.2f}, "
            f"Completed relocations={completed_relocations}"
        )

        return record

    def _get_idle_workers(self, employee_info):
        """Return (environment employee index, (x,y)) for idle non-managers."""
        from environment.employee import EmployeeState
        result = []
        for idx, worker in enumerate(self.env.employees):
            if getattr(worker, 'is_manager', False):
                continue
            if getattr(worker, 'state', None) != EmployeeState.IDLE:
                continue
            if idx >= len(employee_info):
                continue
            row = employee_info[idx]
            if len(row) < 2:
                continue
            result.append((idx, (int(row[0]), int(row[1]))))
        return result

    def _calculate_order_distance(self, worker_pos, order):
        """Estimate worker travel to the closest required product."""
        grid = self.env.warehouse_grid
        required = getattr(order, 'items', [])
        if isinstance(required, dict):
            required = required.keys()
        distances = []
        for item in required:
            item_type = getattr(item, 'item_type', item)
            try:
                locations = grid.find_item_locations(int(item_type))
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            for loc in locations:
                distances.append(grid.manhattan_distance(worker_pos, loc))
        return min(distances) if distances else grid.width + grid.height

    def _get_naive_order_assignments(self, queue_info, employee_info) -> list:
        """Greedy unique matching using 70% proximity and 30% order value."""
        assignments = [0] * 20
        workers = self._get_idle_workers(employee_info)
        orders = list(self.env.order_queue.orders)[:20]
        if not workers or not orders:
            return assignments

        def order_value(order):
            for attr in ('value', 'total_value', 'reward', 'order_value'):
                value = getattr(order, attr, None)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        pass
            return 0.0

        values = [order_value(o) for o in orders]
        max_value = max(max(values), 1.0)
        pairs = []
        for worker_idx, pos in workers:
            for order_idx, order in enumerate(orders):
                distance = self._calculate_order_distance(pos, order)
                distance_score = 1.0 / (1.0 + distance)
                value_score = max(0.0, values[order_idx]) / max_value
                score = 0.7 * distance_score + 0.3 * value_score
                pairs.append((score, worker_idx, order_idx))

        used_workers = set()
        used_orders = set()
        for score, worker_idx, order_idx in sorted(pairs, reverse=True):
            if worker_idx in used_workers or order_idx in used_orders:
                continue
            # Order slot holds the 1-based employee index (0 means no assignment).
            assignments[order_idx] = worker_idx + 1
            used_workers.add(worker_idx)
            used_orders.add(order_idx)
        return assignments

    def record_reward(self, reward: float):
        """
        WEEK 3 STEP 1: Reward tracking and learning - students should expand this

        TODO WEEK 3 STEP 1: Students should implement:
        - Proper reward tracking
        - Performance analysis
        - Adaptive parameter adjustment
        - Multi-objective optimization
        """
        self.reward_history.append(reward)

        # Skeleton optimization - doesn't actually learn anything useful
        if len(self.reward_history) > 10:
            # "Update" weights randomly (this doesn't actually improve performance)
            if reward > 0:
                self.staffing_weights += np.random.randn(4) * 0.01
                self.layout_weights += np.random.randn(3) * 0.01

    def should_update_policy(self) -> bool:
        """
        WEEK 3 STEP 2: Policy updates and adaptation - students should improve this

        TODO WEEK 3 STEP 2: Students should implement proper update schedules
        """
        return len(self.action_history) % 50 == 0

    def get_performance_metrics(self) -> Dict:
        """
        Get agent performance metrics for analysis
        Students can use this to debug their improvements
        """
        if not self.reward_history:
            return {"avg_reward": 0, "total_actions": 0}

        return {
            "avg_reward": np.mean(self.reward_history[-100:]),
            "total_actions": len(self.action_history),
            "exploration_rate": self.exploration_rate,
            "recent_performance": (
                np.mean(self.reward_history[-10:])
                if len(self.reward_history) >= 10 else 0
            )
        }


def create_skeleton_optimization_agent(env) -> SkeletonOptimizationAgent:
    """Factory function to create skeleton Optimization agent"""
    return SkeletonOptimizationAgent(env)


# TODO: Students should implement these advanced components:

class StudentOptimizationAgent(SkeletonOptimizationAgent):
    """
    Template for students to implement their improved Optimization agent

    Suggested improvements:
    1. Replace random staffing with demand-based hiring
    2. Implement proper layout optimization using item frequencies
    3. Add distance-based order assignment
    4. Implement basic Q-optimization or policy gradients
    5. Add proper exploration vs exploitation balance
    """

    def __init__(self, env):
        super().__init__(env)
        self.name = "StudentOptimization"

        # TODO: Students implement these
        # self.q_table = {}
        # self.policy_network = SimpleNeuralNetwork()
        # self.experience_buffer = []
        # self.target_network = None

    def _get_improved_staffing_action(self, financial_state, employee_info, queue_info):
        """
        WEEK 2 STEP 1: Students implement intelligent staffing:
        - Hire when queue is growing
        - Fire when queue is empty for extended periods
        - Consider profit margins before hiring
        - Balance managers vs workers
        """
        pass

    def _get_improved_layout_action(self, observation):
        """
        WEEK 1 STEP 1: Students implement smart layout optimization:
        - Move frequently accessed items closer to delivery
        - Group items that are often ordered together
        - Only optimize when queue is manageable
        - Track swap effectiveness
        """
        pass

    def _get_improved_order_assignments(self, queue_info, employee_info):
        """
        WEEK 2 STEP 2: Students implement efficient order assignment:
        - Assign orders to closest available employees
        - Prioritize high-value or urgent orders
        - Consider employee current locations
        - Balance workload across employees
        """
        pass

    def learn_from_experience(self, state, action, reward, next_state, done):
        """
        WEEK 3 STEP 3: Students implement multi-objective optimization:
        - Performance trend analysis
        - Adaptive parameter tuning
        - Multi-objective trade-off handling
        - Robust optimization techniques
        """
        pass
