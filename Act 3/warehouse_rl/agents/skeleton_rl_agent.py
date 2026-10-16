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
        
        action = {
            'staffing_action': self._get_naive_staffing_action(financial_state, employee_info),
            'layout_swap': self._get_naive_layout_action(current_timestep),
            'order_assignments': self._get_naive_order_assignments(queue_info, employee_info)
        }
        
        # TODO: Students should implement proper action recording for optimization
        self.action_history.append(action.copy())
        
        return action
    
    def _get_naive_staffing_action(self, financial_state, employee_info) -> int:
        """
        WEEK 2 STEP 1: Staffing decisions - students should improve this!
        
        Current problems:
        - Ignores queue length and workload
        - Random decisions regardless of profit
        - No consideration of employee efficiency
        """
        
        # TODO WEEK 2 STEP 1: Students should implement intelligent staffing logic
        # Current approach: Make random decisions based on "vibes"
        
        current_profit = financial_state[0]
        num_employees = np.sum(employee_info[:, 0] > 0)  # Count active employees
        
        # Terrible logic: Random decisions with slight bias
        # BUT: Hire managers more frequently so layout optimization can work
        has_manager = np.any(employee_info[:, 5] == 1)  # Check if we have a manager
        
        if np.random.random() < 0.3:  # 30% chance to hire
            if num_employees < 20:  # Don't go completely overboard
                return 1  # Hire worker
        elif np.random.random() < 0.1:  # 10% chance to fire
            if num_employees > 1:  # Don't fire everyone
                return 2  # Fire worker
        elif np.random.random() < 0.2:  # 20% chance to hire manager (increased from 5%)
            if not has_manager or num_employees > 10:  # Hire manager if we don't have one, or if we have lots of workers
                return 3  # Hire manager
        
        return 0  # No action
    
    def _get_naive_layout_action(self, current_timestep) -> list:
        """Greedy profit-oriented slotting: move frequently picked items nearer dispatch.

        Estimate travel savings using item access frequency and the nearest truck bay.
        Add a small co-occurrence benefit for items typically picked together.
        Require a manager, limit relocation frequency, and reject unprofitable swaps.
        """
        from environment.employee import EmployeeState
        grid = self.env.warehouse_grid
        manager = next((e for e in self.env.employees if e.is_manager), None)

        # Relocations require an idle manager and low warehouse pressure.
        if manager is None or manager.state != EmployeeState.IDLE:
            return [0, 0]
        if current_timestep < 100 or current_timestep % 50 != 0:
            return [0, 0]
        if len(self.env.order_queue.orders) > max(5, len(self.env.employees) * 3):
            return [0, 0]

        # Avoid repeatedly moving stock that just moved.
        if not hasattr(self, "_layout_cooldown"):
            self._layout_cooldown = {}
        self._layout_cooldown = {
            p: t for p, t in self._layout_cooldown.items()
            if current_timestep - t < 400
        }

        stations = grid.truck_bay_positions
        positions = []
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.cell_types[y, x] == 1:  # CellType.STORAGE
                    item = int(grid.item_grid[y, x])
                    if item >= 0:
                        idx = y * grid.width + x
                        distance = min(abs(x - sx) + abs(y - sy) for sx, sy in stations)
                        positions.append((idx, item, float(distance)))

        if len(positions) < 2:
            return [0, 0]

        frequencies = grid.item_access_frequency
        cooc = grid.item_cooccurrence
        total_access = float(np.sum(frequencies))

        # Add a small prior so optimization can start with limited observations.
        def demand(item):
            return float(frequencies[item]) + 0.1 * total_access / max(1, len(frequencies)) + 0.05

        # Expected reduced travel when swapping item a at far spot with b at near spot:
        # (d_far - d_near) * (demand_a - demand_b).
        best_gain, best_swap = 0.0, None
        far_positions = sorted(positions, key=lambda p: p[2], reverse=True)[:35]
        near_positions = sorted(positions, key=lambda p: p[2])[:35]

        for a_idx, a_item, a_distance in far_positions:
            if a_idx in self._layout_cooldown:
                continue
            for b_idx, b_item, b_distance in near_positions:
                if a_idx == b_idx or a_item == b_item or b_idx in self._layout_cooldown:
                    continue
                if a_distance <= b_distance:
                    continue
                gain = (a_distance - b_distance) * (demand(a_item) - demand(b_item))
                if gain <= 1.0:
                    continue

                # Prefer swaps that bring frequently co-ordered stock closer.
                a_x, a_y = a_idx % grid.width, a_idx // grid.width
                b_x, b_y = b_idx % grid.width, b_idx // grid.width
                affinity = 0.0
                for other_idx, other_item, _ in positions:
                    if other_idx in (a_idx, b_idx):
                        continue
                    x, y = other_idx % grid.width, other_idx // grid.width
                    before = abs(a_x - x) + abs(a_y - y)
                    after = abs(b_x - x) + abs(b_y - y)
                    affinity += float(cooc[a_item, other_item] - cooc[b_item, other_item]) * (before - after)
                gain += 0.02 * affinity

                if gain > best_gain:
                    best_gain, best_swap = gain, (a_idx, b_idx)

        # A manager's travel and time are costly: require a meaningful gain.
        if best_swap is not None and best_gain > 2.0:
            for pos in best_swap:
                self._layout_cooldown[pos] = int(current_timestep)
            return list(best_swap)
        return [0, 0]

    def _get_naive_order_assignments(self, queue_info, employee_info) -> list:
        """
        WEEK 2 STEP 2: Order assignment - students should improve this!
        
        Current problems:
        - Ignores employee locations
        - No consideration of order priority/value
        - Random assignments regardless of efficiency
        - Doesn't check if employees are actually available
        """
        
        # TODO WEEK 2 STEP 2: Students should implement intelligent order assignment
        # Current approach: Random assignments
        
        assignments = [0] * 20  # No assignments by default
        
        # Count how many employees we have (very naive)
        num_employees = int(np.sum(employee_info[:, 0] > 0))
        
        # Randomly assign first few orders to first few employees
        if num_employees > 0:
            for i in range(min(3, num_employees)):  # Only assign 3 orders max
                if np.random.random() < 0.6:  # 60% chance to assign
                    assignments[i] = (i % num_employees) + 1  # Random employee
        
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
        return len(self.action_history) % 50 == 0  # Arbitrary update frequency
    
    def get_performance_metrics(self) -> Dict:
        """
        Get agent performance metrics for analysis
        Students can use this to debug their improvements
        """
        if not self.reward_history:
            return {"avg_reward": 0, "total_actions": 0}
        
        return {
            "avg_reward": np.mean(self.reward_history[-100:]),  # Last 100 rewards
            "total_actions": len(self.action_history),
            "exploration_rate": self.exploration_rate,
            "recent_performance": np.mean(self.reward_history[-10:]) if len(self.reward_history) >= 10 else 0
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