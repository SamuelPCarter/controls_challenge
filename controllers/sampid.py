from . import BaseController
import numpy as np
import cvxpy as cp

class Controller(BaseController):
    """
    A custom feedback controller using Model Predictive Control (MPC) with explicit jerk minimization
    """
    def __init__(self):
        # Initialize the state and control vectors
        self.state = np.zeros((2, 1))
        self.control = 0
        
        # System dynamics (example for a simplified car model)
        self.A = np.array([[1, 0.1],
                           [0, 1]])
        self.B = np.array([[0.5],
                           [1]])
        
        # Prediction horizon
        self.horizon = 20
        
        # Define cost matrices with higher weights on key performance metrics
        self.Q = np.diag([100, 50])  # High weight on position error and velocity error
        self.R = np.diag([1])        # Regular weight on control effort
        self.P = np.diag([10])       # Additional penalty for jerk (change in control)
        
        # Define the optimization variables
        self.x = cp.Variable((2, self.horizon + 1))
        self.u = cp.Variable((1, self.horizon))
        self.delta_u = cp.Variable((1, self.horizon - 1))  # Variable for change in control
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Calculate the current error
        error = target_lataccel - current_lataccel
        
        # Assuming state is a tuple, extract relevant elements
        lateral_velocity = state[1]  # Example index, change based on actual structure
        
        # Set up the initial state
        self.state[0, 0] = error
        self.state[1, 0] = lateral_velocity
        
        # Define the cost function
        cost = 0
        constraints = []
        
        for t in range(self.horizon):
            cost += cp.quad_form(self.x[:, t], self.Q) + cp.quad_form(self.u[:, t], self.R)
            constraints += [self.x[:, t + 1] == self.A @ self.x[:, t] + self.B @ self.u[:, t]]
            constraints += [cp.abs(self.u[:, t]) <= 1]  # Add control input constraint to avoid high jerk
            
            if t > 0:
                cost += cp.quad_form(self.u[:, t] - self.u[:, t - 1], self.P)  # Penalize jerk
            
        # Terminal cost
        cost += cp.quad_form(self.x[:, self.horizon], self.Q)
        
        # Define the optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        # Solve the optimization problem
        prob.solve()
        
        # Apply the first control input
        self.control = self.u.value[0, 0]
        
        # Update the state using the applied control input
        self.state = self.A @ self.state + self.B * self.control
        
        return self.control
