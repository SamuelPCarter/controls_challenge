from . import BaseController
import numpy as np

class Controller(BaseController):
    """
    A simple PID controller
    """
    def __init__(self):
        # Further fine-tuned PID coefficients for improved performance
        self.p = 0.21
        self.i = 0.10
        self.d = -0.015
        
        # Initialize integral of the error
        self.error_integral = 0
        # Initialize previous error
        self.prev_error = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Calculate the current error
        error = (target_lataccel - current_lataccel)
        # Accumulate the error over time for integral control
        self.error_integral += error
        # Calculate the difference in error for derivative control
        error_diff = error - self.prev_error
        # Update the previous error
        self.prev_error = error
        # Compute the control output by summing the proportional, integral, and derivative terms
        return self.p * error + self.i * self.error_integral + self.d * error_diff

