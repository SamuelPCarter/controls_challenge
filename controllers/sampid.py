from . import BaseController
import numpy as np

class Controller(BaseController):
    """
    A simple PID controller
    """
    def __init__(self,):
        # PID coefficients
        self.p = 0.18  # Proportional gain: provides a faster response to errors
        self.i = 0.08  # Integral gain: corrects accumulated errors over time
        self.d = -0.05  # Derivative gain: smooths out control signal and reduces overshoot
        
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

