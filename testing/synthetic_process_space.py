import numpy as np
import matplotlib.pyplot as plt
import noise

class SyntheticProcessResponse:
    def __init__(self, 
                 param_bounds: dict,
                 output_bounds= (0.0, 1.0), 
                 num_gaussians: int = 5, 
                 noise_scale: float = 0.1, 
                 noise_frequency: float = 2.0, 
                 response_bounds: tuple = (0, 1),
                 seed: int | None = None,
                 calibration_samples=10000):
        """
        Initializes the synthetic process response space using modern NumPy RNG.

        Args:
            param_bounds (dict): Dictionary where keys are parameter names and values are 
                                 tuples of (lower_bound, upper_bound).
            num_gaussians (int): Number of "peaks" or "valleys" to generate in the space.
            noise_scale (float): Magnitude of the Perlin noise added to the signal.
            noise_frequency (float): Frequency of the Perlin noise.
            response_bounds (tuple): Tuple of (min_response, max_response) to scale the output.
            seed (int | None, optional): Random seed for the dedicated generator.
        """
        # --- Modern NumPy Random Generator ---
        # We create a specific generator instance for this class. 
        # This isolates the randomness of this environment from the rest of your code.
        self.rng = np.random.default_rng(seed)

        self.param_bounds = param_bounds
        self.output_bounds = output_bounds
        self.param_names = sorted(list(param_bounds.keys()))
        self.dims = len(self.param_names)
        self.response_bounds = response_bounds
        self.noise_scale = noise_scale
        self.noise_freq = noise_frequency
        self.raw_min = 0.0
        self.raw_max = 1.0

        # --- Generate Random Gaussians ---
        self.gaussians = []
        
        for _ in range(num_gaussians):
            # Random center within the defined bounds
            center = np.array([
                self.rng.uniform(self.param_bounds[p][0], self.param_bounds[p][1]) 
                for p in self.param_names
            ])
            
            # Random bandwidth (width of the bell curve).
            bandwidths = np.array([
                (self.param_bounds[p][1] - self.param_bounds[p][0]) * self.rng.uniform(0.1, 0.5)
                for p in self.param_names
            ])
            
            # Random amplitude
            amplitude = self.rng.uniform(-1.0, 2.0)
            
            self.gaussians.append({
                'center': center,
                'bandwidth': bandwidths,
                'amplitude': amplitude
            })

        # Random offset for Perlin noise
        self.noise_offset = self.rng.uniform(0, 100, self.dims)

        # calibrate the output scale
        self._calibrate_bounds(calibration_samples)

    
    def _raw_evaluate(self, point):
        """Internal method to compute the unscaled response for a numpy array point."""
        # Gaussian Component
        gaussian_sum = 0.0
        for g in self.gaussians:
            diff = (point - g['center']) ** 2
            width = 2 * (g['bandwidth'] ** 2)
            exponent = -np.sum(diff / width)
            gaussian_sum += g['amplitude'] * np.exp(exponent)

        # Perlin Noise Component
        norm_point = []
        for i, p_name in enumerate(self.param_names):
            low, high = self.param_bounds[p_name]
            norm_val = ((point[i] - low) / (high - low)) * self.noise_freq
            norm_point.append(norm_val + self.noise_offset[i])

        noise_val = 0.0
        if self.dims == 1:
            noise_val = noise.pnoise1(norm_point[0])
        elif self.dims == 2:
            noise_val = noise.pnoise2(norm_point[0], norm_point[1])
        elif self.dims == 3:
            noise_val = noise.pnoise3(norm_point[0], norm_point[1], norm_point[2])
        else:
            noise_val = noise.pnoise3(norm_point[0], norm_point[1], sum(norm_point[2:]))

        return gaussian_sum + (noise_val * self.noise_scale)
    
    def _calibrate_bounds(self, num_samples):
        """Samples the parameter space to estimate the global min and max."""
        # Generate random samples across the N-dimensional space
        samples = np.zeros((num_samples, self.dims))
        for i, p_name in enumerate(self.param_names):
            low, high = self.param_bounds[p_name]
            samples[:, i] = self.rng.uniform(low, high, num_samples)

        # Evaluate all samples to find empirical min/max
        raw_responses = np.array([self._raw_evaluate(pt) for pt in samples])
        
        self.raw_min = np.min(raw_responses)
        self.raw_max = np.max(raw_responses)
        
        # Prevent division by zero in case of a completely flat landscape
        if np.isclose(self.raw_min, self.raw_max):
            self.raw_max = self.raw_min + 1e-9


    def evaluate(self, params):
        """Queries the synthetic space and returns a scaled response."""
        try:
            point = np.array([params[k] for k in self.param_names])
        except KeyError as e:
            raise KeyError(f"Missing parameter in input: {e}")

        # 1. Get raw response
        raw_val = self._raw_evaluate(point)

        # 2. Scale to target bounds
        t_min, t_max = self.output_bounds
        scaled_val = t_min + ((raw_val - self.raw_min) * (t_max - t_min)) / (self.raw_max - self.raw_min)

        # 3. Clip the output
        # Because calibration is empirical, a real query might slightly exceed the 
        # sampled raw_min or raw_max. Clipping ensures strict adherence to output_bounds.
        return float(np.clip(scaled_val, t_min, t_max))