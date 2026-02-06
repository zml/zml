#%%
import torch

class BoxMullerGenerator:
    def __init__(self, seed, device="cpu"):
        self.seed = seed
        self.device = device
        self.counter = 0

    def _hash(self, positions):
        """
        A simple vectorized hash function (split-mix style) 
        to turn a sequence of numbers into pseudorandom noise.
        """
        x = positions + self.seed
        x = (x ^ (x >> 16)) * 0x45d9f3b
        x = (x ^ (x >> 16)) * 0x45d9f3b
        x = x ^ (x >> 16)
        # Normalize to [0, 1]
        return (x % 1000000).float() / 1000000.0

    def randn(self, shape):
        num_elements = torch.prod(torch.tensor(shape)).item()
        
        # Create a range of indices (the "counters")
        indices = torch.arange(self.counter, self.counter + num_elements, device=self.device)
        self.counter += num_elements # Increment for next call
        
        # Generate two sets of uniform noise for Box-Muller
        u1 = self._hash(indices)
        u2 = self._hash(indices + 0x9e3779b9) # Offset for second stream
        
        # Box-Muller Transform: Uniform -> Normal (Gaussian)
        # Standard formula: z = sqrt(-2 ln(u1)) * cos(2pi * u2)
        mag = torch.sqrt(-2.0 * torch.log(u1 + 1e-10))
        dist = mag * torch.cos(2.0 * 3.1415926535 * u2)
        
        return dist.view(shape)

class PolarGenerator:
    def __init__(self, seed, device="cpu"):
        self.seed = seed
        self.device = device
        self.counter = 0

    def _hash(self, positions):
        """
        Same vectorized split-mix hash as before.
        """
        x = positions + self.seed
        x = (x ^ (x >> 16)) * 0x45d9f3b
        x = (x ^ (x >> 16)) * 0x45d9f3b
        x = x ^ (x >> 16)
        # Normalize to [0, 1]
        return (x % 1000000).float() / 1000000.0

    def randn(self, shape):
        num_elements = torch.prod(torch.tensor(shape)).item()
        
        # Buffer to collect valid samples
        collected_samples = []
        count = 0
        
        # Rejection Sampling Loop
        # We loop until we have enough valid normal numbers
        while count < num_elements:
            # Estimate how many pairs we need to generate.
            # The area of unit circle is pi, square is 4. Acceptance rate is pi/4 (~0.785).
            # Each accepted pair yields 2 numbers. 
            # We add a buffer (+16) to minimize the chance of needing a second loop.
            needed = num_elements - count
            batch_size = int((needed / 2) * 1.3) + 16
            
            indices = torch.arange(self.counter, self.counter + batch_size, device=self.device)
            self.counter += batch_size
            
            # 1. Generate uniform noise in range [-1, 1]
            u = self._hash(indices) * 2.0 - 1.0
            v = self._hash(indices + 0x9e3779b9) * 2.0 - 1.0
            
            # 2. Calculate squared radius
            s = u**2 + v**2
            
            # 3. Rejection step: Keep only points inside the unit circle (and not 0)
            mask = (s > 0) & (s < 1)
            
            s_valid = s[mask]
            u_valid = u[mask]
            v_valid = v[mask]
            
            # 4. Polar Transform
            # Multiplier = sqrt(-2 ln(s) / s)
            # This replaces the sin/cos calculation
            multiplier = torch.sqrt(-2.0 * torch.log(s_valid) / s_valid)
            
            z1 = u_valid * multiplier
            z2 = v_valid * multiplier
            
            # Both z1 and z2 are valid normal variables
            batch_result = torch.cat([z1, z2])
            
            collected_samples.append(batch_result)
            count += batch_result.numel()
            
        # Concatenate batches and trim to exact requested size
        total_output = torch.cat(collected_samples)
        return total_output[:num_elements].view(shape)

if __name__ == "__main__":
    generator = BoxMullerGenerator(seed=42, device="cpu")
    samples = generator.randn((2, 3, 4))
    print(samples)

# %%
