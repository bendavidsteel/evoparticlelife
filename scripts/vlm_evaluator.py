"""VLM-based evaluation of life-likeness for particle simulations using vLLM.

Inspired by the ASAL paper (https://arxiv.org/abs/2412.17799) which uses
vision-language models to quantify previously qualitative phenomena.
"""

import os
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple
import base64
from io import BytesIO
from PIL import Image

from evo_particle_life import species_to_color
from render import draw_particles_2d_fast


class VLMLifeEvaluator:
    """Evaluates life-likeness of particle simulations using local VLM via vLLM."""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct", tensor_parallel_size: int = 1):
        """Initialize VLM evaluator with vLLM.

        Args:
            model_name: HuggingFace model name (vision-language model)
                        Default: Qwen3-VL-8B-Instruct (most recent open source VLM, Sep 2025)
                        Alternatives: "Qwen/Qwen3-VL-2B-Instruct" (smaller, faster)
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
        from vllm import LLM, SamplingParams

        self.model_name = model_name
        print(f"Loading VLM model: {model_name}...")

        # Initialize vLLM with vision support
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.7,
            trust_remote_code=True,
            max_model_len=4096,
        )

        self.sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=512,
            top_p=0.9,
        )

        print("VLM model loaded successfully!")

    def render_snapshot(self, positions, species, box_size, img_size=512):
        """Render a snapshot of the simulation state.

        Args:
            positions: (N, 2) particle positions
            species: (N, species_dims) species vectors
            box_size: Size of simulation box
            img_size: Output image size

        Returns:
            PIL Image
        """
        size = jnp.array([box_size, box_size]) if not hasattr(box_size, '__len__') else box_size
        colours = species_to_color(species)
        image_array = draw_particles_2d_fast(positions, colours, size, img_size=img_size)

        # Convert to PIL Image
        image_np = np.array(image_array * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def evaluate_life_likeness(
        self,
        positions,
        species,
        box_size,
        context: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate how life-like the current simulation state appears.

        Args:
            positions: Particle positions
            species: Species vectors
            box_size: Simulation box size
            context: Optional context about the simulation

        Returns:
            Dictionary with scores for different life-like qualities
        """
        # Render snapshot
        image = self.render_snapshot(positions, species, box_size)

        # Save temporarily for vLLM
        temp_path = Path("/tmp/vlm_eval_snapshot.png")
        image.save(temp_path)

        # Construct evaluation prompt
        prompt = self._construct_evaluation_prompt(context)

        # Query model with vision
        from vllm.multimodal.image import ImagePixelData

        # Prepare multimodal input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(temp_path)},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Generate response
        outputs = self.llm.chat(
            messages=messages,
            sampling_params=self.sampling_params,
        )

        response_text = outputs[0].outputs[0].text

        # Parse response
        scores = self._parse_evaluation_response(response_text)
        return scores

    def evaluate_trajectory(
        self,
        trajectory_snapshots: List[Tuple],
        box_size,
        num_snapshots: int = 4
    ) -> Dict[str, float]:
        """Evaluate life-likeness based on temporal dynamics.

        Args:
            trajectory_snapshots: List of (positions, species) tuples over time
            box_size: Simulation box size
            num_snapshots: Number of snapshots to evaluate

        Returns:
            Dictionary with temporal life-likeness scores
        """
        # Sample snapshots evenly from trajectory
        indices = np.linspace(0, len(trajectory_snapshots) - 1, num_snapshots, dtype=int)
        snapshots = [trajectory_snapshots[i] for i in indices]

        # Render snapshots
        images = [self.render_snapshot(pos, spec, box_size)
                 for pos, spec in snapshots]

        # Save images temporarily
        temp_paths = []
        for i, img in enumerate(images):
            temp_path = Path(f"/tmp/vlm_eval_frame_{i}.png")
            img.save(temp_path)
            temp_paths.append(str(temp_path))

        # Construct temporal evaluation prompt
        prompt = self._construct_temporal_prompt()

        # Prepare multimodal input with all images
        content = []
        for i, img_path in enumerate(temp_paths):
            content.append({"type": "image", "image": img_path})
            content.append({"type": "text", "text": f"**Frame {i+1}** (time step {indices[i]})\n"})

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        # Generate response
        outputs = self.llm.chat(
            messages=messages,
            sampling_params=self.sampling_params,
        )

        response_text = outputs[0].outputs[0].text

        # Parse response
        scores = self._parse_temporal_response(response_text)
        return scores

    def _construct_evaluation_prompt(self, context: Optional[str] = None) -> str:
        """Construct prompt for evaluating single snapshot."""
        base_prompt = """You are evaluating a particle-based artificial life simulation.
The image shows particles (colored dots) interacting in a 2D space. Different colors represent different species.

Please evaluate the following life-like qualities on a scale of 0-10:

1. **Organization** (0-10): How organized or structured do the particles appear?
   - 0 = completely random/uniform distribution
   - 10 = highly organized patterns, clear structures

2. **Complexity** (0-10): How complex are the visible patterns?
   - 0 = simple, homogeneous
   - 10 = intricate, multi-scale patterns

3. **Cohesion** (0-10): Do particles form coherent groups or clusters?
   - 0 = completely dispersed
   - 10 = strong, well-defined groups

4. **Diversity** (0-10): How diverse are the spatial patterns?
   - 0 = uniform, repetitive
   - 10 = rich variety of local configurations

5. **Life-likeness** (0-10): Overall, how "alive" does this system appear?
   - Consider: emergent patterns, organization, dynamic appearance
   - 0 = dead/static/random
   - 10 = highly reminiscent of biological or living systems

Please respond with ONLY the scores in this exact format:
organization: X
complexity: X
cohesion: X
diversity: X
life_likeness: X

Where X is a number from 0-10 (can include decimals like 7.5)."""

        if context:
            base_prompt += f"\n\nContext: {context}"

        return base_prompt

    def _construct_temporal_prompt(self) -> str:
        """Construct prompt for evaluating temporal dynamics."""
        return """You are evaluating the temporal dynamics of a particle-based artificial life simulation.
The images show sequential snapshots of the system over time.

Please evaluate the following temporal life-like qualities on a scale of 0-10:

1. **Persistence** (0-10): Do structures persist over time or constantly dissolve?
   - 0 = no persistence, complete chaos
   - 10 = stable structures maintained

2. **Dynamics** (0-10): Is there interesting motion and change?
   - 0 = static, no change
   - 10 = rich dynamics, continual evolution

3. **Emergence** (0-10): Do new patterns/structures emerge over time?
   - 0 = no emergence, stays the same
   - 10 = continuous emergence of new patterns

4. **Coherence** (0-10): Do changes appear coherent or random?
   - 0 = random, incoherent changes
   - 10 = coherent, purposeful-looking evolution

5. **Temporal life-likeness** (0-10): Overall life-like quality of temporal evolution
   - Consider: growth, adaptation, stable dynamics
   - 0 = dead/static/random
   - 10 = highly life-like temporal behavior

Please respond with ONLY the scores in this exact format:
persistence: X
dynamics: X
emergence: X
coherence: X
temporal_life_likeness: X

Where X is a number from 0-10 (can include decimals)."""

    def _parse_evaluation_response(self, response: str) -> Dict[str, float]:
        """Parse model's response into scores."""
        scores = {}
        for line in response.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_').replace('-', '_')
                try:
                    scores[key] = float(value.strip())
                except ValueError:
                    pass
        return scores

    def _parse_temporal_response(self, response: str) -> Dict[str, float]:
        """Parse temporal evaluation response."""
        return self._parse_evaluation_response(response)


def evaluate_simulation_snapshot(sim, evaluator: Optional[VLMLifeEvaluator] = None) -> Dict[str, float]:
    """Convenience function to evaluate current simulation state.

    Args:
        sim: ParticleLife simulation instance
        evaluator: VLMLifeEvaluator instance (creates one if None)

    Returns:
        Dictionary of life-likeness scores
    """
    if evaluator is None:
        evaluator = VLMLifeEvaluator()

    box_size = sim.size[0] if hasattr(sim.size, '__len__') else sim.size
    return evaluator.evaluate_life_likeness(
        sim.positions,
        sim.species,
        box_size,
        context=f"Particles: {sim.num_particles}, Species dim: {sim.species_dims}"
    )


if __name__ == "__main__":
    # Test the evaluator
    import jax.numpy as jnp
    from evo_particle_life import ParticleLife

    print("Testing vLLM Life Evaluator...")

    # Create small simulation
    sim = ParticleLife(
        num_particles=500,
        species_dims=2,
        size=jnp.array([2.0, 2.0]),
        n_dims=2,
        steps_per_frame=10,
    )

    # Run for a bit
    print("Running simulation...")
    for _ in range(50):
        sim.step()

    # Evaluate
    print("\nEvaluating life-likeness...")
    evaluator = VLMLifeEvaluator()
    scores = evaluate_simulation_snapshot(sim, evaluator)

    print("\nLife-likeness scores:")
    for key, value in scores.items():
        print(f"  {key}: {value:.1f}/10")
