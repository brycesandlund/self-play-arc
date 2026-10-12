from dataclasses import asdict, dataclass


@dataclass
class Config:
    size: int = 8
    colors: int = 11  # Ten ARC colors and X padding.
    puzzle_dim: int = 16
    instance_dim: int = 16
    board_width: int = 24
    transform_width: int = 12
    transform_steps: int = 2
    hyper_width: int = 64
    solver_width: int = 32
    solver_steps: int = 2
    support: int = 3
    puzzles: int = 4
    train_episodes: int = 2
    validation_episodes: int = 2
    inner_steps: int = 1
    solver_lr: float = 0.01
    solver_clip: float = 10.0
    generator_lr: float = 0.001
    generator_clip: float = 5.0
    target_probability: float = 0.8
    difficulty_weight: float = 1.0
    progress_weight: float = 1.0
    normalize_generator_terms: bool = True
    replay_capacity: int = 64
    replay_weight: float = 0.5
    augment: bool = True
    seed: int = 42

    def __post_init__(self):
        positive = (self.size, self.colors, self.puzzle_dim, self.instance_dim,
                    self.board_width, self.transform_width, self.transform_steps,
                    self.hyper_width, self.solver_width, self.solver_steps,
                    self.support, self.puzzles, self.train_episodes,
                    self.validation_episodes, self.inner_steps)
        if min(positive) < 1 or self.colors < 2:
            raise ValueError("Dimensions, episode counts and steps must be positive")
        if not 0 < self.target_probability < 1:
            raise ValueError("target_probability must be in (0, 1)")
        if min(self.solver_lr, self.generator_lr) <= 0:
            raise ValueError("Learning rates must be positive")
        if min(self.replay_capacity, self.replay_weight, self.difficulty_weight,
               self.progress_weight, self.solver_clip, self.generator_clip) < 0:
            raise ValueError("Weights, clips and replay capacity must be nonnegative")

    def to_dict(self):
        return asdict(self)

