"""
Data augmentation dut to lack of training samples
"""
import Augmentor

p = Augmentor.Pipeline("./data/train/class_1")
p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.random_distortion(probability=0.3, grid_height=2, grid_width=2, magnitude=2)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.flip_top_bottom(probability=0.5)
p.sample(700)

