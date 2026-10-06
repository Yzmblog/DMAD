# VBench Text-to-Video Evaluation

## 0. Prerequisites

```bash
pip install vbench

# detectron2 (required by some VBench dimensions)
pip install detectron2@git+https://github.com/facebookresearch/detectron2.git
```

## 1. Prompt JSON

`prompts.json` contains the 944 VBench-T2V prompts with GPT-4o-augmented
versions and VBench `auxiliary_info` (object labels, colors, spatial
relationships, etc.) needed for semantic evaluation dimensions.

Format:

```json
[
  {
    "prompt": "a bird and a cat",
    "dimensions": ["multiple_objects"],
    "augmented_prompt": "In a sunlit garden, a sleek black cat ...",
    "auxiliary_info": {"multiple_objects": {"object": "bird and cat"}}
  }
]
```

## 2. Generate Videos

Use [experiments/dmad/sample.sh](../../experiments/dmad/sample.sh) with `NUM_SAMPLES=5` and this `prompts.json` (see the
[README](../../README.md#evaluation)). It writes

```
<output_dir>/
  In a still frame, a stop sign-0.mp4
  ...
  vbench_eval_info.json   # VBench-compatible JSON with prompts, dimensions, auxiliary_info, video_list
```

Prompts whose videos already exist on disk are skipped. `vbench_eval_info.json` is a drop-in replacement for
`VBench_full_info.json`, with `video_list` already populated and `auxiliary_info` for all 16 dimensions.

## 3. Evaluate with VBench

Use the generated `vbench_eval_info.json` as `--full_json_dir` in VBench
standard mode. This supports all 16 dimensions including semantic ones
(object_class, color, etc.) that require auxiliary metadata:

### Evaluate all 16 dimensions

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  # for compatibility with PyTorch>=2.6.0
export VBENCH_CACHE_DIR=~/.cache/vbench

VIDEO_DIR=<output_dir>
RESULT_DIR=${VIDEO_DIR}_eval_results
EVAL_JSON=$VIDEO_DIR/vbench_eval_info.json

DIMS=(
  subject_consistency background_consistency temporal_flickering
  motion_smoothness dynamic_degree aesthetic_quality imaging_quality
  object_class multiple_objects human_action color
  spatial_relationship scene appearance_style
  overall_consistency temporal_style
)

NGPUS=8

for dim in "${DIMS[@]}"; do
    echo "=== Evaluating: $dim ==="
    vbench evaluate \
        --ngpus=$NGPUS \
        --videos_path "$VIDEO_DIR" \
        --dimension "$dim" \
        --output_path "$RESULT_DIR" \
        --full_json_dir "$EVAL_JSON"
done
```

## 4. Compute Final Scores

```bash
python evaluation/vbench_text2video/cal_scores.py --result_dir "$RESULT_DIR"
```

This prints all 16 dimension scores (raw + normalized), then the aggregated
Quality Score, Semantic Score, and Total Score:

```
Dimension                         Raw     Norm Weight Group
----------------------------------------------------------------
  subject consistency           0.9543   0.9459    1.0 Quality
  background consistency        0.9712   0.9611    1.0 Quality
  ...
  overall consistency           0.2341   0.6432    1.0 Semantic

  Quality Score                 0.8234   (weighted avg of 7 quality dims)
  Semantic Score                0.7456   (weighted avg of 9 semantic dims)
  Total Score                   0.8078   (quality*4 + semantic*1) / 5
```

| Aggregate | Dimensions |
|-----------|-----------|
| **Quality Score** | subject_consistency, background_consistency, temporal_flickering, motion_smoothness, aesthetic_quality, imaging_quality, dynamic_degree |
| **Semantic Score** | object_class, multiple_objects, human_action, color, spatial_relationship, scene, appearance_style, temporal_style, overall_consistency |
| **Total Score** | Weighted average of Quality and Semantic scores |
