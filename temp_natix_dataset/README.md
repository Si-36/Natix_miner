---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: id
    dtype: int64
  - name: width
    dtype: int64
  - name: height
    dtype: int64
  - name: license
    dtype: int64
  - name: flickr_url
    dtype: float64
  - name: coco_url
    dtype: float64
  - name: date_captured
    dtype: int64
  - name: gps
    dtype: string
  - name: city_name
    dtype: string
  - name: scene_description
    dtype: string
  - name: video_info.frame_id
    dtype: string
  - name: video_info.seq_id
    dtype: int64
  - name: video_info.vid_id
    dtype: string
  - name: scene_level_tags.daytime
    dtype: string
  - name: scene_level_tags.scene_environment
    dtype: string
  - name: scene_level_tags.travel_alteration
    dtype: string
  - name: scene_level_tags.weather
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 12891129770.855
    num_examples: 6251
  - name: test
    num_bytes: 2128265480.026
    num_examples: 2298
  download_size: 10524103439
  dataset_size: 15019395250.881
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
