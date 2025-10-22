#!/usr/bin/env bash
set +e  # 실패해도 종료하지 않음

CUDA_VISIBLE_DEVICES=0 python fleet_v3/run.py --graph_size 40 --baseline rollout --run_name 'hcvrp40_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 20 --spatial_tau 0.05 || true

CUDA_VISIBLE_DEVICES=0 python fleet_v3/run.py --graph_size 60 --baseline rollout --run_name 'hcvrp60_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 30 --spatial_tau 0.05 || true