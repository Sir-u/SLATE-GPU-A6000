#!/usr/bin/env bash
set +e  # 실패해도 종료하지 않음

CUDA_VISIBLE_DEVICES=2 python fleet_v3/run.py --graph_size 120 --baseline rollout --run_name 'hcvrp120_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 60 --spatial_tau 0.05 || true

CUDA_VISIBLE_DEVICES=2 python fleet_v5/run.py --graph_size 80 --baseline rollout --run_name 'hcvrp80_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 40 --spatial_tau 0.05 || true