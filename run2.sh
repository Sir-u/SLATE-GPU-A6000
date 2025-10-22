#!/usr/bin/env bash
set +e  # 실패해도 종료하지 않음

CUDA_VISIBLE_DEVICES=1 python fleet_v3/run.py --graph_size 80 --baseline rollout --run_name 'hcvrp80_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 40 --spatial_tau 0.05 || true

CUDA_VISIBLE_DEVICES=1 python fleet_v3/run.py --graph_size 100 --baseline rollout --run_name 'hcvrp100_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 50 --spatial_tau 0.05 || true