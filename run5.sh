#!/usr/bin/env bash
set +e  # 실패해도 종료하지 않음

CUDA_VISIBLE_DEVICES=4 python fleet_v5/run.py --graph_size 140 --baseline rollout --run_name 'hcvrp140_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 70 --spatial_tau 0.05 || true

CUDA_VISIBLE_DEVICES=4 python fleet_v5/run.py --graph_size 160 --baseline rollout --run_name 'hcvrp160_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 80 --spatial_tau 0.05 || true