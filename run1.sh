#!/usr/bin/env bash
set +e  # 실패해도 종료하지 않음

python fleet_v3/run.py --graph_size 40 --baseline rollout --run_name 'hcvrp40_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 20 --spatial_tau 0.05 || true

python fleet_v3/run.py --graph_size 60 --baseline rollout --run_name 'hcvrp60_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 30 --spatial_tau 0.05 || true

python fleet_v3/run.py --graph_size 80 --baseline rollout --run_name 'hcvrp80_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 40 --spatial_tau 0.05 || true

python fleet_v3/run.py --graph_size 100 --baseline rollout --run_name 'hcvrp100_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 50 --spatial_tau 0.05 || true

python fleet_v3/run.py --graph_size 120 --baseline rollout --run_name 'hcvrp120_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 60 --spatial_tau 0.05 || true

python fleet_v5/run.py --graph_size 80 --baseline rollout --run_name 'hcvrp80_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 40 --spatial_tau 0.05 || true

python fleet_v5/run.py --graph_size 100 --baseline rollout --run_name 'hcvrp100_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 50 --spatial_tau 0.05 || true

python fleet_v5/run.py --graph_size 120 --baseline rollout --run_name 'hcvrp120_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 60 --spatial_tau 0.05 || true

python fleet_v5/run.py --graph_size 140 --baseline rollout --run_name 'hcvrp140_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 70 --spatial_tau 0.05 || true

python fleet_v5/run.py --graph_size 160 --baseline rollout --run_name 'hcvrp160_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 80 --spatial_tau 0.05 || true

python fleet_v3/run.py --graph_size 80 --baseline rollout --run_name 'hcvrp80_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 32 --spatial_tau 0.05 || true

python fleet_v5/run.py --graph_size 100 --baseline rollout --run_name 'hcvrp100_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 40 --spatial_tau 0.05 || true

python fleet_v3/run.py --graph_size 80 --baseline rollout --run_name 'hcvrp80_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 48 --spatial_tau 0.05 || true

python fleet_v5/run.py --graph_size 100 --baseline rollout --run_name 'hcvrp100_rollout_slate' --obj min-sum --spatial_encoder --spatial_window 60 --spatial_tau 0.05 || true