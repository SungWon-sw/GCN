import argparse
import copy
import os

import optuna
import torch
from ogb.graphproppred import Evaluator

from dataset import prepare_dataset, make_loaders, apply_pe
from train import (
    build_model_and_optimizer,
    run_training,
    evaluate,
    load_config,
    select_device,
)


def parse_args():
    p = argparse.ArgumentParser(description="Optuna hyperparameter search for src/train.py (ogbg-ppa, GCN+centroid-VN)")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--epochs", type=int, default=30,
                   help="트라이얼당 학습 에폭 수 (전체 재학습 400 에폭과 별개). cosine 스케줄이 이 길이로 압축되고 "
                        "warmup 은 epochs//10 으로 비례 축소됨")
    p.add_argument("--timeout", type=int, default=None, help="탐색 전체 제한 시간(초), 기본 무제한")
    # 탐색 공간이 바뀌었으므로 옛 study(v1)와 이름을 분리 (Optuna 는 같은 이름에 분포가 다르면 에러)
    p.add_argument("--study-name", default="ppa_gcnplus_centroid_vn_v2")
    p.add_argument("--storage", default="sqlite:///optuna_ppa.db", help="중단 후 이어서 탐색하려면 sqlite 파일 유지")
    p.add_argument("--seed", type=int, default=42, help="Optuna 샘플러 시드")
    p.add_argument("--retrain-best", action="store_true",
                    help="탐색 종료 후 best_params로 cfg['train']['epochs']만큼 전체 재학습 + 테스트 평가")
    p.add_argument("--best-config-out", default="configs/best_config.yaml")
    p.add_argument("--model-out", default="best_model_optuna.pt",
                    help="--retrain-best 저장 파일명. train.py의 'best_model.pt'와 겹치지 않도록 기본값을 다르게 둠")
    return p.parse_args()


def apply_params(cfg, p):
    """Optuna 파라미터 dict 를 cfg 에 반영. p 에 없는 키는 cfg 의 기존 값을 유지
    (옛 study 의 best_params 처럼 새 옵션이 빠져 있어도 동작)."""
    for k in ('lr', 'weight_decay', 'drop_ratio', 'emb_dim', 'batch_size'):
        if k in p:
            cfg['train'][k] = p[k]
    for k in ('num_layers', 'num_sublayers', 'edge_attr', 'norm', 'ffn', 'ffn_ratio', 'backbone_residual'):
        if k in p:
            cfg['model'][k] = p[k]
    pe = cfg['model'].setdefault('pos_encoding', {})
    if 'pos_encoding' in p:
        pe['enabled'] = p['pos_encoding']
    if 'pe_dim' in p:
        pe['dim'] = p['pe_dim']


def make_objective(base_cfg, cached_dataset, split_idx, num_classes, device, epochs):
    evaluator = Evaluator(base_cfg['data']['dataset_name'])

    def objective(trial):
        # GCN+ 논문(Sec.4/A.3) 탐색 공간 주변으로 좁힘: lr ∈ [1e-4,1e-3], dropout ∈ [0.05,0.3],
        # BN/FFN/PE on-off. 고정(config 값 유지): optimizer=adamw, post_mp_layers=2,
        # edge_attr=True, backbone_residual=True (ppa 엣지 특징/잔차는 구조의 핵심이라 탐색 제외 -> 탐색 차원 축소).
        params = {
            'lr': trial.suggest_float('lr', 1e-4, 1e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
            'drop_ratio': trial.suggest_float('drop_ratio', 0.05, 0.3, step=0.05),
            'emb_dim': trial.suggest_categorical('emb_dim', [256, 384, 512]),
            'num_layers': trial.suggest_int('num_layers', 3, 6),
            'num_sublayers': trial.suggest_int('num_sublayers', 1, 2),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64]),
            # --- GCN+ 옵션 ---
            'norm': trial.suggest_categorical('norm', ['batch', 'layer']),
            'ffn': trial.suggest_categorical('ffn', [True, False]),
            'pos_encoding': trial.suggest_categorical('pos_encoding', [True, False]),
        }
        if params['ffn']:
            params['ffn_ratio'] = trial.suggest_categorical('ffn_ratio', [1, 2, 4])
        if params['pos_encoding']:
            params['pe_dim'] = trial.suggest_categorical('pe_dim', [8, 16, 32])
        cfg = copy.deepcopy(base_cfg)
        apply_params(cfg, params)
        cfg['train']['warmup_epochs'] = max(1, epochs // 10)   # 400ep/warmup10 비율(2.5%)보다 짧은 탐색용으로 10% 유지

        train_loader, val_loader, _ = make_loaders(
            apply_pe(cached_dataset, cfg), split_idx,
            batch_size=cfg['train']['batch_size'],
            num_workers=cfg['train']['num_workers']
        )

        model, optimizer, criterion = build_model_and_optimizer(cfg, num_classes, device)

        trial_model_path = f"optuna_trial_{trial.number}.pt"

        def on_epoch_end(epoch, val_acc):
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        try:
            best_acc = run_training(
                cfg, model, optimizer, criterion, train_loader, val_loader,
                evaluator, device, epochs=epochs,
                model_name=trial_model_path, on_epoch_end=on_epoch_end
            )
        finally:
            if os.path.exists(trial_model_path):
                os.remove(trial_model_path)

        return best_acc

    return objective


def main():
    args = parse_args()
    base_cfg = load_config(args.config)

    device = select_device(base_cfg)
    print('Using device:', device)

    # 무거운 부분(원본 로드 + VN 캐시 래핑)은 탐색 전체에서 딱 한 번만 수행.
    # 위치 인코딩(data.pe)은 트라이얼마다 켜짐/차원이 달라지므로 여기선 끄고, 트라이얼에서 apply_pe 로 붙임.
    no_pe_cfg = copy.deepcopy(base_cfg)
    no_pe_cfg.setdefault('model', {}).setdefault('pos_encoding', {})['enabled'] = False
    cached_dataset, split_idx, num_tasks, num_classes = prepare_dataset(no_pe_cfg)
    print(f'num_tasks: {num_tasks}, num_classes: {num_classes}')

    objective = make_objective(
        base_cfg, cached_dataset, split_idx, num_classes, device, epochs=args.epochs
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=8),
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    print("\n==> Optuna 탐색 완료")
    print(f"Best trial #{study.best_trial.number} | Val Acc: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    best_cfg = copy.deepcopy(base_cfg)
    apply_params(best_cfg, study.best_params)

    import yaml
    os.makedirs(os.path.dirname(args.best_config_out) or ".", exist_ok=True)
    with open(args.best_config_out, 'w', encoding='utf-8') as f:
        yaml.safe_dump(best_cfg, f, allow_unicode=True, sort_keys=False)
    print(f"Best config 저장: {args.best_config_out}")

    if args.retrain_best:
        print(f"\n==> best_params로 {best_cfg['train']['epochs']} 에폭 전체 재학습 시작")
        train_loader, val_loader, test_loader = make_loaders(
            apply_pe(cached_dataset, best_cfg), split_idx,
            batch_size=best_cfg['train']['batch_size'],
            num_workers=best_cfg['train']['num_workers']
        )
        model, optimizer, criterion = build_model_and_optimizer(best_cfg, num_classes, device)
        evaluator = Evaluator(best_cfg['data']['dataset_name'])
        model_name = args.model_out
        run_training(
            best_cfg, model, optimizer, criterion, train_loader, val_loader,
            evaluator, device, epochs=best_cfg['train']['epochs'], model_name=model_name
        )
        state_dict = torch.load(model_name, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        test_acc = evaluate(model, test_loader, evaluator, device)
        print(f'==> Final Test Accuracy (best params, full retrain): {test_acc:.4f}')


if __name__ == "__main__":
    main()
