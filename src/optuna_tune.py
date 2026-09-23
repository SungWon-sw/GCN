import argparse
import copy
import os

import optuna
import torch
from ogb.graphproppred import Evaluator

from dataset import prepare_dataset, make_loaders
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
    p.add_argument("--epochs", type=int, default=20, help="트라이얼당 학습 에폭 수 (전체 재학습과 별개, 짧게 탐색)")
    p.add_argument("--timeout", type=int, default=None, help="탐색 전체 제한 시간(초), 기본 무제한")
    p.add_argument("--study-name", default="ppa_gcn_centroid_vn")
    p.add_argument("--storage", default="sqlite:///optuna_ppa.db", help="중단 후 이어서 탐색하려면 sqlite 파일 유지")
    p.add_argument("--seed", type=int, default=42, help="Optuna 샘플러 시드")
    p.add_argument("--retrain-best", action="store_true",
                    help="탐색 종료 후 best_params로 cfg['train']['epochs']만큼 전체 재학습 + 테스트 평가")
    p.add_argument("--best-config-out", default="configs/best_config.yaml")
    p.add_argument("--model-out", default="best_model_optuna.pt",
                    help="--retrain-best 저장 파일명. train.py의 'best_model.pt'와 겹치지 않도록 기본값을 다르게 둠")
    return p.parse_args()


def make_objective(base_cfg, cached_dataset, split_idx, num_classes, device, epochs):
    evaluator = Evaluator(base_cfg['data']['dataset_name'])

    def objective(trial):
        cfg = copy.deepcopy(base_cfg)
        cfg['train']['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        cfg['train']['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        cfg['train']['drop_ratio'] = trial.suggest_float('drop_ratio', 0.35, 0.6)
        cfg['train']['emb_dim'] = trial.suggest_categorical('emb_dim', [128, 200, 300, 384])
        cfg['model']['num_layers'] = trial.suggest_int('num_layers', 2, 5)
        cfg['model']['num_sublayers'] = trial.suggest_int('num_sublayers', 1, 4)
        cfg['train']['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

        train_loader, val_loader, _ = make_loaders(
            cached_dataset, split_idx,
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

    # 무거운 부분(원본 로드 + VN 캐시 래핑)은 탐색 전체에서 딱 한 번만 수행
    cached_dataset, split_idx, num_tasks, num_classes = prepare_dataset(base_cfg)
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
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    print("\n==> Optuna 탐색 완료")
    print(f"Best trial #{study.best_trial.number} | Val Acc: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    best_cfg = copy.deepcopy(base_cfg)
    best_cfg['train'].update({
        'lr': study.best_params['lr'],
        'weight_decay': study.best_params['weight_decay'],
        'drop_ratio': study.best_params['drop_ratio'],
        'emb_dim': study.best_params['emb_dim'],
        'batch_size': study.best_params['batch_size'],
    })
    best_cfg['model']['num_layers'] = study.best_params['num_layers']
    best_cfg['model']['num_sublayers'] = study.best_params['num_sublayers']

    import yaml
    os.makedirs(os.path.dirname(args.best_config_out) or ".", exist_ok=True)
    with open(args.best_config_out, 'w', encoding='utf-8') as f:
        yaml.safe_dump(best_cfg, f, allow_unicode=True, sort_keys=False)
    print(f"Best config 저장: {args.best_config_out}")

    if args.retrain_best:
        print(f"\n==> best_params로 {best_cfg['train']['epochs']} 에폭 전체 재학습 시작")
        train_loader, val_loader, test_loader = make_loaders(
            cached_dataset, split_idx,
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
