"""Train a LightZero MCTS agent on the trading env.

python -m rltrading.train_lz --algo stochastic_muzero --sentiment finbert --seed 0 --max_env_step 200000
"""
import argparse
import json
import os
import time

from easydict import EasyDict

from rltrading import config as C
from rltrading.data.features import make_arrays

ALGOS = ["muzero", "efficientzero", "gumbel_muzero", "stochastic_muzero", "sampled_muzero",
         "sampled_efficientzero", "unizero"]


def build_cfg(a):
    use_sent = a.sentiment != "none"
    arrays = str(make_arrays(a.sentiment if use_sent else None, use_sentiment=use_sent, use_forum=a.forum))
    meta = json.load(open(arrays.replace(".npz", ".json")))
    obs_dim = len(meta["cols"]) + 1
    continuous = a.algo.startswith("sampled")
    positions = (0.0, 1.0) if a.long_only else (-1.0, 0.0, 1.0)
    n_act = 1 if continuous else len(positions)
    exp = os.path.join(a.exp_root, f"{a.algo}_{a.sentiment}{'_forum' if a.forum else ''}"
                       f"{'_long' if a.long_only else ''}{a.tag}_seed{a.seed}")
    env = dict(
        arrays_path=arrays, episode_len=a.episode_len, cost_bps=a.cost_bps, positions=positions,
        continuous=continuous, long_only=a.long_only, reward_scale=1000.0,
        collector_env_num=a.collectors, evaluator_env_num=a.evaluators, n_evaluator_episode=a.evaluators,
        manager=dict(shared_memory=False),
    )
    model = dict(observation_shape=obs_dim, action_space_size=n_act, model_type="mlp",
                 latent_state_dim=a.latent, self_supervised_learning_loss=True,
                 discrete_action_encoding_type="one_hot", norm_type="LN",
                 reward_support_range=(-60., 61., 1.), value_support_range=(-60., 61., 1.))
    policy = dict(
        model=model, model_path=None, cuda=False, device="cpu", env_type="not_board_games",
        game_segment_length=50, batch_size=a.batch_size, optim_type="AdamW",
        learning_rate=a.lr, piecewise_decay_lr_scheduler=False, weight_decay=1e-4,
        num_simulations=a.sims, reanalyze_ratio=0.0, n_episode=a.collectors,
        eval_freq=a.eval_freq, replay_buffer_size=int(1e6), collector_env_num=a.collectors,
        evaluator_env_num=a.evaluators, discount_factor=a.gamma, td_steps=a.td_steps,
        num_unroll_steps=5, ssl_loss_weight=2, grad_clip_value=5, replay_ratio=a.replay_ratio,
        update_per_collect=None,
        learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=a.ckpt_every))),
    )
    if a.algo in ("muzero", "gumbel_muzero", "stochastic_muzero", "efficientzero"):
        # default SSL projection head (1024 wide) dominated CPU training time; shrink it ~4x
        model.update(proj_hid=256, proj_out=256, pred_hid=128, pred_out=256)
    if a.algo == "stochastic_muzero":
        model.update(chance_space_size=a.chance)
        policy.update(use_ture_chance_label_in_chance_encoder=False)
    if a.algo == "gumbel_muzero":
        policy.update(max_num_considered_actions=n_act)
    if a.algo == "efficientzero":
        model.update(lstm_hidden_size=a.latent)
    if continuous:
        model.update(continuous_action_space=True, num_of_sampled_actions=a.K, sigma_type="conditioned")
        policy.update(cos_lr_scheduler=False)
        if a.algo == "sampled_efficientzero":
            model.update(lstm_hidden_size=a.latent)
    if a.algo == "unizero":
        model.pop("latent_state_dim")
        model.pop("reward_support_range"); model.pop("value_support_range")
        model["world_model_cfg"] = dict(
            final_norm_option_in_obs_head="LayerNorm", final_norm_option_in_encoder="LayerNorm",
            predict_latent_loss_type="mse", max_blocks=10, max_tokens=20, context_length=8,
            context_length_for_recurrent=8, device="cpu", action_space_size=n_act, num_layers=2,
            num_heads=2, embed_dim=64, env_num=max(a.collectors, a.evaluators),
            collector_env_num=a.collectors, evaluator_env_num=a.evaluators, obs_type="vector",
            norm_type="LN", rotary_emb=True)
        policy.update(use_wandb=False, use_augmentation=False, target_update_freq=100, learning_rate=1e-4)
    main = EasyDict(dict(exp_name=exp, env=env, policy=policy))
    create = EasyDict(dict(
        env=dict(type="trading_lightzero", import_names=["rltrading.envs.lz_env"]),
        env_manager=dict(type=a.env_manager),
        policy=dict(type=a.algo, import_names=[f"lzero.policy.{a.algo}"]),
    ))
    return main, create


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="muzero", choices=ALGOS)
    p.add_argument("--sentiment", default="none", choices=["none", "finbert", "nosible"])
    p.add_argument("--forum", action="store_true")
    p.add_argument("--long_only", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_env_step", type=int, default=200_000)
    p.add_argument("--sims", type=int, default=25)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--chance", type=int, default=16)
    p.add_argument("--latent", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--td_steps", type=int, default=5)
    p.add_argument("--replay_ratio", type=float, default=0.25)
    p.add_argument("--episode_len", type=int, default=128)
    p.add_argument("--cost_bps", type=float, default=5.0)
    p.add_argument("--collectors", type=int, default=8)
    p.add_argument("--evaluators", type=int, default=5)
    p.add_argument("--eval_freq", type=int, default=2000)
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--env_manager", default="base", choices=["base", "subprocess"])
    p.add_argument("--exp_root", default="exp")  # LightZero prefixes "./": keep relative, run from ~/rlt
    p.add_argument("--tag", default="")
    p.add_argument("--threads", type=int, default=3)
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    import torch
    torch.set_num_threads(a.threads)
    main_cfg, create_cfg = build_cfg(a)
    t0 = time.time()
    if a.algo == "unizero":
        from lzero.entry import train_unizero
        train_unizero([main_cfg, create_cfg], seed=a.seed, max_env_step=a.max_env_step)
    else:
        from lzero.entry import train_muzero
        train_muzero([main_cfg, create_cfg], seed=a.seed, max_env_step=a.max_env_step)
    os.makedirs(main_cfg.exp_name, exist_ok=True)
    with open(os.path.join(main_cfg.exp_name, "run_args.json"), "w") as f:
        json.dump({**vars(a), "wall_sec": time.time() - t0}, f, indent=1)
    print("DONE", main_cfg.exp_name, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
