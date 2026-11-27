"""Full LoRA save metadata: Kohya ss_* keys plus a sanitised reproducibility recipe.
No captions, tags, folder names, paths, user identity, or prompt content are stored.
"""
import json
import types

from fizgig.training import metadata as M


def _ds(directory, count, repeats=1, weight=1.0, reg=False, buckets=((512, 512),), **settings):
    values = {
        "image_directory": directory, "num_repeats": repeats, "loss_weight": weight, "is_reg": reg,
        "get_total_image_count": lambda: count,
        "batch_manager": types.SimpleNamespace(bucket_resos=list(buckets)),
    }
    values.update(settings)
    return types.SimpleNamespace(**values)


SECRET = "hf_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"


def _md(tmp_path):
    main = str(tmp_path / "data" / "private project folder")
    sub = "H:\\Stable_Diffusion\\Users\\PrivateName\\Close ups"
    reg = str(tmp_path / "private regularization")
    dss = [_ds(main, 20, repeats=2, weight=0.5, buckets=((768, 1024), (1024, 768)),
               resolution=(1024, 1024), batch_size=2, enable_bucket=True, bucket_no_upscale=True,
               caption_extension=".txt", cache_directory=str(tmp_path / "cache"),
               tag_shuffle=True, keep_first_tags=2, tag_dropout=10.0, caption_dropout=5.0,
               caption_variants=4, caption_variant_cap=8),
           _ds(sub, 5, repeats=3), _ds(reg, 10, reg=True)]
    md = M.training_metadata(
        seed=7, batch_size=1, grad_accum=2, epochs=10, epoch=4, steps=200, steps_per_epoch=50, max_steps=500,
        num_images=75, scheduler="cosine", warmup_steps=10, resolution=[1024, 1024],
        buckets=M.bucket_resolutions(dss), base_precision="int8",
        base_model="H:\\Stable_Diffusion\\models\\krea2_dev.safetensors", datasets=dss,
        training_method="global_replay", selective_block_check=False, validation_groups=3, best_epoch=3,
        best_val_loss=0.1234, current_lr=1e-4, started_at=1000.0, now=1600.0,
        training_spec={
            "network": {"type": "lora", "rank": 16, "alpha": 16, "ema_decay": 0.98},
            "optimizer": {"name": "Muon", "args": "weight_decay=0.01", "learning_rate": 2e-4,
                          "adaptive": True, "adaptive_min": 2e-4, "adaptive_max": 8e-4},
            "memory": {"blocks_to_swap": 15, "gradient_checkpointing": False},
            "execution": {"caption_dropout": 0.2, "batch_size": 1, "gradient_accumulation_steps": 2},
            "validation": {"holdout_images": 4, "early_stop": False},
            "private": {"username": "PrivateName", "tags": ["private tag"],
                        "caption": "a private caption", "source_path": str(tmp_path)},
        })
    return md, dss


def test_kohya_keys_and_training_config(tmp_path):
    md, dss = _md(tmp_path)
    for k in ("ss_seed", "ss_batch_size_per_device", "ss_gradient_accumulation_steps",
              "ss_num_epochs", "ss_epoch", "ss_steps", "ss_num_batches_per_epoch", "ss_max_train_steps",
              "ss_num_train_images", "ss_num_reg_images", "ss_num_datasets", "ss_num_reg_datasets",
              "ss_lr_scheduler", "ss_resolution", "ss_mixed_precision", "ss_base_precision",
              "ss_sd_model_name", "ss_dataset_dirs", "ss_reg_dataset_dirs", "ss_bucket_info",
              "ss_training_started_at", "ss_training_finished_at", M.TRAINING_CONFIG_KEY):
        assert k in md and isinstance(md[k], str), k
    assert md["ss_seed"] == "7"
    assert md["ss_resolution"] == "(1024, 1024)"
    assert md["ss_lr_scheduler"] == "cosine"
    assert md["ss_sd_model_name"] == "krea2_dev.safetensors"
    assert md["ss_num_reg_images"] == "10"
    dirs = json.loads(md["ss_dataset_dirs"])
    assert dirs == {
        "dataset_01": {"n_repeats": 2, "img_count": 20},
        "dataset_02": {"n_repeats": 3, "img_count": 5},
    }
    assert json.loads(md["ss_reg_dataset_dirs"]) == {
        "reg_dataset_01": {"n_repeats": 1, "img_count": 10}
    }
    cfg = json.loads(md[M.TRAINING_CONFIG_KEY])
    assert cfg["training_method"] == "global_replay"
    assert cfg["selective_block_check"] is False
    layout = cfg["dataset_layout"]["dataset_01"]
    assert layout["weight"] == 0.5 and layout["repeats"] == 2 and layout["img_count"] == 20
    assert layout["batch_size"] == 2 and layout["enable_bucket"] is True
    assert layout["bucket_no_upscale"] is True and layout["cache_enabled"] is True
    assert layout["resolution"] == [1024, 1024] and layout["caption_extension"] == ".txt"
    assert layout["tag_shuffle"] is True and layout["keep_first_tags"] == 2
    assert layout["tag_dropout"] == 10.0 and layout["caption_dropout"] == 5.0
    assert layout["caption_variants"] == 4 and layout["caption_variant_cap"] == 8
    assert cfg["validation_groups"] == 3 and cfg["epoch"] == 4 and cfg["best_epoch"] == 3
    assert cfg["best_val_loss"] == 0.1234 and cfg["current_lr"] == 1e-4
    assert cfg["training_seconds"] == 600.0
    assert cfg["trained_resolutions"][:2] == ["768x1024", "1024x768"]
    assert cfg["training_spec"]["network"]["rank"] == 16
    assert cfg["training_spec"]["optimizer"]["adaptive_max"] == 8e-4
    assert cfg["training_spec"]["memory"]["blocks_to_swap"] == 15
    assert cfg["training_spec"]["execution"]["caption_dropout"] == 0.2


def test_no_paths_tags_captions_or_user_info_leak(tmp_path):
    md, _ = _md(tmp_path)
    blob = json.dumps(md)
    for bad in (str(tmp_path), "Stable_Diffusion", "private project folder", "PrivateName",
                "Close ups", "private tag", "private caption", "source_path", "username"):
        assert bad not in blob, bad
    md2 = M.training_metadata(base_model=f"/x/{SECRET}.safetensors", training_method=SECRET,
                              datasets=[_ds(f"/home/u/{SECRET}", 1)])
    assert SECRET not in json.dumps(md2)


def test_existing_build_metadata_is_unchanged():
    md = M.build_metadata(None, "krea2", 0, title="t", trigger_phrase="zx")
    assert md["modelspec.architecture"] == "Krea-2/lora" and md["modelspec.title"] == "t"
    assert M.TRAINING_CONFIG_KEY not in md
