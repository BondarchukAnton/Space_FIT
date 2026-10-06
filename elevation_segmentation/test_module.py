"""Проверки масок, функций потерь, окон и восстановления обучения."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn
from .dataset import load_records, inspect_dataset, ElevationDataset
from .losses import ElevationLoss
from .inference import predict_probability, ElevationPredictor
from .model import create_model, freeze_batchnorm
from .utils import confusion, scores
from . import train

torch.set_num_threads(2)


def fixture(root):
    """Создает синтетический тестовый датасет и изолированную тестовую конфигурацию.

    Все пути, включая dataset_root и output_dir, направляются во временную директорию.
    Параметры pretrained_path и resume_training сбрасываются для исключения зависимостей
    от пользовательской рабочей конфигурации или каталога runs.
    """
    root = Path(root)
    rows = []
    for split, count in [('train', 3), ('val', 1)]:
        for i in range(count):
            name = f'{split}_{i}'
            rgb = np.full((64, 96, 3), 40 + i * 30 + (10 if split == 'val' else 0), np.uint8)
            rgb[:, :, 0] = np.arange(96, dtype=np.uint8)[None, :] + i * 20
            mask = np.zeros((64, 96), np.uint8)
            if i != 1:
                mask[15:45, 20:50] = 1
            mask[:3] = 255
            for folder, a in [('images', rgb), ('masks', mask)]:
                path = root / split / folder / f'{name}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(a).save(path)
            rows.append({
                'split': split,
                'name': name,
                'location_id': name,
                'image': f'{split}/images/{name}.png',
                'mask': f'{split}/masks/{name}.png',
                'group': 'mixed' if i != 1 else 'background',
            })
    (root / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    cfg = json.loads(Path(__file__).with_name('config.json').read_text())
    output_dir = root / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.update(
        dataset_root=str(root),
        output_dir=str(output_dir),
        pretrained_path=None,
        same_num_classes=True,
        resume_training=False,
        crop_size=64,
        tile_size=64,
        overlap=16,
        epochs=2,
        crops_per_image=1,
        batch_size=1,
        freeze_batchnorm=True,
        num_workers=0,
        accumulation_steps=2,
        save_every_steps=1,
        encoder_weights=None,
        device='cpu',
        amp=False,
    )
    return cfg


class TinyModel(nn.Module):
    """Небольшая модель с Dropout для проверки случайного состояния при resume."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 3, padding=1)
        self.decoder = nn.Sequential(nn.ReLU(), nn.Dropout(.3), nn.Conv2d(4, 1, 1))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ModuleTests(unittest.TestCase):
    def test_ignore_loss_and_background(self):
        loss = ElevationLoss()
        logits = torch.zeros((1, 1, 8, 8), requires_grad=True)
        target = torch.zeros((1, 8, 8), dtype=torch.long)
        target[:, :2] = 255
        a = loss(logits, target)
        a.backward()
        self.assertTrue(torch.isfinite(a))
        self.assertEqual(logits.grad[:, :, :2].abs().sum(), 0)
        self.assertGreater(logits.grad[:, :, 2:].abs().sum(), 0)
        changed = logits.detach().clone()
        changed[:, :, :2] = 100
        torch.testing.assert_close(loss(changed, target), a)
        target.fill_(255)
        self.assertEqual(loss(logits, target), 0)

    def test_metrics_ignore(self):
        cm = confusion(np.array([[.9, .8, .1, .1]]), np.array([[1, 0, 1, 255]]), .5)
        self.assertEqual(cm.tolist(), [[0, 1], [1, 1]])
        self.assertAlmostEqual(scores(cm)['iou_elevation'], 1 / 3)

    def test_tiles_and_small_image(self):
        class Zero(nn.Module):
            def forward(self, x):
                return x[:, :1] * 0

        for shape in [(19, 23), (105, 149), (64, 64), (1080, 1920)]:
            rgb = np.zeros((*shape, 3), np.uint8)
            p = predict_probability(Zero(), rgb, 'cpu', 512 if shape[0] > 500 else 64, 16, False)
            self.assertEqual(p.shape, shape)
            np.testing.assert_allclose(p, .5, atol=1e-6)

    def test_dataset_determinism_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = fixture(root)
            rows = load_records(root)
            report, digest = inspect_dataset(rows)
            self.assertEqual(report['train']['images'], 3)
            dataset = ElevationDataset(rows['train'], cfg)
            a, m = dataset[0]
            b, n = dataset[0]
            torch.testing.assert_close(a, b)
            torch.testing.assert_close(m, n)
            self.assertTrue(set(m.unique().tolist()) <= {0, 1, 255})
            dataset.epoch = 1
            c, _ = dataset[0]
            self.assertFalse(torch.equal(a, c))
            Image.fromarray(np.full((64, 96), 2, np.uint8)).save(rows['train'][0]['mask'])
            with self.assertRaises(ValueError):
                inspect_dataset(rows)

    def test_real_smp_forward_backward_and_predictor(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            cfg = fixture(tmp_root)
            cfg.update(encoder_weights=None, tile_size=64, overlap=16, amp=False)
            model = create_model(cfg, False)
            model.train()
            freeze_batchnorm(model)
            logits = model(torch.rand(1, 3, 64, 64))
            self.assertEqual(tuple(logits.shape), (1, 1, 64, 64))
            ElevationLoss()(logits, torch.zeros((1, 64, 64), dtype=torch.long)).backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))
            path = tmp_root / 'weights.pt'
            torch.save({'format_version': 1, 'config': cfg, 'model_state_dict': model.state_dict()}, path)
            predictor = ElevationPredictor(path, 'cpu')
            result = predictor.predict(np.zeros((41, 55, 3), np.uint8))
            self.assertEqual(result.shape, (41, 55))
            self.assertTrue(set(np.unique(result)) <= {0, 1})

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data'
            data.mkdir()
            cfg = fixture(data)
            config = root / 'config.json'
            config.write_text(json.dumps(cfg))
            baseline = root / 'baseline'
            resumed = root / 'resumed'

            def run(out, resume=None):
                argv = ['train', '--config', str(config), '--output', str(out)]
                if resume:
                    argv += ['--resume', str(resume)]
                with patch.object(sys, 'argv', argv):
                    train.main()

            with patch.object(train, 'create_model', side_effect=lambda *a, **kw: TinyModel()):
                run(baseline)
                original_save = train.atomic_save

                def interrupt(state, path):
                    original_save(state, path)
                    if state['global_step'] == 1 and state['next_batch'] > 0:
                        raise KeyboardInterrupt()

                with patch.object(train, 'atomic_save', side_effect=interrupt):
                    run(resumed)
                mid = torch.load(resumed / 'last.pt', weights_only=True)
                self.assertEqual(mid['next_batch'], 2)
                run(resumed, resumed / 'last.pt')

            a = torch.load(baseline / 'last.pt', weights_only=True)
            b = torch.load(resumed / 'last.pt', weights_only=True)
            self.assertEqual(a['global_step'], b['global_step'])
            self.assertEqual(b['epoch'], 2)
            for key in a['model_state_dict']:
                torch.testing.assert_close(a['model_state_dict'][key], b['model_state_dict'][key], rtol=0, atol=0)

    def test_weights_loading_with_and_without_head(self):
        class DummyModel(nn.Module):
            def __init__(self, out_channels=1):
                super().__init__()
                self.encoder = nn.Conv2d(3, 8, 3, padding=1)
                self.decoder = nn.Conv2d(8, 4, 3, padding=1)
                self.segmentation_head = nn.Conv2d(4, out_channels, 1)

        source_model = DummyModel(out_channels=1)
        with torch.no_grad():
            source_model.encoder.weight.fill_(1.5)
            source_model.decoder.weight.fill_(2.5)
            source_model.segmentation_head.weight.fill_(3.5)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt_path = tmp_path / 'source.pt'
            torch.save({'model_state_dict': source_model.state_dict()}, ckpt_path)

            # 1. Загрузка со всей головой (same_num_classes=True)
            target1 = DummyModel(out_channels=1)
            train.initialize_weights(target1, str(ckpt_path), same_num_classes=True)
            torch.testing.assert_close(target1.encoder.weight, source_model.encoder.weight)
            torch.testing.assert_close(target1.segmentation_head.weight, source_model.segmentation_head.weight)

            # 2. Загрузка без головы (same_num_classes=False)
            target2 = DummyModel(out_channels=1)
            with torch.no_grad():
                target2.segmentation_head.weight.fill_(0.0)
            train.initialize_weights(target2, str(ckpt_path), same_num_classes=False)
            torch.testing.assert_close(target2.encoder.weight, source_model.encoder.weight)
            torch.testing.assert_close(target2.decoder.weight, source_model.decoder.weight)
            self.assertEqual(float(target2.segmentation_head.weight[0, 0, 0, 0]), 0.0)

            # 3. Несовместимые конфигурации дают понятные ошибки
            invalid_cfg1 = {'resume_training': True, 'pretrained_path': None, 'same_num_classes': True}
            with self.assertRaises(ValueError):
                train.validate_config({**fixture(tmp_path), **invalid_cfg1})

            invalid_cfg2 = {'resume_training': True, 'pretrained_path': str(ckpt_path), 'same_num_classes': False}
            with self.assertRaises(ValueError):
                train.validate_config({**fixture(tmp_path), **invalid_cfg2})

    def test_resume_with_old_reducelronplateau_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = fixture(root)
            ckpt_path = root / 'old_plateau.pt'
            rows = load_records(root)
            _, fingerprint = inspect_dataset(rows)
            model = TinyModel()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            torch.save({
                'format_version': 1,
                'config': fixture(root),
                'dataset_fingerprint': fingerprint,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler': {'mode': 'max', 'patience': 5},
                'scheduler_type': 'ReduceLROnPlateau',
                'epoch': 1,
                'next_batch': 0,
                'global_step': 1,
                'best_iou': 0.5,
                'loss_sum': 1.0,
                'loss_batches': 1
            }, ckpt_path)

            # При resume_training=True должна быть понятная ошибка
            cfg_resume = fixture(root)
            cfg_resume.update(
                pretrained_path=str(ckpt_path),
                resume_training=True,
                same_num_classes=True,
                output_dir=str(root / 'output_resume')
            )
            cfg_path = root / 'config.json'
            cfg_path.write_text(json.dumps(cfg_resume))
            with patch.object(train, 'create_model', side_effect=lambda *a, **kw: TinyModel()):
                with patch.object(sys, 'argv', ['train', '--config', str(cfg_path)]):
                    with self.assertRaises(ValueError) as ctx:
                        train.main()
                    self.assertIn('ReduceLROnPlateau', str(ctx.exception))

    def test_train_main_with_pretrained_checkpoint_modes(self):
        """Проверка запуска нового обучения с временным предобученным чекпоинтом (с головой и без)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / 'data'
            data_dir.mkdir()
            cfg_base = fixture(data_dir)

            source_model = TinyModel()
            with torch.no_grad():
                source_model.encoder.weight.fill_(2.0)
            source_ckpt = root / 'source_ckpt.pt'
            torch.save({
                'format_version': 1,
                'config': cfg_base,
                'model_state_dict': source_model.state_dict(),
            }, source_ckpt)

            # 1. Новое обучение с полной загрузкой модели (same_num_classes=True, resume_training=False)
            out_full = root / 'out_full'
            cfg_full = fixture(data_dir)
            cfg_full.update(
                pretrained_path=str(source_ckpt),
                same_num_classes=True,
                resume_training=False,
                output_dir=str(out_full),
                epochs=1,
            )
            cfg_full_path = root / 'cfg_full.json'
            cfg_full_path.write_text(json.dumps(cfg_full))

            with patch.object(train, 'create_model', side_effect=lambda *a, **kw: TinyModel()):
                with patch.object(sys, 'argv', ['train', '--config', str(cfg_full_path)]):
                    train.main()
            self.assertTrue((out_full / 'last.pt').exists())
            saved_full = torch.load(out_full / 'last.pt', weights_only=True)
            self.assertEqual(saved_full['epoch'], 1)

            # 2. Новое обучение с загрузкой общих слоев (same_num_classes=False, resume_training=False)
            out_shared = root / 'out_shared'
            cfg_shared = fixture(data_dir)
            cfg_shared.update(
                pretrained_path=str(source_ckpt),
                same_num_classes=False,
                resume_training=False,
                output_dir=str(out_shared),
                epochs=1,
            )
            cfg_shared_path = root / 'cfg_shared.json'
            cfg_shared_path.write_text(json.dumps(cfg_shared))

            with patch.object(train, 'create_model', side_effect=lambda *a, **kw: TinyModel()):
                with patch.object(sys, 'argv', ['train', '--config', str(cfg_shared_path)]):
                    train.main()
            self.assertTrue((out_shared / 'last.pt').exists())
            saved_shared = torch.load(out_shared / 'last.pt', weights_only=True)
            self.assertEqual(saved_shared['epoch'], 1)

    def test_augmentations_mask_values_and_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = fixture(root)
            cfg['crop_size'] = 128
            rows = load_records(root)
            dataset = ElevationDataset(rows['train'], cfg, is_train=True)
            for idx in range(min(5, len(dataset))):
                img, mask = dataset[idx]
                self.assertEqual(img.shape, (3, 128, 128))
                self.assertEqual(mask.shape, (128, 128))
                self.assertEqual(img.dtype, torch.float32)
                self.assertEqual(mask.dtype, torch.int64)
                self.assertTrue(0.0 <= img.min() and img.max() <= 1.0)
                mask_unique = set(torch.unique(mask).tolist())
                self.assertTrue(mask_unique.issubset({0, 1, 255}))

    def test_onecycle_and_none_scheduler_progression(self):
        model = nn.Linear(4, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=1e-3, total_steps=10, pct_start=0.3, div_factor=10.0, final_div_factor=100.0
        )
        lrs = []
        for _ in range(10):
            lrs.append(opt.param_groups[0]['lr'])
            opt.step()
            sched.step()
        # В OneCycleLR скорость обучения сначала растет, затем убывает
        self.assertLess(lrs[0], lrs[3])
        self.assertGreater(lrs[3], lrs[-1])


if __name__ == '__main__':
    unittest.main()
