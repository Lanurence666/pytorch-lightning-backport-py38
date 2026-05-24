import sys
import os
import time
import json
import traceback
import tempfile
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    DeviceStatsMonitor,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

RESULTS = {}


def record_result(category, test_name, passed, details="", metrics=None):
    key = f"{category}.{test_name}"
    RESULTS[key] = {
        "passed": passed,
        "details": details,
        "metrics": metrics or {},
    }
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {test_name}: {details}")
    if metrics:
        for k, v in metrics.items():
            print(f"         {k}: {v}")


# ============================================================
# 1. Core API Functional Verification
# ============================================================
def test_core_api():
    print("\n" + "=" * 60)
    print("1. Core API Functional Verification")
    print("=" * 60)

    # 1.1 Basic imports
    try:
        from pytorch_lightning import Trainer, LightningModule, LightningDataModule, Callback
        from pytorch_lightning.strategies import SingleDeviceStrategy, DDPStrategy
        from pytorch_lightning.plugins import MixedPrecision
        record_result("core_api", "basic_imports", True, "All core imports successful")
    except Exception as e:
        record_result("core_api", "basic_imports", False, str(e))
        return

    # 1.2 seed_everything
    try:
        pl.seed_everything(42)
        a = torch.rand(5)
        pl.seed_everything(42)
        b = torch.rand(5)
        assert torch.allclose(a, b), "seed_everything not reproducible"
        record_result("core_api", "seed_everything", True, "Reproducible seeding works")
    except Exception as e:
        record_result("core_api", "seed_everything", False, str(e))

    # 1.3 Trainer instantiation
    try:
        trainer = Trainer(
            max_epochs=1,
            accelerator="auto",
            devices=1,
            enable_progress_bar=False,
            logger=False,
        )
        record_result("core_api", "trainer_instantiation", True, f"Trainer created: accelerator={trainer.accelerator}")
    except Exception as e:
        record_result("core_api", "trainer_instantiation", False, str(e))

    # 1.4 LightningModule definition
    try:
        class TestModule(LightningModule):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 2)

            def forward(self, x):
                return self.layer(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                loss = F.cross_entropy(y_hat, y)
                return loss

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=0.001)

        model = TestModule()
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 2), f"Unexpected output shape: {out.shape}"
        record_result("core_api", "lightning_module", True, f"LightningModule works, output shape: {out.shape}")
    except Exception as e:
        record_result("core_api", "lightning_module", False, str(e))

    # 1.5 LightningDataModule
    try:
        class TestDataModule(LightningDataModule):
            def setup(self, stage=None):
                x = torch.randn(100, 10)
                y = torch.randint(0, 2, (100,))
                dataset = TensorDataset(x, y)
                self.train_ds, self.val_ds = random_split(dataset, [80, 20])

            def train_dataloader(self):
                return DataLoader(self.train_ds, batch_size=16)

            def val_dataloader(self):
                return DataLoader(self.val_ds, batch_size=16)

        dm = TestDataModule()
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert len(batch) == 2
        record_result("core_api", "data_module", True, "LightningDataModule works")
    except Exception as e:
        record_result("core_api", "data_module", False, str(e))

    # 1.6 Callbacks
    try:
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3),
            ModelCheckpoint(monitor="val_loss", save_top_k=1),
            LearningRateMonitor(logging_interval="epoch"),
        ]
        record_result("core_api", "callbacks", True, f"All {len(callbacks)} callbacks created")
    except Exception as e:
        record_result("core_api", "callbacks", False, str(e))

    # 1.7 Loggers
    try:
        tmpdir = tempfile.mkdtemp()
        csv_logger = CSVLogger(tmpdir, name="test")
        tb_logger = TensorBoardLogger(tmpdir, name="test_tb")
        record_result("core_api", "loggers", True, "CSVLogger and TensorBoardLogger created")
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception as e:
        record_result("core_api", "loggers", False, str(e))

    # 1.8 NaN handling with custom PyTorch
    try:
        nan_tensor = torch.tensor(float('nan'))
        isnan = torch.isnan(nan_tensor).item()
        full_nan = torch.full((3,), float('nan'))
        all_nan = torch.isnan(full_nan).all().item()
        record_result("core_api", "nan_handling", isnan and all_nan,
                      f"NaN creation: {isnan}, NaN fill: {all_nan}")
    except Exception as e:
        record_result("core_api", "nan_handling", False, str(e))


# ============================================================
# 2. Training & Inference Pipeline Test
# ============================================================
def test_training_inference():
    print("\n" + "=" * 60)
    print("2. Training & Inference Pipeline Test")
    print("=" * 60)

    class SimpleModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 64)
            self.layer2 = nn.Linear(64, 32)
            self.layer3 = nn.Linear(32, 2)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = self.dropout(x)
            x = F.relu(self.layer2(x))
            return self.layer3(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            self.log("train_loss", loss, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", acc, prog_bar=True)
            return loss

        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            self.log("test_loss", loss)
            self.log("test_acc", acc)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)

    class SimpleDataModule(LightningDataModule):
        def setup(self, stage=None):
            pl.seed_everything(42)
            x = torch.randn(500, 10)
            y = torch.randint(0, 2, (500,))
            dataset = TensorDataset(x, y)
            self.train_ds, self.val_ds, self.test_ds = random_split(dataset, [350, 100, 50])

        def train_dataloader(self):
            return DataLoader(self.train_ds, batch_size=32, shuffle=True)

        def val_dataloader(self):
            return DataLoader(self.val_ds, batch_size=32)

        def test_dataloader(self):
            return DataLoader(self.test_ds, batch_size=32)

    # 2.1 CPU Training
    try:
        model = SimpleModel()
        dm = SimpleDataModule()
        tmpdir = tempfile.mkdtemp()

        trainer = Trainer(
            max_epochs=5,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            logger=CSVLogger(tmpdir, name="cpu_test"),
            callbacks=[LearningRateMonitor(logging_interval="epoch")],
        )
        start_time = time.time()
        trainer.fit(model, dm)
        cpu_train_time = time.time() - start_time

        test_result = trainer.test(model, dm)
        cpu_test_acc = test_result[0].get("test_acc", 0)

        record_result("training", "cpu_training", True,
                      f"5 epochs in {cpu_train_time:.2f}s",
                      {"train_time_s": round(cpu_train_time, 2), "test_acc": round(cpu_test_acc, 4)})
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception as e:
        record_result("training", "cpu_training", False, str(e))
        traceback.print_exc()

    # 2.2 GPU Training
    if torch.cuda.is_available():
        try:
            model = SimpleModel()
            dm = SimpleDataModule()
            tmpdir = tempfile.mkdtemp()

            trainer = Trainer(
                max_epochs=5,
                accelerator="gpu",
                devices=1,
                enable_progress_bar=False,
                logger=CSVLogger(tmpdir, name="gpu_test"),
            )
            start_time = time.time()
            trainer.fit(model, dm)
            gpu_train_time = time.time() - start_time

            test_result = trainer.test(model, dm)
            gpu_test_acc = test_result[0].get("test_acc", 0)

            record_result("training", "gpu_training", True,
                          f"5 epochs in {gpu_train_time:.2f}s",
                          {"train_time_s": round(gpu_train_time, 2), "test_acc": round(gpu_test_acc, 4)})
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception as e:
            record_result("training", "gpu_training", False, str(e))
            traceback.print_exc()
    else:
        record_result("training", "gpu_training", False, "CUDA not available")

    # 2.3 Inference/Prediction
    try:
        model = SimpleModel()
        dm = SimpleDataModule()
        dm.setup()

        trainer = Trainer(
            max_epochs=2,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
        )
        trainer.fit(model, dm)

        predictions = trainer.predict(model, dm.test_dataloader())
        total_preds = sum(len(p) for p in predictions)
        record_result("training", "inference_predict", True,
                      f"Predicted {total_preds} samples",
                      {"num_predictions": total_preds})
    except Exception as e:
        record_result("training", "inference_predict", False, str(e))
        traceback.print_exc()

    # 2.4 Checkpoint save/load
    try:
        model = SimpleModel()
        dm = SimpleDataModule()
        tmpdir = tempfile.mkdtemp()

        trainer = Trainer(
            max_epochs=3,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
            callbacks=[ModelCheckpoint(dirpath=tmpdir, save_last=True)],
        )
        trainer.fit(model, dm)

        ckpt_path = os.path.join(tmpdir, "last.ckpt")
        assert os.path.exists(ckpt_path), "Checkpoint not found"

        loaded_model = SimpleModel.load_from_checkpoint(ckpt_path)
        record_result("training", "checkpoint_save_load", True,
                      f"Checkpoint saved and loaded: {ckpt_path}")
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception as e:
        record_result("training", "checkpoint_save_load", False, str(e))
        traceback.print_exc()


# ============================================================
# 3. Performance Metrics Quantitative Comparison
# ============================================================
def test_performance_metrics():
    print("\n" + "=" * 60)
    print("3. Performance Metrics Quantitative Comparison")
    print("=" * 60)

    class PerfModel(LightningModule):
        def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            return self.net(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            self.log("val_loss", loss)
            self.log("val_acc", acc)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)

    # Generate larger synthetic dataset
    pl.seed_everything(42)
    x = torch.randn(5000, 784)
    y = torch.randint(0, 10, (5000,))
    dataset = TensorDataset(x, y)
    train_ds, val_ds = random_split(dataset, [4000, 1000])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # 3.1 CPU performance
    try:
        model = PerfModel()
        trainer = Trainer(
            max_epochs=10,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
        )

        start_time = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        elapsed = time.time() - start_time

        record_result("performance", "cpu_speed", True,
                      f"10 epochs in {elapsed:.2f}s",
                      {"total_time_s": round(elapsed, 2),
                       "epochs_per_s": round(10 / elapsed, 3),
                       "samples_per_s": round(40000 / elapsed, 1)})
    except Exception as e:
        record_result("performance", "cpu_speed", False, str(e))

    # 3.2 GPU performance & memory
    if torch.cuda.is_available():
        try:
            model = PerfModel()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated() / 1024**2

            trainer = Trainer(
                max_epochs=10,
                accelerator="gpu",
                devices=1,
                enable_progress_bar=False,
                logger=False,
            )

            start_time = time.time()
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            elapsed = time.time() - start_time

            mem_peak = torch.cuda.max_memory_allocated() / 1024**2
            mem_after = torch.cuda.memory_allocated() / 1024**2

            record_result("performance", "gpu_speed_memory", True,
                          f"10 epochs in {elapsed:.2f}s, peak GPU mem: {mem_peak:.1f}MB",
                          {"total_time_s": round(elapsed, 2),
                           "epochs_per_s": round(10 / elapsed, 3),
                           "samples_per_s": round(40000 / elapsed, 1),
                           "gpu_peak_mem_mb": round(mem_peak, 1),
                           "gpu_after_mem_mb": round(mem_after, 1)})
        except Exception as e:
            record_result("performance", "gpu_speed_memory", False, str(e))
    else:
        record_result("performance", "gpu_speed_memory", False, "CUDA not available")

    # 3.3 GPU utilization check
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_mem / 1024**3
            record_result("performance", "gpu_info", True,
                          f"{gpu_name}, {gpu_mem_total:.1f}GB total",
                          {"gpu_name": gpu_name, "gpu_total_mem_gb": round(gpu_mem_total, 1)})
        except Exception as e:
            record_result("performance", "gpu_info", False, str(e))


# ============================================================
# 4. Special Scenario Compatibility Tests
# ============================================================
def test_special_scenarios():
    print("\n" + "=" * 60)
    print("4. Special Scenario Compatibility Tests")
    print("=" * 60)

    class SpecialModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 2)

        def forward(self, x):
            return self.layer(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            self.log("val_loss", loss)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
            return [optimizer], [scheduler]

    pl.seed_everything(42)
    x = torch.randn(200, 10)
    y = torch.randint(0, 2, (200,))
    dataset = TensorDataset(x, y)
    train_ds, val_ds = random_split(dataset, [160, 40])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    # 4.1 Mixed Precision Training (16-bit)
    if torch.cuda.is_available():
        try:
            model = SpecialModel()
            trainer = Trainer(
                max_epochs=3,
                accelerator="gpu",
                devices=1,
                precision="16-mixed",
                enable_progress_bar=False,
                logger=False,
            )
            start_time = time.time()
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            elapsed = time.time() - start_time
            record_result("special", "mixed_precision_16", True,
                          f"FP16 training completed in {elapsed:.2f}s",
                          {"train_time_s": round(elapsed, 2)})
        except Exception as e:
            record_result("special", "mixed_precision_16", False, str(e))
            traceback.print_exc()
    else:
        record_result("special", "mixed_precision_16", False, "CUDA not available")

    # 4.2 Mixed Precision Training (bf16)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        try:
            model = SpecialModel()
            trainer = Trainer(
                max_epochs=3,
                accelerator="gpu",
                devices=1,
                precision="bf16-mixed",
                enable_progress_bar=False,
                logger=False,
            )
            start_time = time.time()
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            elapsed = time.time() - start_time
            record_result("special", "mixed_precision_bf16", True,
                          f"BF16 training completed in {elapsed:.2f}s",
                          {"train_time_s": round(elapsed, 2)})
        except Exception as e:
            record_result("special", "mixed_precision_bf16", False, str(e))
    else:
        record_result("special", "mixed_precision_bf16", False, "CUDA/BF16 not available")

    # 4.3 Double Precision (64-bit)
    try:
        model = SpecialModel().double()
        x_d = x.double()
        y_d = y
        ds = TensorDataset(x_d, y_d)
        train_d, val_d = random_split(ds, [160, 40])
        tl = DataLoader(train_d, batch_size=16, shuffle=True)
        vl = DataLoader(val_d, batch_size=16)

        trainer = Trainer(
            max_epochs=2,
            accelerator="cpu",
            devices=1,
            precision="64-true",
            enable_progress_bar=False,
            logger=False,
        )
        trainer.fit(model, train_dataloaders=tl, val_dataloaders=vl)
        record_result("special", "double_precision", True, "FP64 training completed")
    except Exception as e:
        record_result("special", "double_precision", False, str(e))
        traceback.print_exc()

    # 4.4 Gradient Accumulation
    try:
        model = SpecialModel()
        trainer = Trainer(
            max_epochs=3,
            accelerator="cpu",
            devices=1,
            accumulate_grad_batches=4,
            enable_progress_bar=False,
            logger=False,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        record_result("special", "gradient_accumulation", True, "Gradient accumulation (4 batches) works")
    except Exception as e:
        record_result("special", "gradient_accumulation", False, str(e))
        traceback.print_exc()

    # 4.5 Learning Rate Scheduler
    try:
        model = SpecialModel()
        tmpdir = tempfile.mkdtemp()
        trainer = Trainer(
            max_epochs=5,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
            callbacks=[LearningRateMonitor(logging_interval="epoch")],
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        record_result("special", "lr_scheduler", True, "LR scheduler with StepLR works")
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception as e:
        record_result("special", "lr_scheduler", False, str(e))
        traceback.print_exc()

    # 4.6 Early Stopping
    try:
        model = SpecialModel()
        tmpdir = tempfile.mkdtemp()
        trainer = Trainer(
            max_epochs=50,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            logger=CSVLogger(tmpdir, name="early_stop"),
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=3, mode="min"),
                ModelCheckpoint(monitor="val_loss", save_top_k=1, dirpath=tmpdir),
            ],
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        stopped_epoch = trainer.current_epoch
        record_result("special", "early_stopping", stopped_epoch < 50,
                      f"Stopped at epoch {stopped_epoch + 1}/50",
                      {"stopped_epoch": stopped_epoch + 1})
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception as e:
        record_result("special", "early_stopping", False, str(e))
        traceback.print_exc()

    # 4.7 DeviceStatsMonitor (GPU)
    if torch.cuda.is_available():
        try:
            model = SpecialModel()
            trainer = Trainer(
                max_epochs=2,
                accelerator="gpu",
                devices=1,
                enable_progress_bar=False,
                logger=False,
                callbacks=[DeviceStatsMonitor()],
            )
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            record_result("special", "device_stats_monitor", True, "DeviceStatsMonitor works on GPU")
        except Exception as e:
            record_result("special", "device_stats_monitor", False, str(e))
    else:
        record_result("special", "device_stats_monitor", False, "CUDA not available")

    # 4.8 Fabric API
    try:
        from lightning_fabric import Fabric
        fabric = Fabric(accelerator="cpu", devices=1)
        fabric.launch()

        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model, optimizer = fabric.setup(model, optimizer)

        x = torch.randn(16, 10)
        y = torch.randint(0, 2, (16,))
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        record_result("special", "fabric_api", True, "Lightning Fabric API works")
    except Exception as e:
        record_result("special", "fabric_api", False, str(e))
        traceback.print_exc()


# ============================================================
# 5. PyTorch Version Compatibility Check
# ============================================================
def test_pytorch_compat():
    print("\n" + "=" * 60)
    print("5. PyTorch Version Compatibility Check")
    print("=" * 60)

    record_result("compat", "torch_version", True,
                  f"torch {torch.__version__}",
                  {"version": torch.__version__})
    record_result("compat", "pl_version", True,
                  f"pytorch_lightning {pl.__version__}",
                  {"version": pl.__version__})
    record_result("compat", "python_version", True,
                  f"Python {sys.version}",
                  {"version": sys.version})
    record_result("compat", "cuda_available", torch.cuda.is_available(),
                  f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        record_result("compat", "cuda_version", True,
                      f"CUDA {torch.version.cuda}",
                      {"version": torch.version.cuda})
        record_result("compat", "cudnn_version", True,
                      f"cuDNN {torch.backends.cudnn.version()}",
                      {"version": str(torch.backends.cudnn.version())})

    # Check torch.compile availability
    has_compile = hasattr(torch, "compile")
    record_result("compat", "torch_compile", has_compile,
                  f"torch.compile available: {has_compile}")

    # Check torch.cuda.amp availability
    has_amp = hasattr(torch.cuda, "amp")
    record_result("compat", "torch_amp", has_amp,
                  f"torch.cuda.amp available: {has_amp}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch Lightning + Custom PyTorch 2.13 Comprehensive Test")
    print("=" * 60)

    test_core_api()
    test_training_inference()
    test_performance_metrics()
    test_special_scenarios()
    test_pytorch_compat()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    total = len(RESULTS)
    passed = sum(1 for v in RESULTS.values() if v["passed"])
    failed = total - passed
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Pass Rate: {passed/total*100:.1f}%")

    if failed > 0:
        print("\nFailed tests:")
        for k, v in RESULTS.items():
            if not v["passed"]:
                print(f"  - {k}: {v['details']}")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "pl_test_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
