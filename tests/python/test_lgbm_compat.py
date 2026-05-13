# Copyright (c) 2026 Malte Londschien
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pyarrow as pa
import lightgbm as lgb
import rgbm


def test_lgbm_serialization_compatibility():
    rng = np.random.default_rng(42)
    n = 1000
    num0 = rng.normal(size=n)
    num1 = rng.normal(size=n)
    cat_strs = rng.choice(["a", "b", "c", "d"], size=n)
    cat_arr = pa.array(cat_strs).dictionary_encode()

    batch = pa.record_batch(
        {
            "num0": pa.array(num0, type=pa.float64()),
            "num1": pa.array(num1, type=pa.float64()),
            "cat0": cat_arr,
        }
    )
    y = (
        num0
        + 2.0 * (cat_strs == "a")
        + 1.0 * (cat_strs == "b")
        + rng.normal(scale=0.1, size=n)
    )

    # rgbm's categorical bin codes equal Arrow's dictionary indices.
    X_lgbm = np.column_stack(
        [num0, num1, cat_arr.indices.to_numpy().astype(np.float64)]
    )

    ds = rgbm.Dataset(
        batch, pa.array(y, type=pa.float64()), max_bin=255, min_data_in_bin=3
    )
    model = rgbm.Booster(objective="gaussian", num_iterations=10, max_depth=3)
    model.fit(ds)
    preds_rgbm = model.predict(batch)

    lgbm_model = lgb.Booster(model_str=model.model_to_string())
    preds_lgbm = lgbm_model.predict(X_lgbm)

    np.testing.assert_allclose(preds_rgbm, preds_lgbm, rtol=1e-7, atol=1e-7)
