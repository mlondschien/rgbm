rgbm
====

A lightweight, Rust-native gradient boosting machine.

Installation
------------

You can install ``rgbm`` with pip

::

   pip install rgbm


Quick start
-----------

.. code-block:: python

   import polars as pl
   import pyarrow as pa
   import rgbm

   df = pl.read_csv("train.csv")
   features = df.drop("y").to_arrow().to_batches()[0]
   labels = pa.array(df["y"].to_numpy())

   dataset = rgbm.Dataset(features, labels)
   booster = rgbm.Booster(objective="gaussian", num_iterations=100)
   booster.fit(dataset)
   predictions = booster.predict(features)


API Reference
-------------

.. autoclass:: rgbm.Dataset(x, y, weights=None, max_bin=255, min_data_in_bin=3, n_jobs=-1, seed=0)

.. autoclass:: rgbm.Booster(objective="gaussian", num_iterations=100, learning_rate=0.1, max_depth=6, min_sum_hessian_in_leaf=1e-3, min_gain_to_split=0.0, lambda_l1=0.0, lambda_l2=0.0, max_leaves=31, leaf_wise=True, n_jobs=-1)
   :members: fit, predict, model_to_string


.. toctree::
   :maxdepth: 1
   :caption: Other

   GitHub <https://github.com/mlondschien/rgbm>
   changelog
