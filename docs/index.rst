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

.. autoclass:: rgbm.Dataset

.. autoclass:: rgbm.Booster
   :members: fit, predict, model_to_string


.. toctree::
   :maxdepth: 1
   :caption: Other

   GitHub <https://github.com/mlondschien/rgbm>
   changelog
