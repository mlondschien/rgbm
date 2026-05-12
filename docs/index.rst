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
   import rgbm

   df = pl.read_csv("train.csv")
   X, y = df.drop("y"), df["y"]

   dataset = rgbm.Dataset(X, y)
   booster = rgbm.Booster(objective="gaussian", num_iterations=100)
   booster.fit(dataset)
   predictions = booster.predict(X)


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api


.. toctree::
   :maxdepth: 1
   :caption: Examples

   Poisson regression with exposure <examples/french_motor_3rd_party_liabilities.ipynb>


.. toctree::
   :maxdepth: 1
   :caption: Other

   GitHub <https://github.com/mlondschien/rgbm>
   changelog
