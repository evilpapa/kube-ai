文本处理
=================

使用 Ray Data，您可以轻松读取和转换大量文本数据。

本指南向您展示如何：

* :ref:`读取文本文件 <reading-text-files>`
* :ref:`转换文本数据 <transforming-text>`
* :ref:`对文本数据进行推理 <performing-inference-on-text>`
* :ref:`保存文本数据 <saving-text>`

.. _reading-text-files:

读取文本文件
------------------

Ray Data 可以读取文本行和 JSONL。或者，
您可以读取原始二进制文件并手动解码数据。

.. tab-set::

    .. tab-item:: Text lines

        要读取文本行，请调用 :func:`~ray.data.read_text` 。
        Ray Data 为每行文本创建一行。

        .. testcode::

            import ray

            ds = ray.data.read_text("s3://anonymous@ray-example-data/this.txt")

            ds.show(3)

        .. testoutput::

            {'text': 'The Zen of Python, by Tim Peters'}
            {'text': 'Beautiful is better than ugly.'}
            {'text': 'Explicit is better than implicit.'}

    .. tab-item:: JSON Lines

        `JSON Lines <https://jsonlines.org/>`_ 是结构化数据的文本格式。它通常用于一次处理一条记录的数据。

        要读取 JSON Lines 文件，请调用 :func:`~ray.data.read_json`。
        Ray Data 为每个 JSON 对象创建一行。

        .. testcode::

            import ray

            ds = ray.data.read_json("s3://anonymous@ray-example-data/logs.json")

            ds.show(3)

        .. testoutput::

            {'timestamp': datetime.datetime(2022, 2, 8, 15, 43, 41), 'size': 48261360}
            {'timestamp': datetime.datetime(2011, 12, 29, 0, 19, 10), 'size': 519523}
            {'timestamp': datetime.datetime(2028, 9, 9, 5, 6, 7), 'size': 2163626}


    .. tab-item:: Other formats

        要阅读其他文本格式，请调用 :func:`~ray.data.read_binary_files`。然后，调用
        :meth:`~ray.data.Dataset.map` 解码您的数据。

        .. testcode::

            from typing import Any, Dict
            from bs4 import BeautifulSoup
            import ray

            def parse_html(row: Dict[str, Any]) -> Dict[str, Any]:
                html = row["bytes"].decode("utf-8")
                soup = BeautifulSoup(html, features="html.parser")
                return {"text": soup.get_text().strip()}

            ds = (
                ray.data.read_binary_files("s3://anonymous@ray-example-data/index.html")
                .map(parse_html)
            )

            ds.show()

        .. testoutput::

            {'text': 'Batoidea\nBatoidea is a superorder of cartilaginous fishes...'}

有关读取文件的更多信息，请参阅 :ref:`加载数据 <loading_data>`。

.. _transforming-text:

转换文本
-----------------

要转换文本，请在函数或者调用类中实现你自己的转换。然后，
调用 :meth:`Dataset.map() <ray.data.Dataset.map>` 或者
:meth:`Dataset.map_batches() <ray.data.Dataset.map_batches>`。Ray Data 会并发处理你的文本。

.. testcode::

    from typing import Any, Dict
    import ray

    def to_lower(row: Dict[str, Any]) -> Dict[str, Any]:
        row["text"] = row["text"].lower()
        return row

    ds = (
        ray.data.read_text("s3://anonymous@ray-example-data/this.txt")
        .map(to_lower)
    )

    ds.show(3)

.. testoutput::

    {'text': 'the zen of python, by tim peters'}
    {'text': 'beautiful is better than ugly.'}
    {'text': 'explicit is better than implicit.'}

有关转换数据的更多信息，请参阅
:ref:`转换数据 <transforming_data>`。

.. _performing-inference-on-text:

对文本进行推理
----------------------------

要使用预训练模型对文本数据执行推理，请实现一个用于设置和调用模型的可调用类。然后，调用
:meth:`Dataset.map_batches() <ray.data.Dataset.map_batches>`。

.. testcode::

    from typing import Dict

    import numpy as np
    from transformers import pipeline

    import ray

    class TextClassifier:
        def __init__(self):

            self.model = pipeline("text-classification")

        def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
            predictions = self.model(list(batch["text"]))
            batch["label"] = [prediction["label"] for prediction in predictions]
            return batch

    ds = (
        ray.data.read_text("s3://anonymous@ray-example-data/this.txt")
        .map_batches(TextClassifier, compute=ray.data.ActorPoolStrategy(size=2))
    )

    ds.show(3)

.. testoutput::

    {'text': 'The Zen of Python, by Tim Peters', 'label': 'POSITIVE'}
    {'text': 'Beautiful is better than ugly.', 'label': 'POSITIVE'}
    {'text': 'Explicit is better than implicit.', 'label': 'POSITIVE'}

有关执行推理的更多信息，请参阅
:ref:`End-to-end: 离线批量推理 <batch_inference_home>`
和 :ref:`使用 actor 批量转换 <transforming_data_actors>`。

.. _saving-text:

保存文本
-----------

要保存文本，请调用类似的方法 :meth:`~ray.data.Dataset.write_parquet`。 
Ray Data 可以以多种格式保存文本。

要查看支持的文件格式的完整列表，请参阅
:ref:`Input/Output 参考 <input-output>`。

.. testcode::

    import ray

    ds = ray.data.read_text("s3://anonymous@ray-example-data/this.txt")

    ds.write_parquet("local:///tmp/results")

有关保存数据的更多信息，请参阅 :ref:`保存数据 <saving-data>`。
