import pytest
from tensor_cast.performance_model.profiling_database.data_source import (
    DataSourcePerformanceModel,
)


def test_data_source_is_abstract():
    with pytest.raises(TypeError):
        DataSourcePerformanceModel()


def test_data_source_subclass_must_implement_lookup():
    class BadSource(DataSourcePerformanceModel):
        pass

    with pytest.raises(TypeError):
        BadSource()


def test_data_source_store_raises_by_default():
    class ReadOnlySource(DataSourcePerformanceModel):
        def lookup(self, op_invoke_info):
            return None

    source = ReadOnlySource()
    with pytest.raises(NotImplementedError, match="read-only"):
        source.store(None, None)
