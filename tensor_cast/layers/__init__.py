from .parallel_embedding import ParallelEmbedding
from .parallel_linear import ColumnParallelLinear, RowParallelLinear

COLWISE_LINEAR = "colwise"
ROWWISE_LINEAR = "rowwise"
PARALLEL_EMBEDDING = "prallel_embedding"

PARALLEL_MODULE_CLS = {
    COLWISE_LINEAR: ColumnParallelLinear,
    ROWWISE_LINEAR: RowParallelLinear,
    PARALLEL_EMBEDDING: ParallelEmbedding,
}
