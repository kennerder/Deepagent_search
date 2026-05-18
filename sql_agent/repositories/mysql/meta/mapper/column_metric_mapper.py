from dataclasses import asdict

from sql_agent.entities.column_metric import ColumnMetric
from sql_agent.models.column_metric import ColumnMetricMySQL


class ColumnMetricMapper:
    @staticmethod
    def to_entity(model: ColumnMetricMySQL) -> ColumnMetric:
        return ColumnMetric(
            column_id=model.column_id,
            metric_id=model.metric_id,
        )

    @staticmethod
    def to_model(entity: ColumnMetric) -> ColumnMetricMySQL:
        return ColumnMetricMySQL(**asdict(entity))
