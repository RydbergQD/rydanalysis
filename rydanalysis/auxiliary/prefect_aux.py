from dataclasses import dataclass, asdict
from prefect import Flow, Parameter, task


@dataclass
class PrefectParams:
    name = ""

    def to_prefect_params(self):
        return {name: Parameter(name, default=value) for name, value in asdict(self).items()}

    def build_flow(self, flow=None):
        @task(name=self.name)
        def collect_params(**kwargs):
            return kwargs

        if flow is None:
            flow = Flow("set " + self.name)
        flow.set_dependencies(
            task=collect_params,
            upstream_tasks=[],
            keyword_tasks=self.to_prefect_params())
        return flow


@dataclass
class PrefectParams:
    name = ""

    @classmethod
    def build_flow(cls, flow=None):
        @task(name=cls.name)
        def collect_params(**kwargs):
            return kwargs

        if flow is None:
            flow = Flow("set " + cls.name)
        for name, field in cls.__dataclass_fields__.items:
            try:
                field.default.build_flow(flow)
            except AttributeError:
                with flow:
                    Parameter(name, default=field.default)