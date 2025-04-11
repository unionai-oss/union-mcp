import json

import union
from flytekit.models.common import NamedEntityIdentifier
from google.protobuf.json_format import MessageToJson
from google.protobuf.message import Message
from pydantic import BaseModel


remote = union.UnionRemote(
    default_project="mcp-testing",
    default_domain="development",
)

id = NamedEntityIdentifier(
    project="mcp-testing",
    domain="development",
)

class TaskMetadata(BaseModel):
    name: str
    description: str
    inputs: dict
    outputs: dict


class WorkflowMetadata(BaseModel):
    name: str
    description: str
    inputs: dict
    outputs: dict


def _proto_to_json(proto: Message) -> dict:
    return json.loads(MessageToJson(proto))


def list_tasks() -> list[TaskMetadata]:
    task_models, _ = remote.client.list_tasks_paginated(id, limit=1000)
    tasks = [t.to_flyte_idl() for t in task_models]
    return [
        TaskMetadata(
            name=task.id.name,
            description=task.short_description,
            inputs=_proto_to_json(task.closure.compiled_task.template.interface.inputs),
            outputs=_proto_to_json(task.closure.compiled_task.template.interface.outputs),
        )
        for task in tasks
    ]


def list_workflows() -> list[WorkflowMetadata]:
    workflow_models, _ = remote.client.list_workflows_paginated(id, limit=1000)
    workflows = [w.to_flyte_idl() for w in workflow_models]
    return [
        WorkflowMetadata(
            name=workflow.id.name,
            description=workflow.short_description,
            inputs=_proto_to_json(workflow.closure.compiled_workflow.primary.template.interface.inputs),
            outputs=_proto_to_json(workflow.closure.compiled_workflow.primary.template.interface.outputs),
        )
        for workflow in workflows
    ]

if __name__ == "__main__":
    tasks = list_tasks()
    workflows = list_workflows()

    print(tasks)
    print(workflows)

    import ipdb; ipdb.set_trace()
