"""MCP utility functions."""

import json

import union
from google.protobuf.json_format import MessageToJson
from google.protobuf.message import Message
from pydantic import BaseModel


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


def proto_to_json(proto: Message) -> dict:
    return json.loads(MessageToJson(proto))


def list_tasks(remote: union.UnionRemote, project: str, domain: str) -> list[TaskMetadata]:
    from flytekit.models.common import NamedEntityIdentifier

    id = NamedEntityIdentifier(project=project, domain=domain)
    task_models, _ = remote.client.list_tasks_paginated(id, limit=100)
    tasks = [t.to_flyte_idl() for t in task_models]
    return [
        TaskMetadata(
            name=task.id.name,
            description=task.short_description,
            inputs=proto_to_json(task.closure.compiled_task.template.interface.inputs),
            outputs=proto_to_json(task.closure.compiled_task.template.interface.outputs),
        )
        for task in tasks
    ]


def list_workflows(remote: union.UnionRemote, project: str, domain: str) -> list[WorkflowMetadata]:
    from flytekit.models.common import NamedEntityIdentifier

    id = NamedEntityIdentifier(project=project, domain=domain)
    workflow_models, _ = remote.client.list_workflows_paginated(id, limit=100)
    workflows = [w.to_flyte_idl() for w in workflow_models]
    return [
        WorkflowMetadata(
            name=workflow.id.name,
            description=workflow.short_description,
            inputs=proto_to_json(workflow.closure.compiled_workflow.primary.template.interface.inputs),
            outputs=proto_to_json(workflow.closure.compiled_workflow.primary.template.interface.outputs),
        )
        for workflow in workflows
    ]
